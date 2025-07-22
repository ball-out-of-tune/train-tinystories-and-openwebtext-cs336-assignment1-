import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import time
import logging
import json
from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)


def _get_bytes_tuple_from_str(input_str) -> tuple[bytes]:
    return tuple(
        # NOTE: Some utf-8 characters are represented by mutitple bytes
        # so need to split them further
        byte.to_bytes()
        for token in input_str
        for byte in token.encode("utf-8")
    )


def _split_str_with_special_tokens(input_str, special_tokens, keep_special_tokens=False):
    # NOTE: Need to escape the existing '|' in the special token before split
    # If keep_tokens, the special tokens themselves will be kept in the resulting list
    if keep_special_tokens:
        return re.split("(" + "|".join([re.escape(special_token) for special_token in special_tokens]) + ")", input_str)
    else:
        return re.split("|".join([re.escape(special_token) for special_token in special_tokens]), input_str)


@dataclass
class PretokenRecord:
    idx: int
    pretokens: tuple[bytes]
    count: int


PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizerTrainer:
    """
    A byte-level BPE model, each pre-token is represented as a sequence of UTF-8 bytes.
    """

    _PRETOKEN_NUM_PROCESSES = 10
    _EOT_TOKEN = "<|endoftext|>"

    def __init__(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
    ):
        # Vocabulary initialization
        # One-to-one mapping from bytestring token to integer ID
        self.input_path = input_path
        self.special_tokens = special_tokens
        self.merged = []
        # Count number of each pretokens
        self.pretoken_records = self._create_pretoken_records()
        # Keep track of the pretokens that contain the byte pair for fast update
        self.byte_pair_counts, self.byte_pair_refs = self._create_bytes_pair_counts_and_refs()
        # Initial vocabulary
        self.vocab = [s.encode("utf-8") for s in special_tokens] + [i.to_bytes() for i in range(256)]

        a_time = time.time()
        while len(self.vocab) < vocab_size:
            self._merge()
        logger.info(f"Time taken for merge: {time.time() - a_time}")

    def _create_bytes_pair_counts_and_refs(self):
        byte_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        byte_pair_refs: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
        # Count byte pairs from the pretokens
        for record in self.pretoken_records:
            pretokens, occurance, record_idx = record.pretokens, record.count, record.idx
            for i in range(len(pretokens) - 1):
                byte_pair = (pretokens[i], pretokens[i + 1])
                # Increment the count by pretoken occurances
                byte_pair_counts[byte_pair] += occurance
                byte_pair_refs[byte_pair].add(record_idx)
        return byte_pair_counts, byte_pair_refs

    def _process_merge(self, pretoken_record, merge_pair):
        pretokens, occurance, record_idx = pretoken_record.pretokens, pretoken_record.count, pretoken_record.idx
        new_pretokens = []
        i = 0
        while i < len(pretokens):
            if i == len(pretokens) - 1:
                # Only 1 char left, just append
                new_pretokens.append(pretokens[i])
                break

            byte_pair = (pretokens[i], pretokens[i + 1])
            if byte_pair == merge_pair:
                logger.debug(f"Find mergable pair {merge_pair} from {pretokens}")
                # Found matched pair.
                # 1. Combine the byte pair as a new token
                merged_token = byte_pair[0] + byte_pair[1]
                new_pretokens.append(merged_token)

                # 2. Update prev/ next byte pair count
                # HACK: The mapping form self.byte_pair_refs to the current pretoken record might still exist
                # even if it might no longer have the pair after merging. Ths will cause unnecessary lookup.
                if i - 1 >= 0:
                    prev_byte_pair = (pretokens[i - 1], pretokens[i])
                    self.byte_pair_counts[prev_byte_pair] -= occurance
                    # Add new pair
                    new_prev_byte_pair = (pretokens[i - 1], merged_token)
                    self.byte_pair_counts[new_prev_byte_pair] += occurance
                    self.byte_pair_refs[new_prev_byte_pair].add(record_idx)
                if i + 2 < len(pretokens):
                    next_byte_pair = (pretokens[i + 1], pretokens[i + 2])
                    self.byte_pair_counts[next_byte_pair] -= occurance
                    # Add new pair
                    new_next_byte_pair = (merged_token, pretokens[i + 2])
                    self.byte_pair_counts[new_next_byte_pair] += occurance
                    self.byte_pair_refs[new_next_byte_pair].add(record_idx)
                i += 2
            else:
                # Not match. Move one step.
                new_pretokens.append(pretokens[i])
                i += 1

        # Return new pretoken
        return tuple(new_pretokens)

    def _merge(self):
        # Select the most frequent pairs
        merge_pair = max((count, pair) for pair, count in self.byte_pair_counts.items())[1]
        self.merged.append(merge_pair)
        self.vocab.append(merge_pair[0] + merge_pair[1])

        # Update the pretokens that were referred by the byte pair
        for record_idx in self.byte_pair_refs[merge_pair]:
            # NOTE: A reference to the record that will be updated in place
            record = self.pretoken_records[record_idx]
            # Process merge and update byte_pair_counts for the prev/ next
            # byte pairs that overlap with the current pair
            new_pretokens = self._process_merge(record, merge_pair)
            # Update record to the new pretokens after merge
            logger.debug(f"Previous pretoken {record.pretokens}")
            record.pretokens = new_pretokens
            logger.debug(f"Pretoken updated to {record.pretokens}")

        # No need to track the counts and reference of the merged pair
        del self.byte_pair_counts[merge_pair]
        del self.byte_pair_refs[merge_pair]

    def _get_pretokens_subcounts(self, chunk):
        """
        Get pretoken counts from a chunk of corpus
        """
        subcounts = defaultdict(int)
        matches = re.finditer(PRETOKEN_PATTERN, chunk)
        for match in matches:
            # Represent each pretoken as a tuple of utf-8 encoded bytes
            pretokens: tuple[bytes] = _get_bytes_tuple_from_str(match.group())
            subcounts[pretokens] += 1
        return subcounts

    def _create_pretoken_records(self):
        counts = {}
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self._PRETOKEN_NUM_PROCESSES, self._EOT_TOKEN.encode("utf-8"))

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            all_chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")

                # Removing special tokens before pre-tokenization
                # and treat the splitted chunks as separated chunks
                sub_chunks = _split_str_with_special_tokens(chunk, self.special_tokens)
                for c in sub_chunks:
                    all_chunks.append(c)

            # Count occurances of each pretoken
            with ProcessPoolExecutor() as executor:
                all_subcounts = list(executor.map(self._get_pretokens_subcounts, all_chunks))

            for subcounts in all_subcounts:
                for pretokens, subcount in subcounts.items():
                    if pretokens not in counts:
                        counts[pretokens] = subcount
                    else:
                        counts[pretokens] += subcount

        records: list[PretokenRecord] = []
        for idx, (pretokens, count) in enumerate(counts.items()):
            records.append(PretokenRecord(idx=idx, pretokens=pretokens, count=count))
        return records

    def get_vocab(self) -> dict[int, bytes]:
        return {idx: v for idx, v in enumerate(self.vocab)}

    def get_merges(self) -> list[tuple[bytes, bytes]]:
        return self.merged


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_rev: dict[bytes, int] = {v: idx for idx, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        # NOTE: Sort the special tokens to favor longer ones
        self.special_tokens.sort(key=lambda x: len(x), reverse=True)

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r+") as f:
            vocab: dict[int, bytes] = {idx: v.encode("utf-8") for v, idx in json.load(f).items()}
        with open(merges_filepath, "rb") as f:
            merges: list[tuple[bytes, bytes]] = [tuple(a.strip().split(b" ")) for a in f.readlines()]
        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    def _tokenize_per_pretokens(self, pretokens: tuple[bytes]) -> list[int]:
        if len(pretokens) == 1 and pretokens[0] in self.special_tokens:
            # Return index of the special token directly without byte merging
            return [self.vocab_rev[pretokens]]

        # Fast look-up at the location of each byte pairs
        byte_pairs_loc = {(pretokens[i], pretokens[i+1]): i for i in range(len(pretokens) - 1)}
        
        # Store intermediate pretokens after each round of merging
        new_pretokens = []
        
        # Step 1: Merge the byte pairs iteratively
        logger.debug(f"Start tokenize {pretokens=}")
        while True:
            merge_bytes_loc = None
            for merge_pair in self.merges:
                if merge_pair in byte_pairs_loc:
                    merge_bytes_loc = byte_pairs_loc[merge_pair]
                    break
            if merge_bytes_loc is None:
                # No bytes to merge from the current pretokens
                break
            else:
                i = 0
                while i < len(pretokens):
                    if i == merge_bytes_loc:
                        new_pretokens.append(pretokens[i] + pretokens[i+1])
                        i += 2
                    else:
                        new_pretokens.append(pretokens[i])
                        i += 1

            # Finish merging for a round
            pretokens = tuple(new_pretokens)
            new_pretokens = []
            # Re-build the byte pairs look-up dict
            byte_pairs_loc = {(pretokens[i], pretokens[i+1]): i for i in range(len(pretokens) - 1)}

        # Step 2: Cast bytes sequenc to int indices
        logger.debug(f"Final pretokens {pretokens=}")
        bytes_idx_arr = []
        for byte in pretokens:
            bytes_idx_arr.append(self.vocab_rev[byte])
        return bytes_idx_arr

    def _tokenize(self, text: str):
        """
        Pretokenize the whole text
        """
        chunks = _split_str_with_special_tokens(text, self.special_tokens, keep_special_tokens=True) if self.special_tokens else [text]
        logger.debug(f"Tokenize {chunks}")
        all_pretokens = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                # NOTE: Keep the special token as is in a tuple without pretokenization
                all_pretokens.append((chunk.encode("utf-8"),))
                continue
            matches = re.finditer(PRETOKEN_PATTERN, chunk)
            for match in matches:
                # Represent each pretoken as a tuple of utf-8 encoded bytes
                all_pretokens.append(_get_bytes_tuple_from_str(match.group()))

        logger.debug(f"All pretokens: {all_pretokens=}")
        # Merge bytes for each pretoken
        # merged_pretokens = self._tokenize_per_pretokens(all_pretokens)
        with ProcessPoolExecutor() as executor:
            merged_pretokens = list(executor.map(self._tokenize_per_pretokens, all_pretokens))

        logger.debug(f"Merged pretokens: {merged_pretokens}")
        return [byte for pretokens in merged_pretokens for byte in pretokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        results = []
        for input_str in iterable:
            results.extend(self.encode(input_str))
        return results

    def encode(self, text: str) -> list[int]:
        return self._tokenize(text)

    def decode(self, ids: list[int]) -> str:
        result_bytes = b"".join([self.vocab[id] for id in ids])
        logger.debug(f"Decode {ids=} into {result_bytes=}")
        return result_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tokenizer_trainer = BPETokenizerTrainer(vocab_size=10, input_path="mytest.txt", special_tokens=["<|endoftext|>"])
