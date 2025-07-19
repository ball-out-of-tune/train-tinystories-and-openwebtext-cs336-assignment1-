import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class PretokenRecord:
    idx: int
    pretokens: tuple[bytes]
    count: int


class BPETokenizer:
    """
    A byte-level BPE model, each pre-token is represented as a sequence of UTF-8 bytes.
    """

    _PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
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
                # NOTE: The mapping form self.byte_pair_refs
                # to the current pretoken record might still exist
                # even if it might no longer have the pair after merging
                if i - 1 >= 0:
                    prev_byte_pair = (pretokens[i-1], pretokens[i])
                    self.byte_pair_counts[prev_byte_pair] -= occurance
                    # Add new pair
                    new_prev_byte_pair = (pretokens[i-1], merged_token)
                    self.byte_pair_counts[new_prev_byte_pair] += occurance
                    self.byte_pair_refs[new_prev_byte_pair].add(record_idx)
                if i + 2 < len(pretokens):
                    next_byte_pair = (pretokens[i+1], pretokens[i+2])
                    self.byte_pair_counts[next_byte_pair] -= occurance
                    # Add new pair
                    new_next_byte_pair = (merged_token, pretokens[i+2])
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

    def _get_pretokens(self, chunk):
        """
        Get pretoken counts from a chunk of corpus
        """
        subcounts = defaultdict(int)
        matches = re.finditer(self._PRETOKEN_PATTERN, chunk)
        for match in matches:
            # Represent each pretoken as a tuple of utf-8 encoded bytes
            pretokens = tuple(
                # NOTE: Some utf-8 characters are represented by mutitple bytes
                # so need to split them further
                byte.to_bytes()
                for token in match.group()
                for byte in token.encode("utf-8")
            )
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
                # NOTE: Need to escape the existing '|' in the special token before split
                sub_chunks = re.split(
                    "|".join([re.escape(special_token) for special_token in self.special_tokens]), chunk
                )
                for c in sub_chunks:
                    all_chunks.append(c)

            # Count occurances of each pretoken
            with ProcessPoolExecutor() as executor:
                all_subcounts = list(executor.map(self._get_pretokens, all_chunks))

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tokenizer = BPETokenizer(vocab_size=10, input_path="mytest.txt", special_tokens=["<|endoftext|>"])
