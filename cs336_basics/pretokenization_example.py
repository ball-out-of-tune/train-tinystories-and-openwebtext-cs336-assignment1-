import os
from typing import BinaryIO
import regex as re

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

## Usage
NUM_PROCESSES = 10
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COUNTS = {}
SPECIAL_TOKENS = ["<|endoftext|>"]

if __name__ == '__main__':
    with open("mytest.txt", "rb") as f:
        boundaries = find_chunk_boundaries(
            f, NUM_PROCESSES, "<|endoftext|>".encode("utf-8"))
        print(boundaries)
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        all_chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Removing special tokens before pre-tokenization
            # and treat the splitted chunks as separated chunks
            sub_chunks = re.split(
                "|".join([re.escape(special_token) for special_token in SPECIAL_TOKENS]
            ), chunk)
            for c in sub_chunks:
                if c != '':
                    all_chunks.append(c)
            
        for chunk in all_chunks:
            matches = re.finditer(PATTERN, chunk)
            print(chunk)
            for match in matches:
                pretoken = tuple(token.encode('utf-8') for token in match.group())
                if pretoken not in COUNTS:
                    COUNTS[pretoken] = 1
                else:
                    COUNTS[pretoken] += 1
                    
        print(COUNTS)