"""
PAGE 9 PROBLEM A AND B
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

from mpmath import j0
from networkx.algorithms.assortativity import pairs
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import regex as re
from multiprocessing import Pool
from collections import Counter, defaultdict
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def count_pretokens_in_chunks(args):
    path, start, end, special_pattern = args
    word_counts = Counter()
    with open(path, "rb") as f:
        f.seek(start)
        chunk_content = f.read(end - start).decode("utf-8", errors="replace")
    fragments = re.split(special_pattern, chunk_content)
    # tokenization: map each word to a tuple of bytes and count it all up
    for frag in fragments:
        for tok in re.findall(PAT, frag):
            byte_tuple = tuple(tok.encode("utf-8"))
            word_counts[byte_tuple] += 1
    return word_counts



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # The BPE Algorithm (naive):
    #   get mapping {word : frequency}
    #   for num_merges times:
    #       for every word:
    #           for every pair in that word:
    #               pair_counts[pair] += word's frequency
    #   
    #       most_frequent_pair = find_most_frequent_pair(pair_counts)
    #       
    #       update vocab to include this new word + tokenID
    #       
    #       update word_counts() -> for every word, if the word contains our most frequent pair, build the new byte-sequence and add to word_counts



    # vocab == 256 bytes + special tokens
    vocab = {i: bytes([i]) for i in range(256)}          # you have your normal bytes...
    for i in range(len(special_tokens)):
        special_token = special_tokens[i]
        vocab[256+i] = special_token.encode("utf-8")     # then each special token is represented as a SEQUENCE of bytes, not just bytes 0-255.
    

    # pretokenizer & special token patterns
    SPECIAL_SPLIT_PATTERN = re.compile("|".join(re.escape(t) for t in special_tokens))

    # read corpus and split on special tokens
    word_counts = Counter[Any]()        # { byte_sequence : frequency }
    ############################
    # SERIAL IMPLEMENTATION
    # with open(input_path, "rb") as f:
    #     content = f.read().decode("utf-8")                                    # open and read the file in UTF-8 bytes
    # fragments = re.split(SPECIAL_SPLIT_PATTERN, content)                      # .split() on special tokens
    # for frag in fragments:                                                    # for every fragment...
    #     for tok in re.findall(PAT, frag):                                         # for every token we regex out...
    #         byte_tuple = tuple(tok.encode("utf-8"))                                   # convert token -> UTF-8 byte sequence
    #         word_counts[byte_tuple] += 1                                              # word_counts[byte sequence]++
    ############################
    ############################
    # PARALLEL IMPLEMENTATION
    num_processes = 4
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")                                                           # split text file into 4 chunks
    tasks = [(input_path, start, end, SPECIAL_SPLIT_PATTERN) for start, end in zip(boundaries[:-1], boundaries[1:])]    # define start, end for the 4 chunks
    with Pool(processes=num_processes) as pool:
        results = pool.map(count_pretokens_in_chunks, tasks)                                                                             # use pool.map to give each chunk to an independent process
    for local_map in results:
        word_counts.update(local_map)                                                                                                    # each process produces a local_map counter, which we can then use with .update() to update the global counter (word_counts)
    ############################

    # at this point in the code, word_counts is a frequency map of all the byte-sequences
    # word_counts == {byte_sequence (word) : freq}

    # Now we implement the merging logic:
    merges = []
    num_merges = vocab_size - (256 + len(special_tokens))   # merge until we reach max vocab size

    ############################
    # SERIAL IMPLEMENTATION
    # for i in range(num_merges):
    #     for byte_tuple, freq in word_counts.items():
    #         for pair in zip(byte_tuple, byte_tuple[1:]):                                  # for every word: for every pair: update pair_counts
    #             if pair not in pair_counts:
    #                 pair_counts[pair] = 0
    #             pair_counts[pair] += freq

    #     most_frequent_pair = max(pair_counts.keys(), 
    #        key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]))          # find the most frequent pair and mint a new token ID
    #     new_token_id = 256 + len(special_tokens) + i

    #     merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))
    #     vocab[new_token_id] = vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]]         # update vocab to include this new token

    #     # update word_counts with newly merged tokens
    #     new_word_counts = {}
    #     # for every word, merge the relevant 2 bytes in those tuples                                              # for word, check if it has this pair
    #     for byte_tuple, freq in word_counts.items():
    #         new_word = []
    #         idx = 0
    #         while idx < len(byte_tuple):
    #             if idx < len(byte_tuple) - 1 and (byte_tuple[idx], byte_tuple[idx+1]) == most_frequent_pair:      # if so, construct a new byte sequence w/ this new byte
    #                 new_word.append(new_token_id)
    #                 idx += 2
    #             else:
    #                 new_word.append(byte_tuple[idx])                                                              # if not, copy over the byte like normal
    #                 idx += 1 
    #         if tuple(new_word) not in new_word_counts:
    #             new_word_counts[tuple(new_word)] = 0
    #         new_word_counts[tuple(new_word)] += freq                                                               # we've construted a new_token! Add this to the counts
    #     word_counts = new_word_counts 
    ############################
    ############################
    # PARALLEL IMPLEMENTATION
    pair_counts = Counter()                     # pair_counts = [(pair1, pair2) : freq]
    pair_to_words = defaultdict(set)       # pair_to_words = [(pair1, pair2) : [list_of_words_containing_this_pair]]

    # IMPORTANT OPTIMIZATION: maintain mapping of (pair) -> [all words containing this pair]. 
    # This avoids us having to iterate over ALL words and checking if they contain the most frequent pair: we just lookup from this dict all the relevant words containing this pair

    for word, freq in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    # Repeat merging a bunch of times...
    for i in range(num_merges):
        if not pair_counts:
            break
        
        # Find the most frequent pair
        most_frequent_pair = max(pair_counts.keys(), key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]))      # (tok_id_1, tok_id_2)
        new_token_id = 256 + len(special_tokens) + i

        # record the merge
        merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))

        # update the vocab to have the new token ID & byte-sequence
        vocab[new_token_id] = vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]]

        # OPTIMIZATION: get only words containing this specific pair
        words_to_process = list(pair_to_words[most_frequent_pair])

        # remove the old most-frequent-pair from our cache - this pair is going to be its own word now!
        del pair_counts[most_frequent_pair]
        del pair_to_words[most_frequent_pair]

        # NOTE: we're iterating over words_to_process, which is a lot smaller than the global word_counts!
        for word in words_to_process:
            # We much (1) remove pair frequencies of this word from the cache, (2) build the new word, (3) add the new word/pairs back to the cache
            freq = word_counts[word]
            del word_counts[word]       # (0) remove word from overall count

            # (1) remove pair frequencies of this word
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                if pair != most_frequent_pair:
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_to_words[pair].discard(word)

            # (2) build new word
            new_word = []
            j = 0
            while j < len(word):
                # iterate through the word: either add existing char, or add the new token ID
                if j < len(word) - 1 and word[j] == most_frequent_pair[0] and word[j+1] == most_frequent_pair[1]:
                    new_word.append(new_token_id)
                    j += 2  # merged w next byte
                else:
                    new_word.append(word[j])
                    j += 1
            new_word_tuple = tuple(new_word)

            # (3) add all new pairs back to the cache. Basically look at all pairs in this NEW WORD we've just constructed, and add the frequencies back to our cache
            for j in range(len(new_word_tuple) - 1):
                pair = (new_word_tuple[j], new_word_tuple[j + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(new_word_tuple)
            
            word_counts[new_word_tuple] += freq # add count of new word back to word_counts
        ############################

    return vocab, merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


import os
import pickle
import time
import sys

def save_bpe_results(vocab, merges, folder="logs"):
    # Create the directory if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")

    # Save Vocab
    vocab_path = os.path.join(folder, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    
    # Save Merges
    merges_path = os.path.join(folder, "merges.pkl")
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"Successfully saved to {vocab_path} and {merges_path}")

def profile_and_save(input_path, target_vocab_size, special_tokens):
    start_time = time.perf_counter()
    
    print(f"Starting BPE training on {os.path.basename(input_path)}...")
    
    # Run the training
    vocab, merges = run_train_bpe(input_path, target_vocab_size, special_tokens)
    
    duration = time.perf_counter() - start_time
    
    # Print basic stats
    print("-" * 30)
    print(f"Time: {duration:.2f}s | Vocab: {len(vocab)} | Merges: {len(merges)}")
    
    # Save to disk
    save_bpe_results(vocab, merges)
    
    return vocab, merges

def profile_bpe_training(input_path, target_vocab_size, special_tokens):
    start_time = time.perf_counter()
    
    print(f"Starting BPE training on {os.path.basename(input_path)}...")
    
    # Run your training function
    vocab, merges = run_train_bpe(input_path, target_vocab_size, special_tokens)
    
    duration = time.perf_counter() - start_time
    
    # Calculate sizes
    # vocab: dict of {int: bytes}
    # merges: list of tuples (bytes, bytes)
    vocab_len = len(vocab)
    merges_len = len(merges)
    
    # Finding the longest token
    longest_token_bytes = max(vocab.values(), key=len)
    
    print("-" * 30)
    print(f"Time Taken:    {duration:.2f} seconds")
    print(f"Vocab Count:   {vocab_len} items")
    print(f"Merges Count:  {merges_len} items")
    print(f"Vocab Memory:  {sys.getsizeof(vocab) / 1024:.2f} KB (Object only)")
    print(f"Merges Memory: {sys.getsizeof(merges) / 1024:.2f} KB (Object only)")
    print("-" * 30)
    print(f"Longest Token Length: {len(longest_token_bytes)} bytes")
    print(f"Longest Token Content: {longest_token_bytes.decode('utf-8', errors='replace')}")
    
    return vocab, merges

if __name__ == "__main__":
    # Ensure this path is correct for your local machine
    input_path = '/Users/willychan/Desktop/classes/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    
    # Run, profile, and save
    # vocab, merges = profile_and_save(input_path, 10000, ['<|endoftext|>'])
    vocab, merges = profile_bpe_training(input_path, 10000, ['<|endoftext|>'])