import regex as re
from typing import Iterable, Iterator
import pickle

# pretokenization pattern
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer:
    # Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. 
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab  # id to bytes
        self.merges = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)   # user defined special tokens, sort by LENGTH
        else:
            self.special_tokens = []

        self.bytes_to_id = {value: key for key, value in vocab.items()}  # bytes to token ID

        if self.special_tokens:
            self.special_pattern = re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")  # capture, not eliminate
        else:
            self.special_pattern = None

    
    # Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
    # same format that your BPE training code output) and (optionally) a list of special tokens.
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> None:

        # ??? SHOULD we be using pickle???
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)


    # Encode an input text into a sequence of token IDs.
    def encode(
        self, 
        text: str
    ) -> list[int]:
        # final sequence of token IDs
        final_ids = []

        if self.special_pattern:
            chunks = re.split(self.special_pattern, text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk: continue

            if chunk in self.special_tokens:
                special_bytes = chunk.encode("utf-8")
                final_ids.append(self.bytes_to_id[special_bytes])
                continue

            pretokens = re.findall(PAT, chunk)  # pretokenizer split string

            for tok in pretokens:
                byte_sequence = [bytes([b]) for b in tok.encode("utf-8")]

                # apply merges to this sequence of bytes based on the rules we got
                for part_1, part_2 in self.merges:

                    # construct a new word based on forward-looking merge rules
                    new_word_bytes = []
                    i = 0
                    while i < len(byte_sequence):
                        if i < len(byte_sequence) - 1 and byte_sequence[i] == part_1 and byte_sequence[i+1] == part_2:
                            new_word_bytes.append(part_1 + part_2)
                            i += 2
                        else:
                            new_word_bytes.append(byte_sequence[i])
                            i += 1
                    byte_sequence = new_word_bytes
            
                # convert byte sequence into token IDs
                for newbyte in byte_sequence:
                    final_ids.append(self.bytes_to_id[newbyte])
        return final_ids

    # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
    # required for memory-efficient tokenization of large files that we cannot directly load into memory.
    def encode_iterable(
        self, 
        iterable: Iterable[str]
    ) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)       # "lazily" yield IDs one by one

    # Decode a sequence of token IDs into text.
    def decode(
        self, 
        ids: list[int]
    ) -> str:

        list_of_byteseqs = [self.vocab[tid] for tid in ids]

        res = b"".join(list_of_byteseqs).decode("utf-8", errors="replace")

        return res