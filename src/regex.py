from .base import Base
import regex as re

GPT4_SPLIT_PATTERN = """'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer:
    def __init__(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # for decode
        self.merges = {}
        self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)

    def tokenize(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        tokens = []
        for ch in text_chunks:
            tokens += list(ch.encode("utf-8"))
        return tokens

    def train(self, text, vocab_size, verbose=False):

        assert vocab_size >= 256
        num_merges = vocab_size - 256

        ids = self.tokenize(text)

        for i in range(num_merges):
            stats = Base.get_stats(ids)
            sub = max(stats, key=stats.get)

            idx = 256 + i
            ids = Base.merge(ids, sub, idx)

            self.merges[sub] = idx
            self.vocab[idx] = self.vocab[sub[0]] + self.vocab[sub[1]]

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = Base.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break  # nothing else can be merged anymore

            idx = self.merges[pair]
            ids = Base.merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        bytes = b"".join(self.vocab[id] for id in ids)

        return bytes.decode("utf-8")
