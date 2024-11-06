from .base import Base


class BasicTokenizer:
    def __init__(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # for decode
        self.merges = {}  # for encode

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255

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

        return bytes.decode("utf-8", errors="replace")
