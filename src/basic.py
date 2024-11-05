from .base import Base


class BasicTokenizer:
    def __init__(self):
        self.vocab = []
        self.num_merge = 0
        pass

    def train(self, text, vocab_size, verbose=False):
        tokens = list(text.encode("utf-8"))
        stats = Base.get_stats(tokens)

        self.vocab = sorted(stats.items(), key=lambda item: item[1], reverse=True)

        while self.num_merge < vocab_size - 256:
            tokens = Base.merge(
                tokens, self.vocab[self.num_merge][0], 256 + self.num_merge
            )
            self.num_merge += 1

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for i in range(self.num_merge):
            tokens = Base.merge(tokens, self.vocab[i][0], 256 + i)

        return tokens

    def decode(self, ids):
        newIds = []
        for i in ids:
            if i > 255:
                newIds.append(self.vocab[i - 255 - 1][0][0])
                newIds.append(self.vocab[i - 255 - 1][0][1])

            else:
                newIds.append(i)

        print(newIds)

        return bytes(list(newIds)).decode()
