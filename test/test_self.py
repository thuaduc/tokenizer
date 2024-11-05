import pytest
import tiktoken
import os

from src import BasicTokenizer, RegexTokenizer, GPT4Tokenizer


def read_text_from_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


@pytest.fixture
def tokenizer():
    # Setup: Create a BasicTokenizer instance and train it with sample text
    filename = "test/taylorswift.txt"
    train_text = read_text_from_file(filename)  # Adjust as necessary
    tokenizer = BasicTokenizer()
    tokenizer.train(train_text, 300)
    return tokenizer


def test_encode_decode_identity(tokenizer):
    sample = "something blue and black"

    # Encoding the sample text
    encoded = tokenizer.encode(sample)

    # Decoding the previously encoded text
    decoded = tokenizer.decode(encoded)

    # Check that the decoded text matches the original sample
    assert decoded == sample

    # Additionally, print out the byte representation of the sample
    print("Byte representation of sample:", list(sample.encode("utf-8")))
    print("Encoded representation:", encoded)
    print("Decoded representation:", decoded)
