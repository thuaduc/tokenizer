import pytest
from src import BasicTokenizer, RegexTokenizer  # Adjust imports as necessary


def read_text_from_file(filename):
    """Read the entire content of a text file."""
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


# Define a list of tokenizers to be used in the tests
TOKENIZERS = [
    (BasicTokenizer, "BasicTokenizer"),
    (RegexTokenizer, "RegexTokenizer"),
]


@pytest.fixture(params=TOKENIZERS)
def tokenizer(request):
    """Fixture to initialize tokenizers with training text."""
    tokenizer_class, tokenizer_name = request.param
    filename = "test/taylorswift.txt"
    train_text = read_text_from_file(filename)
    tokenizer_instance = tokenizer_class()
    tokenizer_instance.train(train_text, 300)
    return tokenizer_instance, tokenizer_name


def test_encode_decode_identity(tokenizer):
    """Test that encoding and then decoding returns the original string using different tokenizers."""
    tokenizer_instance, tokenizer_name = tokenizer
    sample = "hello world!!!? (안녕하세요!) lol123 😉"

    # Encoding the sample text
    encoded = tokenizer_instance.encode(sample)

    # Decoding the previously encoded text
    decoded = tokenizer_instance.decode(encoded)

    # Print the encoded and decoded outputs
    print(f"Encoded representation ({tokenizer_name}):", encoded)
    print(f"Normal representation ({tokenizer_name}):", list(sample.encode("utf-8")))

    # Assert that the decoded text matches the original sample
    assert decoded == sample, f"Expected '{sample}', but got '{decoded}'."


def test_wiki():
    tokenizer_instance = BasicTokenizer()
    text = "aaabdaaabac"
    tokenizer_instance.train(text, 256 + 3)
    ids = tokenizer_instance.encode(text)
    print(ids)
    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer_instance.decode(ids) == text
