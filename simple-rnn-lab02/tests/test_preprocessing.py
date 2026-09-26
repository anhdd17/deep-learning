from src.preprocessing import normalize, tokenize
from src.vocabulary import Vocabulary, PAD_ID, UNK_ID


def test_normalize_lowercase():
    assert normalize("I Love This MOVIE") == "i love this movie"

def test_normalize_strips_html():
    assert normalize("<br />Great film<br/>") == "great film"

def test_normalize_removes_punctuation():
    assert normalize("wow!!! amazing...") == "wow amazing"

def test_tokenize_basic():
    assert tokenize("I love this movie") == ["i", "love", "this", "movie"]

def test_tokenize_empty():
    assert tokenize("") == []


# --- Vocabulary ---

def _make_vocab():
    vocab = Vocabulary()
    corpus = [["i", "love", "this", "movie"], ["i", "hate", "this", "film"]]
    vocab.build(corpus, vocab_size=100, min_freq=1)
    return vocab

def test_vocab_contains_special_tokens():
    vocab = _make_vocab()
    assert "<PAD>" in vocab.token2id
    assert "<UNK>" in vocab.token2id

def test_vocab_known_token():
    vocab = _make_vocab()
    ids = vocab.encode(["i", "love"], max_seq_len=5)
    assert ids[0] != UNK_ID
    assert ids[1] != UNK_ID

def test_vocab_unknown_token():
    vocab = _make_vocab()
    ids = vocab.encode(["unknown_word_xyz"], max_seq_len=5)
    assert ids[0] == UNK_ID

def test_vocab_padding():
    vocab = _make_vocab()
    ids = vocab.encode(["i"], max_seq_len=5)
    assert len(ids) == 5
    assert ids[1] == PAD_ID
    assert ids[4] == PAD_ID

def test_vocab_truncation():
    vocab = _make_vocab()
    ids = vocab.encode(["i", "love", "this", "movie"], max_seq_len=2)
    assert len(ids) == 2
