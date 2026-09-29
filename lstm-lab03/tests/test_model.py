import torch
from src.rnn   import MySimpleRNN
from src.model import SentimentRNN   # backward-compat alias

VOCAB_SIZE = 100
EMBED_DIM  = 16
HIDDEN_DIM = 32
BATCH      = 8
SEQ_LEN    = 20


def test_rnn_output_shape():
    rnn = MySimpleRNN(input_size=EMBED_DIM, hidden_size=HIDDEN_DIM)
    x   = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    h_T = rnn(x)
    assert h_T.shape == (BATCH, HIDDEN_DIM)


def test_rnn_h0_is_zeros():
    """h_0 phải là zeros — không leak state giữa các batch."""
    rnn = MySimpleRNN(input_size=EMBED_DIM, hidden_size=HIDDEN_DIM)
    x1  = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    x2  = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    h1  = rnn(x1)
    h2  = rnn(x2)
    assert not torch.allclose(h1, h2)


def test_rnn_backward():
    rnn  = MySimpleRNN(input_size=EMBED_DIM, hidden_size=HIDDEN_DIM)
    x    = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    loss = rnn(x).sum()
    loss.backward()
    assert rnn.W_xh.grad is not None
    assert rnn.W_hh.grad is not None
    assert rnn.b.grad   is not None


def test_model_output_shape_custom():
    model = SentimentRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, rnn_type="custom")
    x     = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logit = model(x)
    assert logit.shape == (BATCH,)


def test_model_output_shape_pytorch():
    model = SentimentRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, rnn_type="pytorch")
    x     = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logit = model(x)
    assert logit.shape == (BATCH,)


def test_model_logit_range():
    """Logit chưa qua sigmoid — sigmoid(logit) phải nằm trong (0, 1)."""
    model = SentimentRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, rnn_type="custom")
    x     = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logit = model(x)
    prob  = torch.sigmoid(logit)
    assert (prob > 0).all() and (prob < 1).all()


def test_padding_ignored():
    """Token PAD (id=0) có embedding bằng vector 0 (padding_idx=0)."""
    model   = SentimentRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, rnn_type="custom")
    pad_vec = model.embedding(torch.tensor([0]))
    assert torch.allclose(pad_vec, torch.zeros_like(pad_vec))
