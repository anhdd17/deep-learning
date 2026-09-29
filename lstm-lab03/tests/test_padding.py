import pytest
import torch
import torch.nn.functional as F
from src.model import build_model

COMBOS = [
    ("rnn",  "custom"),
    ("lstm", "custom"),
    ("rnn",  "pytorch"),
    ("lstm", "pytorch"),
    ("gru",  "pytorch"),
]

def _make_cfg(rnn_type, impl):
    return {
        "data":  {"vocab_size": 200, "max_len": 500},
        "model": {
            "type": rnn_type, "impl": impl,
            "embed_dim": 16, "hidden_size": 32,
            "bidirectional": False, "num_layers": 1,
            "dropout": 0.0, "forget_bias": 1.0,
        },
    }


@pytest.mark.parametrize("rnn_type,impl", COMBOS)
def test_padding_invariance(rnn_type, impl):
    """Cùng review phải cho cùng logit dù pad đến 50 hay 500."""
    torch.manual_seed(0)
    model   = build_model(_make_cfg(rnn_type, impl)).eval()
    ids     = torch.tensor([[5, 17, 42, 9, 3]])
    lengths = torch.tensor([5])
    short   = F.pad(ids, (0, 45))   # pad đến 50
    long    = F.pad(ids, (0, 495))  # pad đến 500

    with torch.no_grad():
        a = model(short, lengths)
        b = model(long,  lengths)

    assert torch.allclose(a, b, atol=1e-5), (
        f"{rnn_type}/{impl}: logit thay đổi khi pad "
        f"({a.item():.6f} vs {b.item():.6f}) — kiểm tra lại masking"
    )
