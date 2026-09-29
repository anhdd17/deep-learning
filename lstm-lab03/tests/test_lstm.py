import torch
import torch.nn as nn
from src.lstm import MyLSTM, MyLSTMCell


def _copy_from_torch(my: MyLSTM, ref: nn.LSTM):
    """Copy weights từ nn.LSTM sang MyLSTM. Chú ý transpose vì layout khác."""
    with torch.no_grad():
        my.cell.W_x.copy_(ref.weight_ih_l0.T)              # (4H, I) → (I, 4H)
        my.cell.W_h.copy_(ref.weight_hh_l0.T)              # (4H, H) → (H, 4H)
        my.cell.b.copy_(ref.bias_ih_l0 + ref.bias_hh_l0)  # PyTorch có 2 bias → cộng lại


def test_matches_torch_lstm():
    torch.manual_seed(0)
    B, T, I, H = 4, 13, 8, 16
    ref = nn.LSTM(I, H, batch_first=True)
    my  = MyLSTM(I, H)
    _copy_from_torch(my, ref)

    x = torch.randn(B, T, I)
    out_ref, (h_ref, c_ref) = ref(x)
    out_my,  (h_my,  c_my)  = my(x)

    assert torch.allclose(out_my, out_ref, atol=1e-5), \
        f"outputs differ: max_diff={( out_my - out_ref).abs().max():.2e}"
    assert torch.allclose(h_my, h_ref[0], atol=1e-5), \
        f"h_n differs: max_diff={(h_my - h_ref[0]).abs().max():.2e}"
    assert torch.allclose(c_my, c_ref[0], atol=1e-5), \
        f"c_n differs: max_diff={(c_my - c_ref[0]).abs().max():.2e}"


def test_gradients_match_torch_lstm():
    torch.manual_seed(0)
    B, T, I, H = 3, 7, 5, 6
    ref = nn.LSTM(I, H, batch_first=True)
    my  = MyLSTM(I, H)
    _copy_from_torch(my, ref)

    x1 = torch.randn(B, T, I, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    ref(x1)[0].sum().backward()
    my(x2)[0].sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5), \
        f"grad differs: max_diff={(x1.grad - x2.grad).abs().max():.2e}"


def test_forget_bias_init():
    H  = 8
    my = MyLSTM(4, H, forget_bias=1.0)
    b  = my.cell.b.detach()
    assert torch.allclose(b[H : 2 * H], torch.ones(H)), \
        "forget gate bias phải = 1.0"
    assert torch.allclose(b[:H], torch.zeros(H)), \
        "input gate bias phải = 0"
    assert torch.allclose(b[2 * H :], torch.zeros(2 * H)), \
        "g và o gate bias phải = 0"


def test_output_shape():
    B, T, I, H = 4, 20, 8, 32
    my      = MyLSTM(I, H)
    x       = torch.randn(B, T, I)
    out, (h, c) = my(x)
    assert out.shape == (B, T, H), f"outputs shape sai: {out.shape}"
    assert h.shape   == (B, H),    f"h_n shape sai: {h.shape}"
    assert c.shape   == (B, H),    f"c_n shape sai: {c.shape}"


def test_return_gates_shape():
    B, T, I, H = 2, 10, 6, 16
    my = MyLSTM(I, H)
    x  = torch.randn(B, T, I)
    _, _, gates = my(x, return_gates=True)
    for k in ("i", "f", "g", "o"):
        assert k in gates, f"thiếu gate '{k}'"
        assert gates[k].shape == (B, T, H), \
            f"gate '{k}' shape sai: {gates[k].shape}"


def test_masking_stops_update():
    """Sau khi hết token thật, h và c phải giữ nguyên."""
    torch.manual_seed(7)
    B, T, I, H = 2, 10, 4, 8
    my      = MyLSTM(I, H)
    x       = torch.randn(B, T, I)
    # sample 0 dài 3, sample 1 dài 10
    lengths = torch.tensor([3, 10])
    _, (h_masked, _) = my(x, lengths=lengths)

    # Chạy lại không có mask, chỉ lấy 3 bước đầu cho sample 0
    _, (h_full, _) = my(x[:1, :3, :])   # (1, 3, I) không có padding

    assert torch.allclose(h_masked[0], h_full[0], atol=1e-6), \
        "Masking không đúng: h của sample bị mask khác với h tính chính xác"
