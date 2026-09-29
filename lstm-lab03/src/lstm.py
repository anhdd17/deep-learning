import math
import torch
import torch.nn as nn


class MyLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, forget_bias: float = 1.0):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        # Layout: W_x (I, 4H), W_h (H, 4H) — giữ convention x @ W như MySimpleRNN
        self.W_x = nn.Parameter(torch.empty(input_size,  4 * hidden_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.b   = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters(forget_bias)

    def reset_parameters(self, forget_bias: float):
        H   = self.hidden_size
        std = 1.0 / math.sqrt(H)
        nn.init.uniform_(self.W_x, -std, std)
        nn.init.uniform_(self.W_h, -std, std)
        nn.init.zeros_(self.b)
        # Forget gate = chunk thứ 2 (thứ tự i, f, g, o) → b[H:2H]
        with torch.no_grad():
            self.b[H : 2 * H].fill_(forget_bias)

    def forward(self, x_t: torch.Tensor, state: tuple) -> tuple:
        # x_t: (B, I),  state = (h_prev, c_prev) mỗi cái (B, H)
        h_prev, c_prev = state
        gates = x_t @ self.W_x + h_prev @ self.W_h + self.b  # (B, 4H)

        # Thứ tự bắt buộc i, f, g, o — khớp với PyTorch để test so sánh hoạt động
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        g = torch.tanh(g)      # candidate cell
        o = torch.sigmoid(o)   # output gate

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t, {"i": i, "f": f, "g": g, "o": o}


class MyLSTM(nn.Module):
    """Unroll MyLSTMCell theo thời gian. Input batch_first: (B, T, I)."""

    def __init__(self, input_size: int, hidden_size: int, forget_bias: float = 1.0):
        super().__init__()
        self.cell        = MyLSTMCell(input_size, hidden_size, forget_bias)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None,
                return_gates: bool = False):
        """
        x:       (B, T, I)
        lengths: (B,) số token thật — None = không padding
        returns:
            outputs: (B, T, H)
            (h_n, c_n): mỗi cái (B, H) — trạng thái tại token thật cuối
            gates (nếu return_gates=True): dict "i"|"f"|"g"|"o" → (B, T, H)
        """
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_size)
        c = x.new_zeros(B, self.hidden_size)

        all_h      = []
        gate_lists = {"i": [], "f": [], "g": [], "o": []} if return_gates else None

        for t in range(T):
            h_new, c_new, gates = self.cell(x[:, t, :], (h, c))

            if lengths is not None:
                # Giữ nguyên h, c khi sample đã hết token thật
                mask = (t < lengths).float().unsqueeze(1).to(x.device)  # (B, 1)
                h = mask * h_new + (1 - mask) * h
                c = mask * c_new + (1 - mask) * c
            else:
                h, c = h_new, c_new

            all_h.append(h)
            if return_gates:
                for k in gate_lists:
                    gate_lists[k].append(gates[k])

        outputs = torch.stack(all_h, dim=1)  # (B, T, H)

        if return_gates:
            gates_out = {k: torch.stack(v, dim=1) for k, v in gate_lists.items()}
            return outputs, (h, c), gates_out

        return outputs, (h, c)
