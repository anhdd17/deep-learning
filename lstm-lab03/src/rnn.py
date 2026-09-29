import torch
import torch.nn as nn
from src.logger import get_logger

logger = get_logger(__name__)


class MySimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b    = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, I)
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)

        for t in range(T):
            h_new = torch.tanh(x[:, t, :] @ self.W_xh + h @ self.W_hh + self.b)
            if lengths is not None:
                # mask=1 khi sample còn token thật tại bước t, =0 khi đã vào vùng PAD
                mask = (t < lengths).float().unsqueeze(1).to(x.device)  # (B, 1)
                h = mask * h_new + (1 - mask) * h
            else:
                h = h_new

        return h  # (B, H) — trạng thái tại token thật cuối của mỗi sample
