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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        batch, seq_len, _ = x.shape

        h = torch.zeros(batch, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]                                    # (batch, input_size)
            h   = torch.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b)  # (batch, hidden_size)

        # h bây giờ là h_T — hidden state của timestep cuối cùng
        return h
