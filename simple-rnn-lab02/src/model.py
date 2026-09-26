import torch
import torch.nn as nn
from src.rnn import MySimpleRNN
from src.logger import get_logger

logger = get_logger(__name__)


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, rnn_type: str = "custom"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if rnn_type == "custom":
            self.rnn = MySimpleRNN(embed_dim, hidden_dim)
        elif rnn_type == "pytorch":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"rnn_type must be 'custom' or 'pytorch', got '{rnn_type}'")

        self.rnn_type = rnn_type
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)

        if self.rnn_type == "custom":
            h_T = self.rnn(emb)          # (batch, hidden_dim)
        else:
            _, h_n = self.rnn(emb)       # h_n: (num_layers=1, batch, hidden_dim)
            h_T = h_n.squeeze(0)         # (batch, hidden_dim)

        logit = self.fc(h_T).squeeze(1)  # (batch,)
        return logit
