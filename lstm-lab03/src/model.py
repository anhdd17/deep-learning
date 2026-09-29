import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.rnn  import MySimpleRNN
from src.lstm import MyLSTM
from src.logger import get_logger

logger = get_logger(__name__)


class SentimentModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 rnn_type: str = "lstm", impl: str = "custom",
                 bidirectional: bool = False, num_layers: int = 1,
                 dropout: float = 0.0, forget_bias: float = 1.0,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn_type      = rnn_type
        self.impl          = impl
        self.bidirectional = bidirectional
        self.num_layers    = num_layers
        num_dir            = 2 if bidirectional else 1

        if impl == "custom":
            if rnn_type == "lstm":
                self.rnn = MyLSTM(embed_dim, hidden_size, forget_bias=forget_bias)
            elif rnn_type == "rnn":
                self.rnn = MySimpleRNN(embed_dim, hidden_size)
            else:
                raise ValueError(f"custom impl không hỗ trợ rnn_type='{rnn_type}'")
        elif impl == "pytorch":
            kwargs = dict(
                input_size   = embed_dim,
                hidden_size  = hidden_size,
                num_layers   = num_layers,
                batch_first  = True,
                bidirectional = bidirectional,
                dropout      = dropout if num_layers > 1 else 0.0,
            )
            if rnn_type == "lstm":
                self.rnn = nn.LSTM(**kwargs)
            elif rnn_type == "rnn":
                self.rnn = nn.RNN(**kwargs)
            elif rnn_type == "gru":
                self.rnn = nn.GRU(**kwargs)
            else:
                raise ValueError(f"rnn_type không hợp lệ: '{rnn_type}'")
        else:
            raise ValueError(f"impl phải là 'custom' hoặc 'pytorch', got '{impl}'")

        self.fc = nn.Linear(hidden_size * num_dir, 1)
        logger.info(
            "SentimentModel | rnn_type=%s impl=%s hidden=%d bi=%s layers=%d",
            rnn_type, impl, hidden_size, bidirectional, num_layers,
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # input_ids: (B, T),  lengths: (B,) optional
        emb    = self.embedding(input_ids)   # (B, T, E)
        h_last = self._run_rnn(emb, lengths) # (B, H * num_dir)
        return self.fc(h_last).squeeze(1)    # (B,) logits

    def forward_from_embeddings(self, emb: torch.Tensor,
                                lengths: torch.Tensor = None) -> torch.Tensor:
        """Nhận embeddings trực tiếp — dùng cho gradient_probe."""
        return self.fc(self._run_rnn(emb, lengths)).squeeze(1)

    def _run_rnn(self, emb: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        if self.impl == "custom":
            if self.rnn_type == "lstm":
                _, (h, _) = self.rnn(emb, lengths)
                return h
            else:
                return self.rnn(emb, lengths)

        # pytorch impl
        if lengths is not None:
            packed = pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            if self.rnn_type == "lstm":
                _, (h_n, _) = self.rnn(packed)
            else:
                _, h_n = self.rnn(packed)
        else:
            if self.rnn_type == "lstm":
                _, (h_n, _) = self.rnn(emb)
            else:
                _, h_n = self.rnn(emb)

        # h_n: (num_layers * num_dir, B, H)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
        return h_last


def build_model(cfg: dict) -> SentimentModel:
    m = cfg["model"]
    return SentimentModel(
        vocab_size    = cfg["data"]["vocab_size"],
        embed_dim     = m["embed_dim"],
        hidden_size   = m["hidden_size"],
        rnn_type      = m.get("type", "lstm"),
        impl          = m.get("impl", "custom"),
        bidirectional = m.get("bidirectional", False),
        num_layers    = m.get("num_layers", 1),
        dropout       = m.get("dropout", 0.0),
        forget_bias   = m.get("forget_bias", 1.0),
    )


class SentimentRNN(SentimentModel):
    """Backward-compat alias giữ nguyên API của Lab 1 (24 test cũ vẫn pass)."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 rnn_type: str = "custom"):
        # Lab 1: rnn_type="custom" | "pytorch" → map sang (rnn_type, impl) mới
        if rnn_type == "custom":
            super().__init__(vocab_size, embed_dim, hidden_dim,
                             rnn_type="rnn", impl="custom")
        else:
            super().__init__(vocab_size, embed_dim, hidden_dim,
                             rnn_type="rnn", impl="pytorch")
