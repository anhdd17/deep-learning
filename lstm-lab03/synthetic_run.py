"""
Task 6 — Synthetic long-range memory task.

So sánh khả năng nhớ xa của RNN, LSTM (forget_bias=1), LSTM (forget_bias=0), GRU
trên FirstTokenDataset với L ∈ {10, 25, 50, 100, 200, 500}.
"""
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.synthetic import FirstTokenDataset
from src.model import SentimentModel
from src.logger import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Hyper-params (spec: embed=16, hidden=32, 20 epochs, lr=1e-3, clip=1.0)
# ---------------------------------------------------------------------------
VOCAB_SIZE  = 103   # PAD=0, NEG=1, POS=2, noise=3..102
EMBED_DIM   = 16
HIDDEN_SIZE = 32
EPOCHS      = 20
LR          = 1e-3
CLIP        = 1.0
BATCH_SIZE  = 64
N_TRAIN     = 10_000
N_VAL       = 2_000
SEED        = 42

SEQ_LENS = [10, 25, 50, 100, 200, 500]

CONFIGS = [
    {"name": "RNN",              "rnn_type": "rnn",  "impl": "custom",  "forget_bias": 1.0},
    {"name": "LSTM (fb=1)",      "rnn_type": "lstm", "impl": "custom",  "forget_bias": 1.0},
    {"name": "LSTM (fb=0)",      "rnn_type": "lstm", "impl": "custom",  "forget_bias": 0.0},
    {"name": "GRU (pytorch)",    "rnn_type": "gru",  "impl": "pytorch", "forget_bias": 1.0},
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one(model, train_loader, val_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for _ in range(EPOCHS):
        model.train()
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
                preds = (torch.sigmoid(model(x, lengths)) >= 0.5).float()
                correct += (preds == y).sum().item()
                total   += len(y)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


def run():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # results[config_name][seq_len] = val_acc
    results = {c["name"]: {} for c in CONFIGS}

    for seq_len in SEQ_LENS:
        set_seed(SEED)
        train_loader = DataLoader(
            FirstTokenDataset(N_TRAIN, seq_len, seed=SEED),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = DataLoader(
            FirstTokenDataset(N_VAL, seq_len, seed=SEED + 1),
            batch_size=BATCH_SIZE,
        )

        for cfg in CONFIGS:
            set_seed(SEED)
            model = SentimentModel(
                vocab_size  = VOCAB_SIZE,
                embed_dim   = EMBED_DIM,
                hidden_size = HIDDEN_SIZE,
                rnn_type    = cfg["rnn_type"],
                impl        = cfg["impl"],
                forget_bias = cfg["forget_bias"],
            ).to(device)

            acc = train_one(model, train_loader, val_loader, device)
            results[cfg["name"]][seq_len] = acc
            logger.info("L=%4d | %-18s | val_acc=%.4f", seq_len, cfg["name"], acc)

    # --- Print table ---
    header = f"{'L':>6} | " + " | ".join(f"{c['name']:^18}" for c in CONFIGS)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for seq_len in SEQ_LENS:
        row = f"{seq_len:>6} | "
        row += " | ".join(
            f"{results[c['name']][seq_len]:^18.4f}" for c in CONFIGS
        )
        print(row)
    print("=" * len(header))

    # --- Plot ---
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg in CONFIGS:
            accs = [results[cfg["name"]][L] for L in SEQ_LENS]
            ax.plot(SEQ_LENS, accs, marker="o", label=cfg["name"])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
        ax.set_xscale("log")
        ax.set_xlabel("Sequence length (L)")
        ax.set_ylabel("Val accuracy")
        ax.set_title("Long-range memory: val accuracy vs sequence length")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = "artifacts/synthetic_results.png"
        fig.savefig(out_path, dpi=150)
        logger.info("Plot saved → %s", out_path)
        plt.close(fig)
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")

    return results


if __name__ == "__main__":
    run()
