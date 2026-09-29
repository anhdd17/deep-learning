"""
Task 5 — Vanishing gradient experiment.

Vẽ ‖∂L/∂e_t‖ theo khoảng cách từ token cuối cho:
  - RNN  (untrained)
  - RNN  (after 1 epoch)
  - LSTM (untrained)
  - LSTM (after 1 epoch)
"""
import torch
import torch.nn as nn
import random
import numpy as np

from src.config import load_config
from src.logger import setup_logging, get_logger
from src.vocabulary import Vocabulary
from src.dataset import load_csv, split_data, make_loaders
from src.model import build_model
from src.train import run_epoch
from src.gradient_probe import collect_grad_profile

setup_logging("INFO")
logger = get_logger(__name__)

SEED        = 42
MIN_LENGTH  = 200
MAX_BATCHES = 20
MAX_POS     = 200


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_and_profile(cfg, vocab, val_loader, device, train_loader=None):
    """Trả về (untrained_profile, trained_profile) cho config đã cho."""
    set_seed(SEED)
    model = build_model(cfg).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    # --- untrained ---
    untrained = collect_grad_profile(
        model, val_loader, loss_fn, device,
        min_length=MIN_LENGTH, max_batches=MAX_BATCHES, max_positions=MAX_POS,
    )

    # --- after 1 epoch ---
    if train_loader is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        run_epoch(model, train_loader, loss_fn, optimizer, clip=5.0, device=device)

    trained = collect_grad_profile(
        model, val_loader, loss_fn, device,
        min_length=MIN_LENGTH, max_batches=MAX_BATCHES, max_positions=MAX_POS,
    )

    return untrained, trained


def main():
    device = get_device()
    logger.info("Device: %s", device)

    cfg_rnn  = load_config("configs/default.yaml")
    cfg_lstm = load_config("configs/lstm.yaml")

    texts, labels = load_csv(cfg_rnn["paths"]["data"] + "/train.csv")
    train_p, val_p, _ = split_data(
        texts, labels,
        cfg_rnn["data"]["train_ratio"],
        cfg_rnn["data"]["val_ratio"],
        seed=SEED,
    )
    vocab = Vocabulary.load(cfg_rnn["paths"]["artifacts"] + "/vocab.json")

    # val loader với max_len=400 để có đủ sample length >= 200
    cfg_long = dict(cfg_rnn)
    cfg_long["data"] = dict(cfg_rnn["data"])
    cfg_long["data"]["max_len"] = 400
    from src.dataset import IMDbDataset
    from torch.utils.data import DataLoader
    val_ds       = IMDbDataset(val_p, vocab, max_seq_len=400)
    val_loader   = DataLoader(val_ds, batch_size=32)
    train_loader, _, _ = make_loaders(cfg_rnn, vocab, train_p, val_p, [])

    logger.info("Profiling RNN...")
    rnn_untrained, rnn_trained = build_and_profile(
        cfg_rnn, vocab, val_loader, device, train_loader
    )

    logger.info("Profiling LSTM...")
    lstm_untrained, lstm_trained = build_and_profile(
        cfg_lstm, vocab, val_loader, device, train_loader
    )

    # --- Plot ---
    try:
        import matplotlib.pyplot as plt

        x = list(range(MAX_POS))
        fig, ax = plt.subplots(figsize=(9, 5))

        for label, profile, color, ls in [
            ("RNN  (untrained)",    rnn_untrained,   "tab:blue",   "--"),
            ("RNN  (1 epoch)",      rnn_trained,     "tab:blue",   "-"),
            ("LSTM (untrained)",    lstm_untrained,  "tab:orange", "--"),
            ("LSTM (1 epoch)",      lstm_trained,    "tab:orange", "-"),
        ]:
            vals = profile.numpy()
            vals = np.where(vals > 0, vals, np.nan)
            ax.plot(x, vals, label=label, color=color, linestyle=ls, linewidth=1.5)

        ax.set_yscale("log")
        ax.set_xlabel("Distance from last real token (0 = last)")
        ax.set_ylabel("Mean ‖∂L/∂e_t‖  (log scale)")
        ax.set_title("Gradient flow by position — RNN vs LSTM")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()

        out = "artifacts/gradient_probe.png"
        fig.savefig(out, dpi=150)
        logger.info("Plot saved → %s", out)
        plt.close(fig)
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")

    # --- Print summary table ---
    checkpoints = [0, 9, 24, 49, 99, 149, 199]
    print(f"\n{'Dist':>6} | {'RNN_init':>10} | {'RNN_1ep':>10} | {'LSTM_init':>10} | {'LSTM_1ep':>10}")
    print("-" * 60)
    for k in checkpoints:
        print(f"{k:>6} | {rnn_untrained[k]:>10.2e} | {rnn_trained[k]:>10.2e} "
              f"| {lstm_untrained[k]:>10.2e} | {lstm_trained[k]:>10.2e}")


if __name__ == "__main__":
    main()
