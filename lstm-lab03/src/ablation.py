import time
import torch
import torch.nn as nn
import numpy as np

from src.model import SentimentModel
from src.train import run_epoch
from src.evaluate import evaluate
from src.logger import get_logger

logger = get_logger(__name__)


def train_model(model, train_loader, val_loader, train_cfg, device,
                use_lengths=True, track_epoch_time=False):
    """
    Train model với early stopping.
    use_lengths=False → bỏ qua lengths (ablation B: no masking).
    Trả về best val_loss model state và thời gian epoch đầu tiên.
    """
    lr       = train_cfg["lr"]
    clip     = train_cfg["grad_clip"] if train_cfg.get("use_clip", True) else None
    patience = train_cfg.get("early_stopping_patience", 3)
    epochs   = train_cfg["epochs"]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss  = float("inf")
    best_state     = None
    patience_ctr   = 0
    epoch1_time    = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if use_lengths:
            train_loss, _ = run_epoch(model, train_loader, criterion,
                                      optimizer, clip, device=device)
        else:
            train_loss, _ = _run_epoch_no_lengths(model, train_loader,
                                                   criterion, optimizer, clip, device)

        epoch_time = time.time() - t0
        if epoch == 1:
            epoch1_time = epoch_time

        if use_lengths:
            val_loss, _ = run_epoch(model, val_loader, criterion, device=device)
        else:
            val_loss, _ = _run_epoch_no_lengths(model, val_loader, criterion,
                                                 None, None, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, epoch1_time


def _run_epoch_no_lengths(model, loader, criterion, optimizer, clip, device):
    """run_epoch variant mà bỏ qua lengths — dùng cho ablation B (no masking)."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for x, _lengths, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths=None)    # no masking
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item()
            total_acc  += (preds == y).float().mean().item()
            n += 1

    return total_loss / n, total_acc / n


def run_ablation_config(name, model_kwargs, train_cfg_override,
                        train_loader, val_loader, test_loader,
                        base_train_cfg, device, seeds,
                        use_lengths=True):
    """
    Chạy một ablation config với nhiều seeds, trả về dict kết quả.
    """
    accs        = []
    epoch_times = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = SentimentModel(**model_kwargs).to(device)

        cfg = {**base_train_cfg, **train_cfg_override}
        model, ep_time = train_model(
            model, train_loader, val_loader, cfg, device,
            use_lengths=use_lengths,
            track_epoch_time=True,
        )

        result = evaluate(model, test_loader, device)
        accs.append(result["accuracy"])
        if ep_time:
            epoch_times.append(ep_time)

        logger.info("%s | seed=%d | acc=%.4f", name, seed, result["accuracy"])

    return {
        "name":       name,
        "accs":       accs,
        "mean":       float(np.mean(accs)),
        "std":        float(np.std(accs)),
        "epoch_time": float(np.mean(epoch_times)) if epoch_times else None,
    }
