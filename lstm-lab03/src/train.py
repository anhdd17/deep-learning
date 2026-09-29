import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.logger import get_logger

logger = get_logger(__name__)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == labels).float().mean().item()


def _total_grad_norm(model) -> float:
    total = sum(
        p.grad.norm() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    return total ** 0.5


def run_epoch(model, loader, criterion, optimizer=None, clip=None,
              writer=None, global_step=0, device=None):
    """Một epoch train hoặc eval. optimizer=None → eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for x, lengths, y in loader:
            if device is not None:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()

                if writer:
                    pre_norm = _total_grad_norm(model)
                    writer.add_scalar("grad_norm/pre_clip", pre_norm, global_step + n_batches)

                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)

                if writer:
                    post_norm = _total_grad_norm(model)
                    writer.add_scalar("grad_norm/post_clip", post_norm, global_step + n_batches)

                optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy(logits, y)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


def train(model, train_loader, val_loader, cfg, save_path, model_cfg, device=None):
    epochs   = cfg["train"]["epochs"]
    lr       = cfg["train"]["lr"]
    clip     = cfg["train"]["grad_clip"]
    patience = cfg["train"].get("early_stopping_patience", 3)
    runs_dir = cfg["paths"]["runs"]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer    = SummaryWriter(log_dir=runs_dir)

    best_val_loss    = float("inf")
    patience_counter = 0
    batches_per_epoch = len(train_loader)

    for epoch in range(1, epochs + 1):
        global_step = (epoch - 1) * batches_per_epoch
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, clip,
            writer=writer, global_step=global_step, device=device,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device=device)

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )
        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg}, save_path)
            logger.info("Checkpoint saved (val_loss=%.4f) → %s", val_loss, save_path)
        else:
            patience_counter += 1
            logger.info("No improvement (%d/%d)", patience_counter, patience)
            if patience_counter >= patience:
                logger.info("Early stopping tại epoch %d", epoch)
                break

    writer.close()
    logger.info("Training done. Best val_loss=%.4f", best_val_loss)
