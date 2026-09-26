import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.logger import get_logger

logger = get_logger(__name__)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer=None, clip=None):
    """Một epoch train hoặc eval. optimizer=None → eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            logits = model(x)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy(logits, y)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


def log_grad_norms(model, writer, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            writer.add_scalar(f"grad_norm/{name}", norm, step)
            logger.debug("grad_norm %s=%.6f", name, norm)


def train(model, train_loader, val_loader, cfg, save_path):
    epochs    = cfg["training"]["epochs"]
    lr        = cfg["training"]["lr"]
    clip      = cfg["training"]["clip_grad_norm"]
    runs_dir  = cfg["paths"]["runs"]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer    = SummaryWriter(log_dir=runs_dir)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, clip)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion)

        # log to console
        logger.info(
            "Epoch %2d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )

        # log to TensorBoard
        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        log_grad_norms(model, writer, epoch)

        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info("Checkpoint saved (val_loss=%.4f) → %s", val_loss, save_path)

    writer.close()
    logger.info("Training done. Best val_loss=%.4f", best_val_loss)
