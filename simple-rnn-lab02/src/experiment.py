import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.logger import get_logger
from src.train import accuracy

logger = get_logger(__name__)

SEQ_LENGTHS = [10, 20, 50, 100]


def train_one_epoch(model, loader, criterion, optimizer, clip):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        logits = model(x)
        loss   = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        total_acc  += accuracy(logits, y)
        n += 1
    return total_loss / n, total_acc / n


def get_grad_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms


def run_experiment(make_loaders_fn, make_model_fn, cfg):
    """
    Train models với seq_lengths khác nhau, log gradient norms.
    make_loaders_fn(seq_len) → (train_loader, val_loader)
    make_model_fn()          → model mới
    """
    epochs   = 5   # đủ để quan sát gradient, không cần train full
    lr       = cfg["training"]["lr"]
    clip     = cfg["training"]["clip_grad_norm"]
    runs_dir = cfg["paths"]["runs"]
    rnn_type = cfg["model"]["rnn_type"]

    criterion = nn.BCEWithLogitsLoss()
    summary   = []

    for seq_len in SEQ_LENGTHS:
        logger.info("=== seq_len=%d rnn_type=%s ===", seq_len, rnn_type)
        train_loader, val_loader = make_loaders_fn(seq_len)
        model     = make_model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        writer    = SummaryWriter(log_dir=f"{runs_dir}/exp_seqlen{seq_len}_{rnn_type}")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, clip)
            grad_norms = get_grad_norms(model)

            # log gradient norms — đây là phần quan trọng nhất của experiment
            for name, norm in grad_norms.items():
                writer.add_scalar(f"grad_norm/{name}", norm, epoch)

            logger.info(
                "seq_len=%3d epoch=%d loss=%.4f acc=%.4f | grad W_xh=%.6f W_hh=%.6f",
                seq_len, epoch, train_loss, train_acc,
                grad_norms.get("rnn.W_xh", grad_norms.get("rnn.weight_ih_l0", 0)),
                grad_norms.get("rnn.W_hh", grad_norms.get("rnn.weight_hh_l0", 0)),
            )

        # val accuracy sau 5 epochs
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                preds  = (torch.sigmoid(logits) >= 0.5).float()
                val_preds.extend(preds.tolist())
                val_labels.extend(y.tolist())
        val_acc = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)

        final_norms = get_grad_norms(model)
        summary.append({
            "seq_len":  seq_len,
            "rnn_type": rnn_type,
            "val_acc":  val_acc,
            "grad_W_xh": final_norms.get("rnn.W_xh", final_norms.get("rnn.weight_ih_l0", 0)),
            "grad_W_hh": final_norms.get("rnn.W_hh", final_norms.get("rnn.weight_hh_l0", 0)),
        })
        writer.close()

    return summary
