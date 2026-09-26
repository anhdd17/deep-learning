"""
train.py — gắn tất cả lại và chạy training.

Cách dùng:
    python train.py                          # dùng config mặc định
    python train.py configs/baseline.yaml    # chỉ định config
"""

import sys
import yaml
import torch
import torch.nn as nn

from src.data    import build_dataloaders, build_test_loader
from src.model   import build_model
from src.engine  import train_one_epoch, evaluate
from src.utils   import set_seed, get_device, save_checkpoint, load_checkpoint


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    name = cfg["optimizer"]["name"].lower()
    lr   = cfg["optimizer"]["lr"]
    wd   = cfg["optimizer"]["weight_decay"]

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def main(config_path: str = "configs/baseline.yaml") -> None:
    # ------------------------------------------------------------------
    # 1. Load config
    # Tất cả hyperparameter đọc từ file yaml, không hardcode
    # → đổi lr hay batch_size chỉ cần sửa yaml, không cần sửa code
    # ------------------------------------------------------------------
    cfg = load_config(config_path)

    # ------------------------------------------------------------------
    # 2. Setup môi trường
    # ------------------------------------------------------------------
    set_seed(cfg["train"]["seed"])
    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 3. Data
    # ------------------------------------------------------------------
    train_loader, val_loader = build_dataloaders(
        data_dir    = cfg["data"]["dir"],
        batch_size  = cfg["data"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = build_model(
        num_classes = cfg["model"]["num_classes"],
        dropout     = cfg["model"]["dropout"],
    ).to(device)  # chuyển toàn bộ weights lên device

    # ------------------------------------------------------------------
    # 5. Optimizer + Loss
    # CrossEntropyLoss = Softmax + NLLLoss gộp lại
    # reduction="mean": loss trả về trung bình của cả batch
    # ------------------------------------------------------------------
    optimizer = build_optimizer(cfg, model)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 6. LR Scheduler
    # CosineAnnealingLR giảm lr theo đường cosine từ lr_max → 0
    # giúp model hội tụ mịn hơn ở cuối, thường tốt hơn lr cố định 5–10%
    # ------------------------------------------------------------------
    num_epochs = cfg["train"]["epochs"]
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    patience     = cfg["train"].get("patience", 5)  # early stopping: dừng sau N epoch không cải thiện
    best_val_acc = 0.0
    epochs_no_improve = 0
    ckpt_dir     = cfg["checkpoint"]["dir"]

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8} {'LR':>10}")
    print("-" * 64)

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # --- Evaluate ---
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # --- Step scheduler sau mỗi epoch ---
        scheduler.step()

        # --- Log ---
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"{epoch:>6} "
            f"{train_loss:>11.4f} "
            f"{train_acc:>9.1%} "
            f"{val_loss:>9.4f} "
            f"{val_acc:>7.1%} "
            f"{current_lr:>10.2e}"
        )

        # --- Checkpoint + Early Stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(
                path      = f"{ckpt_dir}/best.pt",
                model     = model,
                optimizer = optimizer,
                epoch     = epoch,
                val_acc   = val_acc,
            )
            print(f"         → saved best checkpoint (val_acc={val_acc:.1%})")
        else:
            epochs_no_improve += 1
            print(f"         → no improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping: val_acc không cải thiện sau {patience} epoch.")
                break

    print(f"\nDone. Best val acc: {best_val_acc:.1%} — saved at {ckpt_dir}/best.pt")

    # ------------------------------------------------------------------
    # 8. Test evaluation — chỉ chạy 1 lần, sau khi chọn xong model
    # Load lại best.pt (chọn theo val) rồi đo trên test set chưa từng thấy.
    # Khoảng cách val → test là thước đo generalization thật sự.
    # ------------------------------------------------------------------
    print("\n--- Final Test Evaluation ---")
    ckpt_info = load_checkpoint(f"{ckpt_dir}/best.pt", model)
    test_loader = build_test_loader(
        data_dir   = cfg["data"]["dir"],
        batch_size = cfg["data"]["batch_size"],
        num_workers= cfg["data"]["num_workers"],
    )
    _, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Checkpoint epoch : {ckpt_info['epoch']}")
    print(f"Val  acc (best)  : {ckpt_info['val_acc']:.1%}")
    print(f"Test acc         : {test_acc:.1%}")
    gap = ckpt_info["val_acc"] - test_acc
    print(f"Val → Test gap   : {gap:+.1%}  {'⚠ overfit vào val?' if gap > 0.03 else '✓ generalize tốt'}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/baseline.yaml"
    main(config_path)
