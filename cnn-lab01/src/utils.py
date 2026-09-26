import random
import torch
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# utils.py có 3 nhóm:
#
#   set_seed()          — cố định random seed để kết quả reproducible
#   get_device()        — tự detect CPU / MPS (Apple Silicon)
#   save/load_checkpoint — lưu và khôi phục trạng thái training
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """
    Cố định seed cho tất cả nguồn random trong PyTorch pipeline.

    Tại sao cần?
    Training DL có nhiều bước ngẫu nhiên:
      - Khởi tạo weights (random)
      - DataLoader shuffle thứ tự ảnh
      - Dropout tắt neuron ngẫu nhiên
      - RandomCrop / RandomFlip trong augmentation

    Nếu không fix seed, chạy 2 lần cùng code cho kết quả khác nhau
    → không thể so sánh thí nghiệm, không thể debug.

    Phải set cả 3 nguồn vì PyTorch, NumPy, Python dùng RNG riêng.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """
    Tự động chọn device tốt nhất hiện có.

    Thứ tự ưu tiên: MPS (Apple Silicon GPU) → CPU
    Trên Apple M4: MPS nhanh hơn CPU khoảng 5–10x cho CNN.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
) -> None:
    """
    Lưu toàn bộ trạng thái training vào file .pt

    Lưu cả optimizer state để có thể train tiếp từ điểm dừng.
    Nếu chỉ lưu model weights, resume training sẽ bị sai
    vì optimizer (Adam, SGD...) có internal state riêng
    (momentum, adaptive learning rate...).

    Args:
        path:      đường dẫn file lưu, ví dụ "runs/best.pt"
        model:     model đang train
        optimizer: optimizer đang dùng
        epoch:     epoch hiện tại (để biết đã train đến đâu)
        val_acc:   accuracy trên val set (để biết đây là checkpoint tốt nhất chưa)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":     epoch,
            "val_acc":   val_acc,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """
    Khôi phục model (và optimizer nếu muốn train tiếp) từ checkpoint.

    Trả về dict chứa epoch và val_acc để biết đang ở đâu.

    Args:
        path:      đường dẫn file checkpoint
        model:     model cần load weights vào
        optimizer: truyền vào nếu muốn train tiếp, None nếu chỉ inference
    """
    # map_location: đảm bảo checkpoint lưu trên GPU vẫn load được trên CPU/MPS
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    return {"epoch": ckpt["epoch"], "val_acc": ckpt["val_acc"]}
