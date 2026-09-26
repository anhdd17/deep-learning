import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# engine.py có 2 hàm:
#
#   train_one_epoch() — chạy 1 lượt qua toàn bộ train set, cập nhật weights
#   evaluate()        — chạy qua val set, KHÔNG cập nhật weights, chỉ đo
#
# Hai hàm này được gọi lặp lại từ script training chính:
#   for epoch in range(num_epochs):
#       train_loss, train_acc = train_one_epoch(...)
#       val_loss,   val_acc   = evaluate(...)
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Chạy 1 epoch training. Trả về (avg_loss, accuracy) trên toàn train set.
    """

    # model.train() bật 2 thứ:
    #   - Dropout: tắt neuron ngẫu nhiên  → chỉ hoạt động khi train
    #   - BatchNorm: dùng batch statistics → chỉ khi train
    # Nếu quên gọi model.train() sau evaluate(), Dropout sẽ vẫn tắt
    # → model không học đúng cách
    model.train()

    running_loss    = torch.tensor(0.0, device=device)
    running_correct = torch.tensor(0,   device=device)
    total_samples   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)   # (B, 3, 32, 32)
        labels = labels.to(device, non_blocking=True)   # (B,)

        # set_to_none=True: giải phóng memory thay vì ghi 0 → nhanh hơn
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        # tích lũy trên device, tránh đồng bộ CPU–device mỗi batch
        running_loss    += loss.detach() * batch_size
        running_correct += (logits.argmax(dim=1) == labels).sum()
        total_samples   += batch_size

    # chỉ gọi .item() 1 lần cuối epoch
    avg_loss = (running_loss / total_samples).item()
    accuracy = (running_correct / total_samples).item()
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Đánh giá model trên val/test set. Trả về (avg_loss, accuracy).
    Không cập nhật weights.
    """

    # model.eval() tắt Dropout và chuyển BatchNorm sang inference mode
    # (dùng running mean/std tích lũy từ train thay vì batch statistics)
    model.eval()

    running_loss    = torch.tensor(0.0, device=device)
    running_correct = torch.tensor(0,   device=device)
    total_samples   = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss    += loss * batch_size
            running_correct += (logits.argmax(dim=1) == labels).sum()
            total_samples   += batch_size

    avg_loss = (running_loss / total_samples).item()
    accuracy = (running_correct / total_samples).item()
    return avg_loss, accuracy
