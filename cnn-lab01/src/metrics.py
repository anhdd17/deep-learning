import torch


# ---------------------------------------------------------------------------
# metrics.py có 2 thứ:
#
#   AverageMeter — theo dõi trung bình loss/acc trong 1 epoch (từng batch)
#   accuracy()   — tính % dự đoán đúng từ logits và labels
#
# Tại sao cần file riêng?
#   engine.py đã tính loss và acc rồi, nhưng cách tính đó chỉ trả về
#   kết quả cuối epoch. Khi muốn log từng bước (mỗi 100 batch in 1 dòng),
#   cần AverageMeter để tích lũy dần.
# ---------------------------------------------------------------------------


class AverageMeter:
    """
    Tích lũy giá trị theo từng batch, trả về trung bình cộng.

    Dùng cho loss và accuracy trong vòng lặp training:
        meter = AverageMeter()
        for images, labels in loader:
            loss = ...
            meter.update(loss.item(), n=len(images))
        print(meter.avg)   # trung bình loss cả epoch
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        # val: giá trị của batch (ví dụ loss trung bình của batch)
        # n:   số sample trong batch (để tính weighted average đúng)
        #
        # Tại sao nhân val × n?
        # loss.item() là trung bình loss của batch đó.
        # Để tính trung bình đúng trên nhiều batch có thể khác batch_size,
        # phải cộng tổng (val × n) rồi chia tổng n, không phải trung bình của trung bình.
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Tính top-1 accuracy: % số sample model đoán đúng class.

    Args:
        logits: (B, num_classes) — output thô của model, chưa softmax
        labels: (B,)             — class index đúng, số nguyên 0–9

    Returns:
        accuracy trong khoảng [0.0, 1.0]
    """
    # argmax(dim=1): với mỗi sample, lấy index có logit cao nhất
    # → đó là class model dự đoán
    # Shape: (B, num_classes) → (B,)
    preds = logits.argmax(dim=1)

    # So sánh từng phần tử: đúng → True (=1), sai → False (=0)
    # .float().mean() → tỉ lệ đúng
    return (preds == labels).float().mean().item()
