import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tại sao CNN thay vì Linear thẳng?
#
# Ảnh 32×32×3 = 3072 số. Nếu dùng Linear(3072, 10):
#   - Không quan tâm vị trí pixel — pixel góc trên trái và góc dưới phải
#     được xử lý độc lập, không biết chúng là hàng xóm của nhau
#   - Quá nhiều tham số, dễ overfit
#
# Conv2d học "bộ lọc" nhỏ (ví dụ 3×3) trượt khắp ảnh:
#   - Dùng chung 1 bộ lọc cho mọi vị trí → ít tham số hơn nhiều
#   - Tự nhiên nhận ra cạnh, góc, texture ở bất kỳ đâu trong ảnh
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """
    CNN 3 khối Conv cho CIFAR-10 (10 lớp, ảnh 32×32 RGB).

    Luồng tensor (batch_size = B):
        Input  →  (B,   3, 32, 32)
        Block1 →  (B,  32, 16, 16)
        Block2 →  (B,  64,  8,  8)
        Block3 →  (B, 128,  4,  4)
        Flatten→  (B, 2048)
        FC     →  (B,  10)   ← logits, chưa qua softmax
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # PHẦN 1 — Feature Extractor (Conv blocks)
        # Nhiệm vụ: biến ảnh thô → vector đặc trưng có ý nghĩa
        # ------------------------------------------------------------------

        self.features = nn.Sequential(

            # === Block 1 ===
            # Conv2d(in_channels, out_channels, kernel_size, padding)
            #   in_channels  = 3   (RGB)
            #   out_channels = 32  (học 32 bộ lọc khác nhau)
            #   kernel_size  = 3   (cửa sổ 3×3 trượt qua ảnh)
            #   padding      = 1   (giữ nguyên H×W sau conv: 32→32)
            # Shape: (B, 3, 32, 32) → (B, 32, 32, 32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),

            # BatchNorm2d: chuẩn hóa output của Conv theo từng channel
            # → training ổn định hơn, học nhanh hơn, ít nhạy cảm với learning rate
            # Shape: không đổi → (B, 32, 32, 32)
            nn.BatchNorm2d(32),

            # ReLU: hàm kích hoạt — giữ số dương, đặt số âm về 0
            # Tại sao ReLU thay vì Sigmoid?
            #   Sigmoid → gradient rất nhỏ ở 2 đầu → vanishing gradient
            #   ReLU    → gradient = 1 với mọi số dương → học nhanh hơn
            # Shape: không đổi → (B, 32, 32, 32)
            nn.ReLU(inplace=True),

            # MaxPool2d(kernel_size=2, stride=2): lấy giá trị lớn nhất
            # trong mỗi ô 2×2 → thu nhỏ ảnh xuống còn 1/2
            # Tại sao cần Pool?
            #   - Giảm số tham số ở các layer sau
            #   - Tăng "tầm nhìn" của các Conv layer sau (mỗi pixel bao quát vùng rộng hơn)
            # Shape: (B, 32, 32, 32) → (B, 32, 16, 16)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # === Block 2 ===
            # in_channels tăng từ 32 → 64: học nhiều đặc trưng phức tạp hơn
            # (Block 1 học cạnh/màu, Block 2 học góc/texture)
            # Shape: (B, 32, 16, 16) → (B, 64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Shape: (B, 64, 16, 16) → (B, 64, 8, 8)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # === Block 3 ===
            # 64 → 128 channels: học đặc trưng cấp cao (hình dạng vật thể)
            # Shape: (B, 64, 8, 8) → (B, 128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Shape: (B, 128, 8, 8) → (B, 128, 4, 4)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ------------------------------------------------------------------
        # PHẦN 2 — Classifier (Fully Connected)
        # Nhiệm vụ: từ vector đặc trưng → dự đoán 10 class
        # ------------------------------------------------------------------

        self.classifier = nn.Sequential(

            # Dropout: tắt ngẫu nhiên 50% neuron trong lúc train
            # → buộc model không phụ thuộc vào một vài neuron cụ thể
            # → giảm overfitting
            # Trong lúc eval (model.eval()), Dropout tự tắt
            nn.Dropout(p=dropout),

            # Linear(in, out): fully connected layer
            # in = 128 × 4 × 4 = 2048  (flatten từ feature map)
            # out = 256
            # Shape: (B, 2048) → (B, 256)
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            # Layer cuối: 256 → 10 (số class)
            # Không có activation ở đây — CrossEntropyLoss tự xử lý softmax
            # Shape: (B, 256) → (B, 10)
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 32, 32)
        x = self.features(x)       # → (B, 128, 4, 4)
        x = x.flatten(start_dim=1) # → (B, 2048)  — giữ nguyên batch dim
        x = self.classifier(x)     # → (B, 10)
        return x


def build_model(num_classes: int = 10, dropout: float = 0.5) -> SimpleCNN:
    return SimpleCNN(num_classes=num_classes, dropout=dropout)
