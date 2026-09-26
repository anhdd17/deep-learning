"""
predict.py — load checkpoint và dự đoán ảnh bất kỳ.

Cách dùng:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --ckpt runs/best.pt
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model
from src.data  import CIFAR10_MEAN, CIFAR10_STD, CLASSES
from src.utils import get_device


TRANSFORM = transforms.Compose([
    transforms.Resize(36),           # resize cạnh ngắn về 36
    transforms.CenterCrop(32),       # crop giữa → vật thể ở trung tâm được giữ lại
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = build_model(num_classes=10, dropout=0.0)  # dropout=0 khi inference
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.1%}")
    return model


@torch.no_grad()
def predict(image_path: str, model: torch.nn.Module, device: torch.device) -> None:
    img    = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)  # (1, 3, 32, 32)

    logits = model(tensor)                            # (1, 10)
    probs  = torch.softmax(logits, dim=1)[0]          # (10,)

    top5_probs, top5_idx = probs.topk(5)

    print(f"\nImage : {image_path}")
    print(f"{'Rank':<5} {'Class':<12} {'Confidence'}")
    print("-" * 30)
    for rank, (prob, idx) in enumerate(zip(top5_probs, top5_idx), 1):
        label = CLASSES[idx.item()]
        print(f"#{rank:<4} {label:<12} {prob.item():.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="đường dẫn tới ảnh cần dự đoán")
    parser.add_argument("--ckpt", default="runs/best.pt", help="đường dẫn checkpoint")
    args = parser.parse_args()

    device = get_device()
    model  = load_model(args.ckpt, device)
    predict(args.image, model, device)


if __name__ == "__main__":
    main()
