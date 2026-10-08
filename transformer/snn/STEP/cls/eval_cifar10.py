import sys, types
if 'imp' not in sys.modules:
    imp = types.ModuleType('imp')
    imp.find_module = lambda *a, **k: None
    imp.load_module = lambda *a, **k: None
    sys.modules['imp'] = imp

import torch
import torchvision
import torchvision.transforms as T
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from models.utils.node import *
from models.static import spikformer_cifar
from timm.models import create_model

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]),
])


def load_model(checkpoint_path, device):
    model = create_model('spikformer_cifar')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available()
                        else 'cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")

    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=TRANSFORM)
    loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = load_model(args.checkpoint, args.device)
    print(f"Model loaded from {args.checkpoint}")

    acc = evaluate(model, loader, args.device)
    print(f"\nTest accuracy: {acc:.2f}%")
