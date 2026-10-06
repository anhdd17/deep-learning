import torch
import torchvision.transforms as T
from PIL import Image
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from models.utils.node import *
from models.static import spikformer_imagenet
from timm.models import create_model

CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(checkpoint_path, device):
    model = create_model(
        'spikformer_imagenet',
        num_classes=4,
        img_size=224,
        patch_size=16,
        embed_dim=384,
        num_heads=8,
        mlp_ratio=4,
        depths=4,
        step=2,
        node_type='LIFNode',
        tau=2.0,
        threshold=1.0,
        act_function='SigmoidGrad',
        alpha=4.0,
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

@torch.no_grad()
def infer(model, image_paths, device):
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        x = TRANSFORM(img).unsqueeze(0).to(device)
        # nodes use step=4 but self.T=2, so need batch divisible by 2
        x = x.repeat(2, 1, 1, 1)
        logits = model(x)
        probs = torch.softmax(logits.float(), dim=-1)[0]
        pred = probs.argmax().item()
        label = os.path.basename(os.path.dirname(path))
        correct = '✓' if CLASSES[pred] == label else '✗'
        print(f"{correct} [{label:12s}] → pred: {CLASSES[pred]:12s} "
              f"({probs[pred]*100:.1f}%)  "
              f"scores: { {c: f'{p*100:.1f}%' for c,p in zip(CLASSES, probs.tolist())} }")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}\n")
    model = load_model(args.checkpoint, args.device)
    infer(model, args.images, args.device)
