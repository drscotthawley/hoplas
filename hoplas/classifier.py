"""Pixel-space classifier for MNIST and CIFAR-10.

Designed as independent eval instrumentation for hoplas:
- Pixel-space (not embedding-space) input → VAE-agnostic, avoids circularity
- Small ResNet backbone (one model, switch dataset)
- [0,1] input range matching VAE decoder output
- forward(x, return_features=True) exposes penultimate embedding for density/coverage eval

Usage:
    from hoplas.classifier import load_classifier
    clf = load_classifier("mnist")   # or "cifar10"
    logits = clf(imgs)               # imgs in [0,1], shape (B,1,28,28) or (B,3,32,32)
    logits, feats = clf(imgs, return_features=True)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")

N_CLASSES = {"mnist": 10, "cifar10": 10}
IN_CHANNELS = {"mnist": 1, "cifar10": 3}


class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = (nn.Sequential(
                          nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_ch))
                      if stride != 1 or in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))


class SmallResNet(nn.Module):
    """~500K-param ResNet; handles MNIST (1ch 28×28) and CIFAR (3ch 32×32)."""

    def __init__(self, in_channels=1, n_classes=10):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # four stages: 32→64→128→256, stride-2 on stages 2-4
        self.layer1 = nn.Sequential(_ResBlock(32,  64,  stride=1), _ResBlock(64,  64))
        self.layer2 = nn.Sequential(_ResBlock(64,  128, stride=2), _ResBlock(128, 128))
        self.layer3 = nn.Sequential(_ResBlock(128, 256, stride=2), _ResBlock(256, 256))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 256
        self.head   = nn.Linear(256, n_classes)

    def forward(self, x, return_features=False):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        feats = self.pool(h).flatten(1)           # (B, 256)
        logits = self.head(feats)
        if return_features:
            return logits, feats
        return logits


def _ckpt_path(dataset: str) -> str:
    return os.path.join(CHECKPOINTS_DIR, f"classifier_{dataset}.pt")


def load_classifier(dataset: str, device=None, strict=True) -> SmallResNet:
    """Load a trained SmallResNet from checkpoints/classifier_{dataset}.pt.

    Mirrors load_vae() API.  Raises FileNotFoundError if no checkpoint exists yet
    (run scripts/train_classifier.py first).
    """
    if dataset not in N_CLASSES:
        raise ValueError(f"dataset must be one of {list(N_CLASSES)}; got {dataset!r}")
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    path = _ckpt_path(dataset)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No classifier checkpoint at {path}. "
            f"Run: python scripts/train_classifier.py --dataset {dataset}")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = SmallResNet(in_channels=IN_CHANNELS[dataset], n_classes=N_CLASSES[dataset])
    model.load_state_dict(blob["state_dict"], strict=strict)
    model.to(device).eval()
    return model
