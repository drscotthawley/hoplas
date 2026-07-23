#!/usr/bin/env python3
"""Train a pixel-space SmallResNet classifier on MNIST or CIFAR-10.

Checkpoint saved to checkpoints/classifier_{dataset}.pt.
Load with: from hoplas.classifier import load_classifier

Input range is [0,1] throughout — matching the VAE decoder's clamp(0,1) output.
Augmentation policy is dataset-specific (no hflip / mild rotation for MNIST).

Usage:
    python scripts/train_classifier.py --dataset mnist
    python scripts/train_classifier.py --dataset cifar10 --batch-size 256
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from hoplas.classifier import SmallResNet, IN_CHANNELS, N_CLASSES, _ckpt_path, CHECKPOINTS_DIR


def build_transforms(dataset: str, train: bool):
    """Per-dataset augmentation policy.

    MNIST: no hflip (flipped 2/3/4/5/7 are non-digits); mild rotation only.
    CIFAR: hflip fine; larger rotations tolerable.
    All inputs stay in [0,1] — no normalization — to match decoder output range.
    """
    if dataset == "mnist":
        if train:
            return transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),                # → [0,1]
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            ])
        return transforms.ToTensor()

    if dataset == "fashion":
        # MNIST-style, but hflip is safe (mirror-invariant garment classes) and no color (grayscale).
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            ])
        return transforms.ToTensor()

    # cifar10
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),                    # → [0,1]
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    return transforms.ToTensor()


def build_loaders(dataset: str, batch_size: int, num_workers: int):
    root = os.path.expanduser(f"~/datasets/{dataset}")
    cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}[dataset]
    train_ds = cls(root=root, train=True,  download=True, transform=build_transforms(dataset, True))
    test_ds  = cls(root=root, train=False, download=True, transform=build_transforms(dataset, False))
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(test_ds,  shuffle=False, **kw))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.cpu:
        device = torch.device("cpu")
    print(f"device={device}  dataset={args.dataset}  epochs={args.epochs}  lr={args.lr}")

    train_loader, test_loader = build_loaders(args.dataset, args.batch_size, args.num_workers)

    model = SmallResNet(in_channels=IN_CHANNELS[args.dataset],
                        n_classes=N_CLASSES[args.dataset]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SmallResNet: {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    save_path = _ckpt_path(args.dataset)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()

        avg_loss = tot_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(),
                        "dataset": args.dataset,
                        "epoch": epoch,
                        "test_acc": acc}, save_path)
            marker = "  ← saved"
        print(f"epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}  test_acc={acc:.4f}  best={best_acc:.4f}{marker}")

    print(f"\nDone. Best test accuracy (ceiling): {best_acc:.4f}")
    print(f"Checkpoint: {save_path}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset",      choices=["mnist", "fashion", "cifar10"], default="mnist",  help="Dataset to train on")
    p.add_argument("--batch-size",   type=int,   default=256,  help="Mini-batch size")
    p.add_argument("--cpu",          action="store_true",       help="Force CPU even if GPU/MPS available")
    p.add_argument("--epochs",       type=int,   default=30,   help="Number of training epochs")
    p.add_argument("--lr",           type=float, default=3e-4, help="Peak learning rate (cosine annealed)")
    p.add_argument("--num-workers",  type=int,   default=4,    help="DataLoader worker processes")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
