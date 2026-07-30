#!/usr/bin/env python3
"""Encode CIFAR-10 images into CIFAR VAE latents for use in ring experiments.
Run once after training the VAE; skips if output file already exists.

Train split can be augmented (hflip copies x stochastic reparameterized samples) to enrich
a small (50k-image) dataset for the ops/ring model, which trains for up to 1000 epochs on
these vectors. Test split is always a single clean deterministic (mu, no hflip) pass, so
evaluation stays comparable across encodings.

Usage:
  python scripts/encode_cifar_vae.py [--vae-path ...cifar_vae_c128pw2.pt] [--out ...cifar_latents.pt] \
      [--hflip] [--n-stochastic 4]
"""

import argparse
import os
import shutil
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from hoplas.vae import _load_cifar_vae, _pick_device


@torch.no_grad()
def encode_dataset(vae, dataset, batch_size=512, hflip=False, n_stochastic=1):
    """hflip: also encode each image mirrored (doubles the pass). n_stochastic: draws per
    view via reparameterization z = mu + sigma*eps (n_stochastic=1 uses mu directly, the old
    deterministic behavior)."""
    device = next(vae.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_z, all_labels = [], []
    for x, labels in tqdm(loader):
        x = x.to(device)
        views = [x, x.flip(-1)] if hflip else [x]
        for v in views:
            mu, logvar = vae.encoder(v)
            if n_stochastic <= 1:
                all_z.append(mu.cpu()); all_labels.append(labels)
            else:
                std = (0.5 * logvar).exp()
                for _ in range(n_stochastic):
                    z = mu + std * torch.randn_like(std)
                    all_z.append(z.cpu()); all_labels.append(labels)
    return torch.cat(all_z), torch.cat(all_labels)


def _disk_free_gb(path):
    return shutil.disk_usage(os.path.dirname(path)).free / 1e9


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vae-path", default="~/datasets/hoplas_vae/cifar_vae.pt")
    p.add_argument("--out",      default="~/datasets/cifar_latents.pt")
    p.add_argument("--hflip",        action="store_true", help="also encode each TRAIN image mirrored (2x)")
    p.add_argument("--n-stochastic", type=int, default=1,
                   help="stochastic z=mu+sigma*eps draws per TRAIN view (default 1 = deterministic mu)")
    args = p.parse_args()

    out_path = os.path.expanduser(args.out)
    if os.path.exists(out_path):
        print(f"{out_path} already exists; nothing to do.")
    else:
        print(f"disk free before: {_disk_free_gb(out_path):.1f} GB")
        vae = _load_cifar_vae(os.path.expanduser(args.vae_path)).to(_pick_device()).eval()
        root = os.path.expanduser("~/datasets/cifar10")
        mult = (2 if args.hflip else 1) * args.n_stochastic
        print(f"Encoding train split... (hflip={args.hflip}, n_stochastic={args.n_stochastic}, {mult}x multiplier)")
        train_z, train_labels = encode_dataset(vae, CIFAR10(root=root, train=True,  download=True, transform=ToTensor()),
                                                hflip=args.hflip, n_stochastic=args.n_stochastic)
        print(f"disk free after train encode: {_disk_free_gb(out_path):.1f} GB")
        print("Encoding test split... (clean, unaugmented)")
        test_z,  test_labels  = encode_dataset(vae, CIFAR10(root=root, train=False, download=True, transform=ToTensor()))
        torch.save({"train_z": train_z, "train_labels": train_labels,
                    "test_z":  test_z,  "test_labels":  test_labels}, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Saved {out_path}  (train={len(train_z)}, test={len(test_z)}, nd={train_z.shape[1]}, "
              f"file_size={size_mb:.1f} MB, disk_free_after={_disk_free_gb(out_path):.1f} GB)")
