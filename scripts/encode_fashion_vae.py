#!/usr/bin/env python3
"""Encode Fashion-MNIST into VAE latents for the operator-reuse experiment.
Writes ~/datasets/fashion_latents.pt in the EncodingsDataset format
({train_z, train_labels, test_z, test_labels}). Run once after the VAE is trained.

Usage:
  python scripts/encode_fashion_vae.py [--vae-path ...fashion_vae_f32pw1.pt] [--out ...fashion_latents.pt]
"""
import argparse
import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from hoplas.vae import _load_cifar_vae, _pick_device


@torch.no_grad()
def encode_dataset(vae, dataset, batch_size=512):
    device = next(vae.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_z, all_labels = [], []
    for x, labels in tqdm(loader):
        mu, _ = vae.encoder(x.to(device))          # mean encode (deterministic)
        all_z.append(mu.cpu()); all_labels.append(labels)
    return torch.cat(all_z), torch.cat(all_labels)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vae-path", default="~/datasets/hoplas_vae/fashion_vae_f32pw1.pt")
    p.add_argument("--out",      default="~/datasets/fashion_latents.pt")
    args = p.parse_args()

    out_path = os.path.expanduser(args.out)
    if os.path.exists(out_path):
        print(f"{out_path} already exists; nothing to do.")
    else:
        vae = _load_cifar_vae(os.path.expanduser(args.vae_path)).to(_pick_device()).eval()
        root = os.path.expanduser("~/datasets/fashion_mnist")
        print("Encoding train split...")
        train_z, train_labels = encode_dataset(vae, FashionMNIST(root=root, train=True,  download=True, transform=ToTensor()))
        print("Encoding test split...")
        test_z,  test_labels  = encode_dataset(vae, FashionMNIST(root=root, train=False, download=True, transform=ToTensor()))
        torch.save({"train_z": train_z, "train_labels": train_labels,
                    "test_z":  test_z,  "test_labels":  test_labels}, out_path)
        print(f"Saved {out_path}  (train={len(train_z)}, test={len(test_z)}, nd={train_z.shape[1]})")
