#!/usr/bin/env python3
"""Encode CIFAR-10 images into CIFAR VAE latents for use in ring experiments.
Run once after training the VAE; skips if output file already exists.
"""

import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from hoplas.vae import load_cifar_vae


@torch.no_grad()
def encode_dataset(vae, dataset, batch_size=512):
    device = next(vae.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_z, all_labels = [], []
    for x, labels in tqdm(loader):
        mu, _ = vae.encoder(x.to(device))
        all_z.append(mu.cpu()); all_labels.append(labels)
    return torch.cat(all_z), torch.cat(all_labels)


if __name__ == "__main__":
    out_path = os.path.expanduser("~/datasets/cifar_latents.pt")
    if os.path.exists(out_path):
        print(f"{out_path} already exists; nothing to do.")
    else:
        vae = load_cifar_vae()
        root = os.path.expanduser("~/datasets/cifar10")
        print("Encoding train split...")
        train_z, train_labels = encode_dataset(vae, CIFAR10(root=root, train=True,  download=True, transform=ToTensor()))
        print("Encoding test split...")
        test_z,  test_labels  = encode_dataset(vae, CIFAR10(root=root, train=False, download=True, transform=ToTensor()))
        torch.save({"train_z": train_z, "train_labels": train_labels,
                    "test_z":  test_z,  "test_labels":  test_labels}, out_path)
        print(f"Saved {out_path}  (train={len(train_z)}, test={len(test_z)}, nd={train_z.shape[1]})")
