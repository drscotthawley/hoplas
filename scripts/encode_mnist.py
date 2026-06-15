#! /usr/bin/env python3
"""Encode MNIST images into VAE latents, save to file for later use in ring experiments.
Run this once to create the file; it skips work if the output already exists.
"""
import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from hoplas.vae import load_vae


@torch.no_grad()
def encode_dataset(vae, dataset, batch_size=512):
    """Encode entire dataset into VAE latents (z = mu)"""
    device = next(vae.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_latents, all_labels = [], []
    must_flatten = None
    for data, labels in tqdm(loader):
        x = data.to(device)
        # next bit is so it should work with linear layers or conv
        if must_flatten is None or must_flatten == False:
            try:
                z = vae.encoder(x)
            except RuntimeError:
                z = vae.encoder(x.view(x.size(0), -1))
                must_flatten = True
        else:
            z = vae.encoder(x.view(x.size(0), -1))
        mu, logvar = z
        all_latents.append(mu.cpu())
        all_labels.append(labels)
    return torch.cat(all_latents), torch.cat(all_labels)


def encode_mnist(vae, filename=None, batch_size=512, mnist_root="~/datasets/mnist"):
    mnist_root = os.path.expanduser(mnist_root)
    print("Acquiring train & test MNIST image datasets...")
    train_ds = MNIST(root=mnist_root, train=True,  download=True, transform=ToTensor())
    test_ds  = MNIST(root=mnist_root, train=False, download=True, transform=ToTensor())

    print("\nEncoding dataset to latents...")
    train_latents, train_labels = encode_dataset(vae, train_ds, batch_size=batch_size)
    test_latents, test_labels = encode_dataset(vae, test_ds, batch_size=batch_size)

    if filename is not None:
        print(f"Saving to {filename} ...")
        torch.save({'train_z': train_latents,     'test_z': test_latents,
                    'train_labels': train_labels, 'test_labels': test_labels}, filename)
    return train_latents, train_labels


if __name__ == "__main__":
    # Encode the dataset - just a single file is sufficient
    latent_data_filename = os.path.expanduser('~/datasets/mnist_latents.pt')
    if not os.path.exists(latent_data_filename):
        vae = load_vae("mnist")
        encode_mnist(vae, filename=latent_data_filename)
    else:
        print(f"{latent_data_filename} already exists; nothing to do.")
