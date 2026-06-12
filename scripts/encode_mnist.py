#! /usr/bin/env python3
"""Encode MNIST images into VAE latents, save to file for later use in ring experiments. 
Run this once to create the file, then comment out the call at the end of the script.
"""
import os
import sys
import subprocess
import torch
from tqdm.auto import tqdm

# Marco's VAE submission lives in a separate file we fetch on first run.
# Download it next to this script and import from there (cwd-independent).
# (Shell out to wget: urllib's default User-Agent gets a disguised 404 from
# raw.githubusercontent's CDN, but wget works fine.)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_MARCO_URL = "https://raw.githubusercontent.com/Ocramaru/dl_experimentation/33621a796421d9cf82d5cf7d1e49eb48f13f2f68/submissions/marco_submission.py"
_marco_path = os.path.join(HERE, "marco_submission.py")
if not os.path.exists(_marco_path):
    print(f"Downloading marco_submission.py -> {_marco_path}")
    subprocess.run(["wget", "-q", "-O", _marco_path, _MARCO_URL], check=True)

from marco_submission import SubmissionInterface, integrate_path, rk4_step

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
sub = SubmissionInterface().to(device)


# @title Encode MNIST to latents & save to file
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

@torch.no_grad()
def encode_dataset(vae, dataset, batch_size=512):
    """Encode entire dataset into VAE latents (z = mu)"""
    device = next(vae.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_latents, all_labels = [], []
    must_flatten = None
    with torch.no_grad():
        for data, labels in tqdm(loader):
            x = data.to(device)
            # next bit is so it should work with linear layers or conv
            if must_flatten is None or must_flatten==False:
                try:
                    z = vae.encoder(x)
                except RuntimeError:
                    z = vae.encoder(x.view(x.size(0), -1))
                    must_flatten = True
            else: z = vae.encoder(x.view(x.size(0), -1))
            mu, logvar = z
            all_latents.append(mu.cpu())
            all_labels.append(labels)
    return torch.cat(all_latents), torch.cat(all_labels)


def encode_mnist(vae, filename=None, batch_size=512):
    print("Acquiring train & test MNIST image datasets...")
    train_ds = MNIST(root='./data', train=True,  download=True, transform=ToTensor())
    test_ds  = MNIST(root='./data', train=False, download=True, transform=ToTensor())

    print(f"\nEncoding dataset to latents...")
    train_latents, train_labels = encode_dataset(vae, train_ds, batch_size=batch_size)
    test_latents, test_labels = encode_dataset(vae, test_ds, batch_size=batch_size)

    if filename is not None:
        print(f"Saving to {filename} ...")
        torch.save({ 'train_z': train_latents,     'test_z': test_latents,
                     'train_labels': train_labels, 'test_labels': test_labels }, filename)
    return train_latents, train_labels




# Encode the dataset - just a single file is sufficient
latent_data_filename = '~/datasets/mnist_latents.pt'
latent_data_filename = os.path.expanduser(latent_data_filename)
if not os.path.exists(latent_data_filename):
    train_latents, train_labels = encode_mnist(sub.vae, filename=latent_data_filename)
