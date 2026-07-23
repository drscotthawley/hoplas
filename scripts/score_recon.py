#!/usr/bin/env python3
"""k=0 reconstruction ceiling: encode the test set, decode (NO operator applied), classify
the decodes with the pixel classifier, and compare to the classifier's clean-image accuracy.

  high recon acc  -> the VAE round-trip preserves class despite blur (blur is cosmetic; safe
                     to score operators on this VAE)
  low  recon acc  -> the VAE is too lossy; the k=0 ceiling starves any operator measurement

Assumes our conv beta-VAE interface (CIFARVAE: .encoder(x)->(mu,logvar), .decoder(z)->img).
The borrowed MNIST VAE has a different interface, so --dataset mnist is not supported here.

Usage:
  python scripts/score_recon.py --dataset fashion
  python scripts/score_recon.py --dataset cifar10 --vae-path ~/datasets/hoplas_vae/cifar_vae_d48.pt
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets as tvd

from hoplas.vae import load_vae, _load_cifar_vae, _pick_device
from hoplas.classifier import load_classifier

_DS = {"fashion": tvd.FashionMNIST, "cifar10": tvd.CIFAR10}
_VAE_KEY = {"fashion": "fashion", "cifar10": "cifar"}   # load_vae() key for the default checkpoint


@torch.no_grad()
def accuracies(vae, clf, loader, device):
    clean_c = recon_c = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        clean_c += (clf(x).argmax(1) == y).sum().item()
        mu, _ = vae.encoder(x)          # mean encode (deterministic, best case)
        recon = vae.decoder(mu)         # k=0: decode without applying any operator
        recon_c += (clf(recon).argmax(1) == y).sum().item()
        total += y.size(0)
    return clean_c / total, recon_c / total


@torch.no_grad()
def compute_fid(vae, loader, device):
    """recon-FID: Frechet Inception Distance between real test images and their
    k=0 reconstructions. Independent of the training loss (Inception features). Lower = better.
    Returns None if torchmetrics isn't available."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        return None
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for x, _ in loader:
        x = x.to(device)
        mu, _ = vae.encoder(x)
        recon = vae.decoder(mu).clamp(0, 1)
        real = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)          # Inception wants 3ch, [0,1]
        fake = recon if recon.size(1) == 3 else recon.repeat(1, 3, 1, 1)
        fid.update(real, real=True)
        fid.update(fake, real=False)
    return fid.compute().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(_DS), required=True)
    p.add_argument("--vae-path", default=None,
                   help="explicit VAE checkpoint (else load_vae() default for the dataset)")
    p.add_argument("--clf", default=None, help="classifier dataset key (default: --dataset)")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    device = _pick_device()

    vae = (_load_cifar_vae(os.path.expanduser(args.vae_path)) if args.vae_path
           else load_vae(_VAE_KEY[args.dataset]))
    vae = vae.to(device).eval()
    clf = load_classifier(args.clf or args.dataset, device=device)

    root = os.path.expanduser(f"~/datasets/{args.dataset}")
    ds = _DS[args.dataset](root=root, train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    clean, recon = accuracies(vae, clf, loader, device)
    tag = args.vae_path or f"load_vae('{_VAE_KEY[args.dataset]}')"
    print(f"[{args.dataset}] VAE = {tag}")
    print(f"[{args.dataset}] classifier clean acc                 = {clean:.4f}")
    print(f"[{args.dataset}] k=0 reconstruction ceiling           = {recon:.4f}")
    print(f"[{args.dataset}] round-trip cost (clean - recon)      = {clean - recon:+.4f}")
    fid = compute_fid(vae, loader, device)
    if fid is not None:
        print(f"[{args.dataset}] recon-FID (real vs decode, lower=better) = {fid:.2f}")
    else:
        print(f"[{args.dataset}] recon-FID: skipped (torchmetrics unavailable)")


if __name__ == "__main__":
    main()
