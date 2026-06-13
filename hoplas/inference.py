#!/usr/bin/env python3
"""Inference demo: load a checkpoint, transform a 10×10 MNIST grid, save PNGs.

Each column holds one digit class (0-9), 10 rows each. Three grids are saved:
  mnist_input.png        raw test images
  mnist_recon.png        VAE round-trip, no op (decoder quality baseline)
  mnist_transformed.png  full pipeline: encode -> proj -> op -> inv_proj -> decode

Expected result if training succeeded: column c should look like digit (c+1)%10.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image

from hoplas.models import Projector
from hoplas.ops import OpWrapper
from hoplas.vae import load_mnist_vae


_MNIST_ROOT = os.path.expanduser("~/datasets/mnist")


def load_for_inference(ckpt_path, device=None):
    """Rebuild proj, trans_op, inv_proj from a checkpoint and return them in eval mode."""
    import torch
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    proj = Projector(nd=a["nd"], n_hid=a["n_hid"], n_layers=a["proj_layers"],
                     proj_resid=a["proj_resid"], unit_norm=a["unit_norm"])
    proj.load_state_dict(ck["proj"])
    trans_op = OpWrapper(a["op"], a["nd"], a["order"], a["op_resid"], a["rank"], a["unit_norm"])
    trans_op.load_state_dict(ck["trans_op"])
    inv_proj = Projector(nd=a["nd"], n_hid=a["n_hid"], n_layers=a["proj_layers"], unit_norm=False)
    inv_proj.load_state_dict(ck["inv_proj"])
    proj.to(device).eval()
    trans_op.to(device).eval()
    inv_proj.to(device).eval()
    print(f"loaded checkpoint: epoch={ck['epoch']}  val_loss={ck['val_loss']:.6f}  op={a['op']}  nd={a['nd']}")
    return proj, trans_op, inv_proj, device


def make_class_ordered_images(n_per_class=10):
    """Return (100, 1, 28, 28) MNIST test images: column c = digit c, n_per_class rows.

    make_grid with nrow=n_classes produces a grid where each column is one digit.
    Ordering: row-major, so the batch is [row0_digit0, row0_digit1, ..., row0_digit9,
                                           row1_digit0, ...].
    """
    n_classes = 10
    ds = MNIST(root=_MNIST_ROOT, train=False, download=True, transform=ToTensor())
    # collect indices per class
    buckets = [[] for _ in range(n_classes)]
    for idx, (_, label) in enumerate(ds):
        if len(buckets[label]) < n_per_class:
            buckets[label].append(idx)
        if all(len(b) == n_per_class for b in buckets):
            break
    # interleave: row-major so make_grid(nrow=n_classes) gives one column per digit
    ordered_indices = [buckets[c][r] for r in range(n_per_class) for c in range(n_classes)]
    imgs = torch.stack([ds[i][0] for i in ordered_indices])   # (100, 1, 28, 28)
    return imgs


@torch.no_grad()
def apply_operation(z_batch, proj, trans_op, inv_proj):
    """Latent -> proj -> op -> inv_proj -> latent'. Returns the transformed latent."""
    zp = proj(z_batch)
    zp_t = trans_op(zp)
    return inv_proj(zp_t)


@torch.no_grad()
def run_demo(ckpt_path, out_dir=".", n_per_class=10, device=None):
    """Full pipeline: load checkpoint, encode MNIST grid, transform, decode, save PNGs."""
    proj, trans_op, inv_proj, device = load_for_inference(ckpt_path, device)
    vae = load_mnist_vae(device=str(device))

    imgs = make_class_ordered_images(n_per_class).to(device)   # (100, 1, 28, 28)

    # encode: image -> latent mu
    mu, _ = vae.encoder(imgs)

    # transformed latent
    mu_t = apply_operation(mu, proj, trans_op, inv_proj)

    # reconstruct latents (no op, for baseline comparison)
    zp = proj(mu)
    mu_recon = inv_proj(zp)

    # decode
    imgs_recon = vae.decoder(mu_recon).clamp(0, 1)
    imgs_transformed = vae.decoder(mu_t).clamp(0, 1)

    nrow = 10  # one column per digit class
    os.makedirs(out_dir, exist_ok=True)
    save_image(make_grid(imgs.cpu(),           nrow=nrow), os.path.join(out_dir, "mnist_input.png"))
    save_image(make_grid(imgs_recon.cpu(),     nrow=nrow), os.path.join(out_dir, "mnist_recon.png"))
    save_image(make_grid(imgs_transformed.cpu(), nrow=nrow), os.path.join(out_dir, "mnist_transformed.png"))
    print(f"saved: {out_dir}/mnist_input.png  mnist_recon.png  mnist_transformed.png")


def main():
    p = argparse.ArgumentParser(description="hoplas inference demo")
    p.add_argument("checkpoint", help="path to .pt checkpoint from train_ring.py")
    p.add_argument("--out-dir", default=".", help="directory for output PNGs")
    p.add_argument("--n-per-class", type=int, default=10, help="rows per digit (default 10 -> 10x10 grid)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = "cpu" if args.cpu else None
    run_demo(args.checkpoint, out_dir=args.out_dir, n_per_class=args.n_per_class, device=device)


if __name__ == "__main__":
    main()
