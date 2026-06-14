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
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image

from hoplas.models import Projector
from hoplas.ops import OpWrapper
from hoplas.vae import load_vae


_MNIST_ROOT = os.path.expanduser("~/datasets/mnist")
_CIFAR_ROOT = os.path.expanduser("~/datasets/cifar10")


def load_for_inference(ckpt_path, device=None):
    """Rebuild proj, trans_op, inv_proj from a checkpoint and return them in eval mode."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    pnd = a.get("pnd", a["nd"])  # backward compat with checkpoints before pnd was added
    proj = Projector(nd=a["nd"], pnd=pnd, n_hid=a["n_hid"], n_layers=a["proj_layers"],
                     proj_resid=a["proj_resid"], unit_norm=a["unit_norm"])
    proj.load_state_dict(ck["proj"])
    trans_op = OpWrapper(a["op"], pnd, a["order"], a["op_resid"], a["rank"], a["unit_norm"])
    trans_op.load_state_dict(ck["trans_op"])
    inv_proj = Projector(nd=pnd, pnd=a["nd"], n_hid=a["n_hid"], n_layers=a["proj_layers"], unit_norm=False)
    inv_proj.load_state_dict(ck["inv_proj"])
    for m in (proj, trans_op, inv_proj): m.to(device).eval()
    print(f"loaded checkpoint: epoch={ck['epoch']}  val_sim_loss={ck['val_sim_loss']:.6f}  op={a['op']}  nd={a['nd']}")
    return proj, trans_op, inv_proj, device, a["dataset"]


def make_class_ordered_images(dataset="mnist", n_per_class=10):
    """Return (100, C, H, W) test images ordered row-major: column c = class c.

    make_grid(result, nrow=10) gives one column per class.
    dataset: "mnist" -> (1,28,28) grayscale; "cifar" -> (3,32,32) RGB.
    """
    n_classes = 10
    if dataset == "mnist":
        ds = MNIST(root=_MNIST_ROOT, train=False, download=True, transform=ToTensor())
    elif dataset == "cifar":
        ds = CIFAR10(root=_CIFAR_ROOT, train=False, download=True, transform=ToTensor())
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")
    buckets = [[] for _ in range(n_classes)]
    for idx, (_, label) in enumerate(ds):
        if len(buckets[label]) < n_per_class:
            buckets[label].append(idx)
        if all(len(b) == n_per_class for b in buckets):
            break
    ordered = [buckets[c][r] for r in range(n_per_class) for c in range(n_classes)]
    return torch.stack([ds[i][0] for i in ordered])


@torch.no_grad()
def apply_operation(z_batch, proj, trans_op, inv_proj, repeat=1):
    """Latent -> proj -> op(^repeat) -> inv_proj -> latent'.

    repeat composes the operator in the projected space (op applied `repeat` times
    before a single inv_proj), so e.g. repeat=2 advances two steps. repeat=n_classes
    would in principle close the ring, though a truly generative op need not return
    exactly to the start.
    """
    h = proj(z_batch)
    for _ in range(repeat):
        h = trans_op(h)
    return inv_proj(h)


@torch.no_grad()
def make_viz_grids(vae, proj, trans_op, inv_proj, imgs, repeat=1):
    """Encode imgs, run recon and transform pipelines, decode both.
    Returns (imgs_input, imgs_recon, imgs_transformed) as CPU tensors in [0,1].
    repeat composes the op that many times (see apply_operation).
    Caller is responsible for setting models to eval mode beforehand."""
    mu, _ = vae.encoder(imgs)
    imgs_recon = vae.decoder(inv_proj(proj(mu))).clamp(0, 1).cpu()
    imgs_xform = vae.decoder(apply_operation(mu, proj, trans_op, inv_proj, repeat=repeat)).clamp(0, 1).cpu()
    return imgs.cpu(), imgs_recon, imgs_xform


@torch.no_grad()
def transition_accuracy(vae, proj, trans_op, inv_proj, classifier, mu, labels, k, n_classes=10):
    """Apply op^k in latent space, decode, classify; score against the shifted label.

    mu:     (B, nd) encoded latents (use the encoder mean, not a sample)
    labels: (B,) true source classes
    k:      number of times to compose the operator (k=0 = recon path, no op)
    Returns (frac_correct, preds, targets) where targets = (labels + k) % n_classes.
    The fraction is the op^k transition accuracy: did composing the op k times
    advance the class k steps? (k=0 anchors pipeline fidelity vs the classifier ceiling.)
    """
    z = apply_operation(mu, proj, trans_op, inv_proj, repeat=k)
    decoded = vae.decoder(z).clamp(0, 1)
    preds = classifier(decoded).argmax(1)
    targets = (labels.to(preds.device) + k) % n_classes
    return (preds == targets).float().mean().item(), preds, targets


@torch.no_grad()
def run_demo(ckpt_path, out_dir=".", n_per_class=10, device=None):
    """Full pipeline: load checkpoint, encode MNIST grid, transform, decode, save PNGs."""
    proj, trans_op, inv_proj, device, dataset = load_for_inference(ckpt_path, device)
    vae = load_vae(dataset, device=str(device))
    imgs = make_class_ordered_images(dataset, n_per_class).to(device)

    imgs_input, imgs_recon, imgs_xform = make_viz_grids(vae, proj, trans_op, inv_proj, imgs)

    os.makedirs(out_dir, exist_ok=True)
    for tensor, fname in [(imgs_input, "mnist_input.png"), (imgs_recon, "mnist_recon.png"), (imgs_xform, "mnist_transformed.png")]:
        save_image(make_grid(tensor, nrow=10), os.path.join(out_dir, fname))
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
