#!/usr/bin/env python3
"""Score a trained hoplas operator with the pixel-space classifier, and plot the
op^k transition-accuracy curve.

For each k = 0..n_classes, compose the learned operator k times in projected
space, decode, and ask the classifier: did the class advance k steps?
  transition_acc(k) = P( classifier(decode(op^k(z))) == (true_label + k) mod n )

This is the behavioral proof of the ring/closure algebra: a clean staircase that
stays high across k (and ideally returns near the k=0 anchor at k=n_classes)
demonstrates the cyclic structure lives in the operator. The k=0 point measures
pipeline fidelity (recon, no op) and should sit near the classifier's clean
accuracy ceiling.

Usage:
    python scripts/score_transitions.py checkpoints/mnist_filmr_expm.pt
    python scripts/score_transitions.py CKPT --max-k 10 --n-samples 5000
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

from hoplas.classifier import load_classifier
from hoplas.inference import load_for_inference, transition_accuracy
from hoplas.vae import load_vae

_ROOTS = {"mnist": os.path.expanduser("~/datasets/mnist"),
          "cifar": os.path.expanduser("~/datasets/cifar10"),
          "cifar10": os.path.expanduser("~/datasets/cifar10")}


def _test_loader(dataset, n_samples, batch_size):
    cls = MNIST if dataset == "mnist" else CIFAR10
    ds = cls(root=_ROOTS[dataset], train=False, download=True, transform=ToTensor())
    if n_samples and n_samples < len(ds):
        # deterministic subset for reproducible numbers
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=g)[:n_samples].tolist()
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


@torch.no_grad()
def score(args):
    proj, trans_op, inv_proj, device, dataset = load_for_inference(args.checkpoint, args.device)
    vae = load_vae(dataset, device=str(device))
    clf_key = "cifar10" if dataset.startswith("cifar") else "mnist"
    classifier = load_classifier(clf_key, device=str(device))

    n_classes = args.n_classes
    max_k = args.max_k if args.max_k is not None else n_classes
    loader = _test_loader(dataset, args.n_samples, args.batch_size)

    # accumulate correct/total per k across batches
    correct = {k: 0 for k in range(max_k + 1)}
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        mu, _ = vae.encoder(imgs)
        for k in range(max_k + 1):
            acc_k, preds, targets = transition_accuracy(
                vae, proj, trans_op, inv_proj, classifier, mu, labels, k, n_classes)
            correct[k] += int((preds == targets).sum().item())
        total += labels.size(0)

    accs = [correct[k] / total for k in range(max_k + 1)]
    print(f"\nop^k transition accuracy  (n={total}, dataset={dataset})")
    print(f"  {'k':>3}  {'target':>6}  {'acc':>7}")
    for k in range(max_k + 1):
        tag = " (recon)" if k == 0 else ""
        print(f"  {k:>3}  {'(i+%d)%%%d' % (k, n_classes):>8}  {accs[k]*100:6.2f}%{tag}")

    if not args.no_plot:
        ckpt_name = os.path.basename(args.checkpoint)
        _plot(accs, max_k, dataset, args.out, n_classes, ckpt_name)
    return accs


def _plot(accs, max_k, dataset, out, n_classes=10, ckpt_name=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = list(range(max_k + 1))
    # shrink markers as the curve gets long so points don't merge into a blob
    ms = 7 if max_k <= 25 else (4 if max_k <= 60 else 3)
    fig, ax = plt.subplots(figsize=(8, 4) if max_k > 25 else (6, 4))
    ax.plot(ks, [a * 100 for a in accs], "o-", lw=1.5, ms=ms, color="#2a6")
    # mark the closure points (multiples of n_classes) where the ring should peak
    mults = [k for k in ks if k > 0 and k % n_classes == 0]
    if mults:
        ax.plot(mults, [accs[k] * 100 for k in mults], "o", ms=ms + 3,
                mfc="none", mec="#c33", mew=1.5, label=f"k ≡ 0 (mod {n_classes}) — closure")
    ax.axhline(accs[0] * 100, ls="--", c="gray", lw=1,
               label=f"k=0 anchor (recon, {accs[0]*100:.1f}%)")
    ax.axhline(100 / n_classes, ls=":", c="lightgray", lw=1, label=f"chance ({100/n_classes:.0f}%)")
    # adaptive ticks: every 1 when short, else a clean step that lands on multiples of n_classes
    step = 1 if max_k <= 20 else (n_classes if max_k <= 120 else 2 * n_classes)
    ax.set_xticks(list(range(0, max_k + 1, step)))
    ax.set_xlabel("k  (operator compositions)")
    ax.set_ylabel("transition accuracy  P(pred = (i+k) mod n)  [%]")
    ax.set_title(f"op^k transition accuracy — {dataset}")
    if ckpt_name:
        ax.text(0.5, 1.005, ckpt_name, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="gray", style="italic")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nsaved plot -> {out}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Score op^k transition accuracy with the pixel classifier.")
    p.add_argument("checkpoint",     help="path to .pt checkpoint from train_ring.py")
    p.add_argument("--max-k",        type=int, default=None, help="highest k to evaluate (default: n_classes, closes the ring)")
    p.add_argument("--n-classes",    type=int, default=10,   help="number of classes (ring size)")
    p.add_argument("--n-samples",    type=int, default=5000, help="test images to score (0 = full test set)")
    p.add_argument("--batch-size",   type=int, default=512,  help="eval batch size")
    p.add_argument("--out",          type=str, default="op_k_transition.png", help="output plot path")
    p.add_argument("--no-plot",      action="store_true",    help="skip plotting, print table only")
    p.add_argument("--device",       type=str, default=None, help="cuda/mps/cpu (default: auto)")
    args = p.parse_args()
    score(args)


if __name__ == "__main__":
    main()
