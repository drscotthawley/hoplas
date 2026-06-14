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

There's also a confusion-analysis mode (--confusion-k) that answers "is the plateau
structure the OPERATOR or the JUDGE?" by deconvolving the fixed-k confusion matrix
against the k=0 (recon) confusion kernel. See confusion_analysis() below.

Usage:
    python scripts/score_transitions.py checkpoints/mnist_filmr_expm.pt
    python scripts/score_transitions.py CKPT --max-k 10 --n-samples 5000
    python scripts/score_transitions.py CKPT --confusion-k 76 77 78   # operator-vs-judge
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

from hoplas.classifier import load_classifier
from hoplas.inference import apply_operation, load_for_inference, transition_accuracy
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


def _effective_n(counts):
    """Effective number of predicted classes = exp(entropy) of the pred histogram.
    n_classes = perfectly uniform output; →1 = collapsed onto a single class."""
    p = np.asarray(counts, dtype=float)
    s = p.sum()
    if s == 0:
        return 0.0
    p = p[p > 0] / s
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def score(args):
    proj, trans_op, inv_proj, device, dataset = load_for_inference(args.checkpoint, args.device)
    vae = load_vae(dataset, device=str(device))
    clf_key = "cifar10" if dataset.startswith("cifar") else "mnist"
    classifier = load_classifier(clf_key, device=str(device))

    n_classes = args.n_classes
    max_k = args.max_k if args.max_k is not None else n_classes
    loader = _test_loader(dataset, args.n_samples, args.batch_size)

    # accumulate correct/total and the pred-class histogram per k across batches
    correct = {k: 0 for k in range(max_k + 1)}
    pred_hist = {k: np.zeros(n_classes) for k in range(max_k + 1)}
    total = 0
    for imgs, labels in loader:
        mu, _ = vae.encoder(imgs.to(device))
        for k in range(max_k + 1):
            acc_k, preds, targets = transition_accuracy(
                vae, proj, trans_op, inv_proj, classifier, mu, labels, k, n_classes)
            correct[k] += int((preds == targets).sum().item())
            np.add.at(pred_hist[k], preds.cpu().numpy(), 1)
        total += labels.size(0)

    accs = [correct[k] / total for k in range(max_k + 1)]
    eff = [_effective_n(pred_hist[k]) for k in range(max_k + 1)]    # output diversity vs k
    print(f"\nop^k transition accuracy  (n={total}, dataset={dataset})")
    print(f"  {'k':>3}  {'target':>8}  {'acc':>7}  {'eff#cls':>7}")
    for k in range(max_k + 1):
        tag = " (recon)" if k == 0 else ""
        print(f"  {k:>3}  {'(i+%d)%%%d' % (k, n_classes):>8}  {accs[k]*100:6.2f}%  {eff[k]:7.2f}{tag}")

    if not args.no_plot:
        ckpt_name = os.path.basename(args.checkpoint)
        _plot(accs, eff, max_k, dataset, args.out, n_classes, ckpt_name)
    return accs, eff


def _plot(accs, eff, max_k, dataset, out, n_classes=10, ckpt_name=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = list(range(max_k + 1))
    # shrink markers as the curve gets long so points don't merge into a blob
    ms = 7 if max_k <= 25 else (4 if max_k <= 60 else 3)
    fig, (ax, axd) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]},
        figsize=(8, 6.5) if max_k > 25 else (6, 6))

    # ── accuracy panel ──
    ax.plot(ks, [a * 100 for a in accs], "o-", lw=1.5, ms=ms, color="#2a6")
    mults = [k for k in ks if k > 0 and k % n_classes == 0]
    if mults:
        ax.plot(mults, [accs[k] * 100 for k in mults], "o", ms=ms + 3,
                mfc="none", mec="#c33", mew=1.5, label=f"k ≡ 0 (mod {n_classes}) — closure")
    ax.axhline(accs[0] * 100, ls="--", c="gray", lw=1,
               label=f"k=0 anchor (recon, {accs[0]*100:.1f}%)")
    ax.axhline(100 / n_classes, ls=":", c="lightgray", lw=1, label=f"chance ({100/n_classes:.0f}%)")
    ax.set_ylabel("transition accuracy\nP(pred = (i+k) mod n)  [%]")
    ax.set_title(f"op^k transition accuracy — {dataset}")
    if ckpt_name:
        ax.text(0.5, 1.01, ckpt_name, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="gray", style="italic")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    # ── output-diversity panel (collapse detector) ──
    axd.plot(ks, eff, "s-", lw=1.5, ms=ms, color="#36c")
    axd.axhline(n_classes, ls=":", c="lightgray", lw=1, label=f"uniform ({n_classes})")
    axd.axhline(1, ls=":", c="lightgray", lw=1, label="collapsed (1)")
    axd.set_ylabel("output diversity\n(eff. # classes)")
    axd.set_xlabel("k  (operator compositions)")
    axd.set_ylim(0, n_classes + 0.5)
    axd.legend(loc="lower left", fontsize=8)
    axd.grid(alpha=0.3)
    # adaptive ticks: every 1 when short, else a clean step landing on multiples of n_classes
    step = 1 if max_k <= 20 else (n_classes if max_k <= 120 else 2 * n_classes)
    axd.set_xticks(list(range(0, max_k + 1, step)))

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nsaved plot -> {out}")


# ───────────────────────── confusion / displacement analysis ─────────────────────────
# "Is the plateau structure the OPERATOR or the JUDGE?"  At a fixed k, capture the full
# confusion matrix C_k[i,pred] and deconvolve it against the k=0 (recon) matrix M, which
# IS the classifier's own confusion kernel on decoded images.  Model:
#     C_k[i,pred]  ≈  Σ_d w(d) · M[(i+d) mod n, pred]
# i.e. "the op displaced each source class i by a distribution w(d) over displacements,
# THEN the same judge confused the result per M."  A SHARP w(d) ⇒ the op has a definite
# displacement and the spread in C_k is just classifier confusion (the user's read).
# A BROAD w(d) ⇒ the op genuinely scattered points.  Either way w is judge-deconvolved.

def _row_norm(counts):
    """Rows -> probability distributions (empty rows left as zeros)."""
    s = counts.sum(1, keepdims=True)
    return counts / np.where(s == 0, 1, s)


def _confusion_counts(vae, proj, trans_op, inv_proj, classifier, loader, ks, n_classes, device):
    """Return {k: (n_classes, n_classes) int count matrix C_k[true_i, pred]} for each k in ks."""
    mats = {k: np.zeros((n_classes, n_classes), dtype=np.int64) for k in ks}
    for imgs, labels in loader:
        mu, _ = vae.encoder(imgs.to(device))
        lab = labels.numpy()
        for k in ks:
            z = apply_operation(mu, proj, trans_op, inv_proj, repeat=k)
            preds = classifier(vae.decoder(z).clamp(0, 1)).argmax(1).cpu().numpy()
            np.add.at(mats[k], (lab, preds), 1)
    return mats


def _fit_displacement(C, M, n):
    """Non-negative least squares for w in  C[i,pred] ≈ Σ_d w(d)·M[(i+d)%n, pred].
    Returns w normalized to sum 1, plus the fitted matrix and relative residual."""
    A = np.zeros((n * n, n))
    for i in range(n):
        for pred in range(n):
            for d in range(n):
                A[i * n + pred, d] = M[(i + d) % n, pred]
    b = C.reshape(-1)
    try:
        from scipy.optimize import nnls
        w, _ = nnls(A, b)
    except Exception:                       # no scipy -> lstsq then clip negatives
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
        w = np.clip(w, 0, None)
    s = w.sum()
    w = w / s if s > 0 else w
    C_hat = (A @ w).reshape(n, n)
    resid = np.abs(C_hat - C).sum() / max(np.abs(C).sum(), 1e-12)
    return w, C_hat, resid


@torch.no_grad()
def confusion_analysis(args):
    proj, trans_op, inv_proj, device, dataset = load_for_inference(args.checkpoint, args.device)
    vae = load_vae(dataset, device=str(device))
    clf_key = "cifar10" if dataset.startswith("cifar") else "mnist"
    classifier = load_classifier(clf_key, device=str(device))
    n = args.n_classes

    ks = sorted(set([0] + list(args.confusion_k)))   # k=0 is always needed as the kernel M
    loader = _test_loader(dataset, args.n_samples, args.batch_size)
    counts = _confusion_counts(vae, proj, trans_op, inv_proj, classifier, loader, ks, n, device)

    M = _row_norm(counts[0].astype(float))           # judge confusion kernel (recon)
    ckpt_name = os.path.basename(args.checkpoint)
    for k in args.confusion_k:
        C = _row_norm(counts[k].astype(float))
        w, C_hat, resid = _fit_displacement(C, M, n)
        d_star = int(np.argmax(w))
        ent = -np.sum([p * np.log(p) for p in w if p > 0])      # nats; 0 = a delta, ln(n) = uniform
        print(f"\n── confusion analysis  k={k}  (dataset={dataset}, n={counts[k].sum()}) ──")
        print(f"  fitted displacement distribution w(d)  [op displaced source by d, then judge confused]:")
        for d in range(n):
            bar = "█" * int(round(w[d] * 40))
            star = "  <- peak" if d == d_star else ""
            print(f"    d={d:>2}  {w[d]*100:6.2f}%  {bar}{star}")
        print(f"  peak displacement d*={d_star}   target-for-this-k = {k % n}   "
              f"w-entropy={ent:.3f} nats (0=sharp, ln{n}={np.log(n):.2f}=uniform)   "
              f"deconv residual={resid*100:.1f}%")
        marg = counts[k].sum(0).astype(float); marg = marg / marg.sum()
        eff_k = _effective_n(counts[k].sum(0))
        top = np.argsort(marg)[::-1][:3]
        print(f"  output diversity: eff #classes = {eff_k:.2f} / {n}   "
              f"most-predicted: " + ", ".join(f"{c}:{marg[c]*100:.0f}%" for c in top))
        sharp = "SHARP -> operator has a definite displacement; spread in C_k is the JUDGE" if ent < 0.5 * np.log(n) \
                else "BROAD -> operator genuinely SCATTERED points (not just confusion)"
        print(f"  verdict: {sharp}")
        if not args.no_plot:
            out = args.confusion_out or f"confusion_k{k}.png"
            _plot_confusion(M, C, C_hat, w, k, n, dataset, ckpt_name, out)


def _plot_confusion(M, C, C_hat, w, k, n, dataset, ckpt_name, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, mat, title in [(axes[0, 0], M, "M = judge kernel (k=0, recon)"),
                           (axes[0, 1], C, f"C_k observed (k={k})"),
                           (axes[1, 0], C_hat, "C_hat = Σ w(d)·M[(i+d)] (fit)")]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xlabel("predicted class"); ax.set_ylabel("true source i")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axw = axes[1, 1]
    axw.bar(range(n), w * 100, color="#2a6")
    axw.axvline(k % n, ls="--", c="#c33", lw=1.5, label=f"target d = k mod n = {k % n}")
    axw.set_xlabel("displacement d = (pred - i) mod n"); axw.set_ylabel("w(d)  [%]")
    axw.set_xticks(range(n)); axw.set_title("fitted displacement w(d)", fontsize=10)
    axw.legend(fontsize=8); axw.grid(alpha=0.3)
    fig.suptitle(f"operator-vs-judge deconvolution — {dataset}  (k={k})\n{ckpt_name}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    print(f"  saved plot -> {out}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Score op^k transition accuracy with the pixel classifier.")
    p.add_argument("checkpoint",     help="path to .pt checkpoint from train_ring.py")
    p.add_argument("--max-k",        type=int, default=None, help="highest k to evaluate (default: n_classes, closes the ring)")
    p.add_argument("--n-classes",    type=int, default=10,   help="number of classes (ring size)")
    p.add_argument("--n-samples",    type=int, default=5000, help="test images to score (0 = full test set)")
    p.add_argument("--batch-size",   type=int, default=512,  help="eval batch size")
    p.add_argument("--out",          type=str, default="op_k_transition.png", help="output plot path (transition curve)")
    p.add_argument("--no-plot",      action="store_true",    help="skip plotting, print tables only")
    p.add_argument("--device",       type=str, default=None, help="cuda/mps/cpu (default: auto)")
    p.add_argument("--confusion-k",  type=int, nargs="+", default=None, metavar="K",
                   help="run operator-vs-judge confusion analysis at these k (skips the transition curve). "
                        "k=0 is auto-included as the deconvolution kernel. e.g. --confusion-k 76 77 78")
    p.add_argument("--confusion-out", type=str, default=None,
                   help="confusion plot path (default: confusion_k{K}.png per k)")
    args = p.parse_args()
    if args.confusion_k:
        confusion_analysis(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
