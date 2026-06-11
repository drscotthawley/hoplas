#!/usr/bin/env python3
"""Supervised learning of nd-dimensional rotations via four model variants."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hoplas.filmr import FiLMR, MatOp, MatOp2, get_rot_nd
from hoplas.ph_layers import PHMLinear


def make_dataset(n_samples: int, nd: int, device: torch.device):
    u = torch.randn(nd, device=device)
    v = torch.randn(nd, device=device)
    R = get_rot_nd(u, v)
    x = torch.randn(n_samples, nd, device=device)
    y = x @ R.T   # row-vector convention
    return x.cpu(), y.cpu()


def build_model(method: str, nd: int, ph_n: int) -> nn.Module:
    if method == "filmr":
        return FiLMR(nd=nd)
    elif method == "matop":
        return MatOp(nd=nd)
    elif method == "matop2":
        return MatOp2(nd=nd)
    elif method == "ph":
        if nd % ph_n != 0:
            raise ValueError(f"nd={nd} must be divisible by ph_n={ph_n} for PHMLinear")
        return PHMLinear(n=ph_n, in_features=nd, out_features=nd)
    raise ValueError(f"Unknown method: {method}")


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if not args.cpu else torch.device("cpu")
    print(f"device={device}  method={args.method}  nd={args.nd}")

    x, y = make_dataset(args.n_samples, args.nd, device)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

    model = build_model(args.method, args.nd, args.ph_n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"epoch {epoch:4d}/{args.epochs}  loss={total_loss / len(x):.6f}")


def main():
    p = argparse.ArgumentParser(description="Learn nd rotations with different model variants.")
    p.add_argument("--method", choices=["filmr", "matop", "matop2", "ph"], default="filmr")
    p.add_argument("--nd", type=int, default=4, help="Dimension of the space")
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ph-n", type=int, default=2,
                   help="Hypercomplex order n for PHMLinear (nd must be divisible by n)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    main_args = p.parse_args()
    train(main_args)


if __name__ == "__main__":
    main()
