#!/usr/bin/env python3
"""Supervised learning of nd-dimensional rotations via four model variants."""

import argparse
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hoplas.filmr import FiLMR, FiLMR_expm, MatOp, MatOp2, get_rot_nd
from hoplas.ph_layers import PHMLinear


def make_dataset(n_samples: int, nd: int, device: torch.device):
    u = torch.randn(nd, device=device)
    v = torch.randn(nd, device=device)
    R = get_rot_nd(u, v)
    x = torch.randn(n_samples, nd, device=device)
    y = x @ R.T   # row-vector convention
    return x.cpu(), y.cpu()


def build_model(method: str, nd: int, order: int) -> nn.Module:
    if method == "filmr":
        return FiLMR(nd=nd)
    elif method == "filmr_expm":
        return FiLMR_expm(nd=nd)
    elif method == "matop":
        return MatOp(nd=nd)
    elif method == "matop2":
        return MatOp2(nd=nd)
    elif method == "ph":
        if nd % order != 0:
            raise ValueError(f"nd={nd} must be divisible by order={order} for PHMLinear")
        return PHMLinear(n=order, in_features=nd, out_features=nd)
    raise ValueError(f"Unknown method: {method}")


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if not args.cpu else torch.device("cpu")
    print(f"device={device}  method={args.method}  nd={args.nd}")

    run_name = f"{args.method}_{args.order}" if args.method == "ph" else args.method
    wandb.init(project="simple rot", name=run_name, config=vars(args))

    x, y = make_dataset(args.n_samples, args.nd, device)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

    model = build_model(args.method, args.nd, args.order).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        avg_loss = total_loss / len(x)
        print(f"epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}")
        wandb.log({"epoch": epoch, "loss": avg_loss})

    wandb.finish()


def main():
    p = argparse.ArgumentParser(description="Learn nd rotations with different model variants.")
    p.add_argument("--method", choices=["filmr", "filmr_expm", "matop", "matop2", "ph"], default="filmr")
    p.add_argument("--nd", type=int, default=64, help="Dimension of the space")
    p.add_argument("--n-samples", type=int, default=40_000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--order", type=int, default=4,
                   help="Hypercomplex order n for PHMLinear (nd must be divisible by order)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    main_args = p.parse_args()
    train(main_args)


if __name__ == "__main__":
    main()
