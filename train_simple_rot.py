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


def make_dataset(n_samples: int, nd: int, device: torch.device,
                 corr: float = 0.0, corr_nd: int = 1):
    """corr in [0,1]: correlation strength among the correlated channels.
    corr_nd in [1,nd]: number of channels that share the correlation
    (e.g. 32 -> first 32 of 64 channels correlated, rest iid).
    Each channel keeps unit marginal variance."""
    u = torch.randn(nd, device=device)
    v = torch.randn(nd, device=device)
    R = get_rot_nd(u, v)
    k = corr_nd                                          # number of correlated channels
    z = torch.randn(n_samples, nd, device=device)        # per-channel noise
    s = torch.randn(n_samples, 1, device=device)         # shared component
    x = z.clone()
    x[:, :k] = (1 - corr) ** 0.5 * z[:, :k] + corr ** 0.5 * s   # Var=1, Corr=corr within block
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

    model_name = f"{args.method}_{args.order}" if args.method == "ph" else args.method
    run_name = f"{model_name}_corr{args.corr}_nd{args.corr_nd}"
    wandb.init(project="simple rot", name=run_name, config=vars(args))
    loss_key = f"loss_corr_nd={args.corr_nd}"

    x, y = make_dataset(args.n_samples, args.nd, device, args.corr, args.corr_nd)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

    model = build_model(args.method, args.nd, args.order).to(device)
    # weight-decay only on >=2D weights (e.g. FiLMR_expm.W); never on gamma/beta/bias.
    # For expm this pulls W toward the minimal-angle generator, keeping matrix_exp
    # accurate and preventing the slow precision drift after convergence.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr)
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
        wandb.log({"epoch": epoch, loss_key: avg_loss})

    wandb.finish()


def main():
    p = argparse.ArgumentParser(description="Learn nd rotations with different model variants.")
    p.add_argument("--method", choices=["filmr", "filmr_expm", "matop", "matop2", "ph"], default="filmr")
    p.add_argument("--nd", type=int, default=64, help="Dimension of the space")
    p.add_argument("--n-samples", type=int, default=40_000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay on >=2D weights (helps FiLMR_expm precision drift)")
    p.add_argument("--order", type=int, default=4,
                   help="Hypercomplex order n for PHMLinear (nd must be divisible by order)")
    p.add_argument("--corr", type=float, default=0.9,
                   help="Correlation strength in [0,1] among the correlated channels")
    p.add_argument("--corr-nd", type=int, default=1,
                   help="Number of channels that share the correlation (1..nd)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    main_args = p.parse_args()
    if not 0.0 <= main_args.corr <= 1.0:
        p.error("--corr must be in [0, 1]")
    if not 1 <= main_args.corr_nd <= main_args.nd:
        p.error(f"--corr-nd must be in [1, {main_args.nd}]")
    train(main_args)


if __name__ == "__main__":
    main()
