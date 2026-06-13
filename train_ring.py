#!/usr/bin/env python3
"""Ring-task training via four model variants. (dataset/models WIP)"""

import argparse
import os
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hoplas.filmr import FiLMR, FiLMR_expm, MatOp, MatOp2
from hoplas.ph_layers import PHMLinear
from hoplas.data import LineDataset, MNISTEncodingsDataset
from hoplas.models import Projector
from hoplas.losses import SIGReg
from hoplas.viz import embedding_scatter3d


class build_op(nn.Module):
    """Builds the transform op, optionally as a residual x + op(x).
    op_resid centers the transform at identity (good for many ring points;
    redundant for filmr_expm, already near-identity via matrix_exp)."""
    def __init__(self, method: str, nd: int, order: int, op_resid: bool = False, rank: int = 2,
                 unit_norm: bool = True):
        super().__init__()
        self.op_resid = op_resid
        self.unit_norm = unit_norm    # L2-normalize output onto unit sphere (matches projector)
        if method == "filmr":
            self.op = FiLMR(nd=nd)
        elif method == "filmr_expm":
            self.op = FiLMR_expm(nd=nd, rank=rank)
        elif method == "matop":
            self.op = MatOp(nd=nd)
        elif method == "matop2":
            self.op = MatOp2(nd=nd)
        elif method == "ph":
            if nd % order != 0:
                raise ValueError(f"nd={nd} must be divisible by order={order} for PHMLinear")
            self.op = PHMLinear(n=order, in_features=nd, out_features=nd)
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, x):
        out = x + self.op(x) if self.op_resid else self.op(x)
        return F.normalize(out, dim=-1) if self.unit_norm else out


@torch.no_grad()
def evaluate(loader, proj, trans_op, inv_proj, sim_fn, device, epoch, args):
    """Run the same losses over a held-out loader (no grad) for generalization metrics."""
    proj.eval(); trans_op.eval(); inv_proj.eval()
    n = tot_loss = tot_sim = tot_sigreg = tot_recon = 0.0
    for x, y in loader:
        xb, yb = x['data'].to(device), y['data'].to(device)
        xproj, yproj = proj(xb), proj(yb)
        xproj_t = trans_op(xproj)
        xprime = inv_proj(xproj)
        sim = sim_fn(xproj_t, yproj)
        sigreg = SIGReg(torch.cat([xproj_t, yproj], dim=0), global_step=epoch)
        recon = sim_fn(xprime, xb)
        loss = (1 - args.lambd) * sim + args.lambd * sigreg + args.lambda_recon * recon
        bs = xb.size(0)
        tot_loss += loss.item() * bs; tot_sim += sim.item() * bs
        tot_sigreg += sigreg.item() * bs; tot_recon += recon.item() * bs; n += bs
    proj.train(); trans_op.train(); inv_proj.train()
    return tot_loss / n, tot_sim / n, tot_sigreg / n, tot_recon / n


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if not args.cpu else torch.device("cpu")
    if args.dataset == "line":
        dataset = LineDataset(nd=args.nd, npoints=args.npoints, noise=args.noise)
        val_dataset = LineDataset(nd=args.nd, npoints=args.npoints, noise=args.noise, debug=False, len=5000)
    else:  # mnist: the encodings dictate nd (overrides --nd)
        dataset = MNISTEncodingsDataset(split="train")
        val_dataset = MNISTEncodingsDataset(split="test", debug=False)
        args.nd = dataset.nd
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"device={device}  op={args.op}  dataset={args.dataset}  nd={args.nd}")

    run_name = f"{args.op}_{args.order}" if args.op == "ph" else args.op
    run_name = f"{args.dataset}_{run_name}"
    run_name = f"{run_name}_{args.tag}" if args.tag else run_name
    project = "ring" if args.dataset == "line" else f"ring-{args.dataset}"
    if not args.no_wandb:
        wandb.init(project=project, name=run_name, config=vars(args))

    proj = Projector(nd=args.nd, n_hid=args.n_hid, n_layers=args.proj_layers, proj_resid=args.proj_resid, unit_norm=args.unit_norm).to(device)
    trans_op = build_op(args.op, args.nd, args.order, args.op_resid, args.rank, args.unit_norm).to(device)
    # inverse projector: maps embedding back to original space (unit_norm=False: output isn't on the sphere)
    inv_proj = Projector(nd=args.nd, n_hid=args.n_hid, n_layers=args.proj_layers, unit_norm=False).to(device)

    n_proj = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    n_op = sum(p.numel() for p in trans_op.parameters() if p.requires_grad)
    n_inv = sum(p.numel() for p in inv_proj.parameters() if p.requires_grad)
    print(f"trainable params: projector={n_proj}  trans_op={n_op}  inv_proj={n_inv}")

    # weight-decay only on >=2D weights (proj Linears, FiLMR_expm.W); never on
    # gamma/beta/bias/norms. For expm this also pulls W toward the minimal-angle
    # generator, keeping matrix_exp accurate and preventing slow precision drift.
    # The op gets its own lr (--op-lr): slowing the op relative to the projector
    # lets the rotation angle climb gently and lock onto the first (single-ring)
    # closure sheet instead of overshooting it.
    op_lr = args.op_lr if args.op_lr is not None else args.lr
    other_params = list(proj.parameters()) + list(inv_proj.parameters())
    op_params = list(trans_op.parameters())
    split = lambda ps: ([p for p in ps if p.ndim >= 2], [p for p in ps if p.ndim < 2])
    other_decay, other_nodecay = split(other_params)
    op_decay, op_nodecay = split(op_params)
    optimizer = torch.optim.AdamW([
        {"params": other_decay,  "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": other_nodecay, "weight_decay": 0.0,             "lr": args.lr},
        {"params": op_decay,     "weight_decay": args.weight_decay, "lr": op_lr},
        {"params": op_nodecay,   "weight_decay": 0.0,             "lr": op_lr},
    ], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.lr / 500)
    sim_fn = nn.MSELoss()
    sim_ema = None  # smoothed sim for the scheduler (sigreg flattens, so don't anneal on total)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{run_name}.pt")
    best_val = float("inf")  # save the checkpoint with the lowest val loss

    try:
        for epoch in range(1, args.epochs + 1):
            total_loss = 0.0
            total_sim = 0.0
            total_sigreg = 0.0
            total_recon = 0.0
            pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
            for x, y in pbar:
                xb, yb = x['data'].to(device), y['data'].to(device)
                optimizer.zero_grad()
                xproj, yproj = proj(xb), proj(yb)  # project into new space
                xproj_t = trans_op(xproj)          # transform/rotate
                xprime = inv_proj(xproj)           # reconstruct original from embedding
                sim_loss = sim_fn(xproj_t, yproj) # pull toward next one in sequence
                sigreg_loss = SIGReg( torch.cat([xproj_t, yproj], dim=0), global_step=epoch )  # pull distribution toward Gaussian
                recon_loss = sim_fn(xprime, xb)   # autoencoder: inv_proj(proj(x)) ~ x
                loss = (1 - args.lambd) * sim_loss + args.lambd * sigreg_loss + args.lambda_recon * recon_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                total_sim += sim_loss.item() * xb.size(0)
                total_sigreg += sigreg_loss.item() * xb.size(0)
                total_recon += recon_loss.item() * xb.size(0)
                pbar.set_postfix(loss=f"{loss.item():.6f}")
            avg_loss = total_loss / len(dataset)
            avg_sim = total_sim / len(dataset)
            avg_sigreg = total_sigreg / len(dataset)
            avg_recon = total_recon / len(dataset)
            sim_ema = avg_sim if sim_ema is None else args.sim_ema * sim_ema + (1 - args.sim_ema) * avg_sim
            scheduler.step(sim_ema)  # anneal on smoothed sim, not total (sigreg flattens by design)
            op_angle = trans_op.op.rotation_angle_deg() if args.op == "filmr_expm" else None
            angle_str = f"  angle={op_angle:.2f}deg" if op_angle is not None else ""
            print(f"epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}  sim={avg_sim:.6f}  sigreg={avg_sigreg:.6f}  recon={avg_recon:.6f}{angle_str}")

            do_val = args.val_every and (epoch % args.val_every == 0 or epoch == 1)
            if do_val:
                val_loss, val_sim, val_sigreg, val_recon = evaluate(
                    val_loader, proj, trans_op, inv_proj, sim_fn, device, epoch, args)
                print(f"      val  loss={val_loss:.6f}  sim={val_sim:.6f}  sigreg={val_sigreg:.6f}  recon={val_recon:.6f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"epoch": epoch, "val_loss": val_loss, "args": vars(args),
                                "proj": proj.state_dict(), "inv_proj": inv_proj.state_dict(),
                                "trans_op": trans_op.state_dict()}, ckpt_path)

            if wandb.run is not None:
                log = {"epoch": epoch, "loss": avg_loss, "lr": optimizer.param_groups[0]["lr"],
                       "sim_loss": avg_sim, "sim_ema": sim_ema, "sigreg_loss": avg_sigreg,
                       "recon_loss": avg_recon}
                if do_val:
                    log.update({"val_loss": val_loss, "val_sim_loss": val_sim,
                                "val_sigreg_loss": val_sigreg, "val_recon_loss": val_recon})
                if op_angle is not None:
                    log["op_angle_deg"] = op_angle
                log["embedding"] = embedding_scatter3d(  # last batch's projections
                    yproj, xproj_t, epoch, args.op, args.order,
                    yproj_labels=y['label'], xproj_t_labels=x['label'])
                wandb.log(log)
    except KeyboardInterrupt:
        print("\ninterrupted — finishing run")
    finally:
        if wandb.run is not None:
            wandb.finish()


def main():
    p = argparse.ArgumentParser(description="Ring task with different model variants.")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--dataset", choices=["line", "mnist"], default="line",
                   help="line=synthetic ring; mnist=VAE encodings (nd forced to 16)")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr-patience", type=int, default=20, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--lambd", type=float, default=0.01,
                   help="SIGReg weight: loss = (1-lambd)*sim + lambd*sigreg")
    p.add_argument("--lambda-recon", type=float, default=1.0,
                   help="Weight on inv_proj autoencoder reconstruction loss (0 disables)")
    p.add_argument("--n-hid", type=int, default=32, help="Projector hidden dim")
    p.add_argument("--nd", type=int, default=3, help="Dimension of the space")
    p.add_argument("--npoints", type=int, default=12, help="Number of quantized points on the line")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--noise", type=float, default=0.01, help="Jitter added to each point")
    p.add_argument("--op", choices=["filmr", "filmr_expm", "matop", "matop2", "ph"], default="filmr_expm")
    p.add_argument("--op-lr", type=float, default=None,
                   help="Separate LR for trans_op (default: same as --lr). Lower it to slow the angle's climb.")
    p.add_argument("--op-resid", action=argparse.BooleanOptionalAction, default=True,
                   help="Wrap trans_op as x + op(x) (centers transform at identity; good for many ring points)")
    p.add_argument("--order", type=int, default=4,
                   help="Hypercomplex order n for PHMLinear (nd must be divisible by order)")
    p.add_argument("--proj-layers", type=int, default=3, help="Projector number of layers")
    p.add_argument("--proj-resid", action="store_true",
                   help="Global nd->nd skip in Projector (learn perturbation of identity)")
    p.add_argument("--rank", type=int, default=2,
                   help="Rotation-plane rank for filmr_expm generator (even, <=nd; 2=single plane)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim-ema", type=float, default=0.8,
                   help="EMA decay for smoothing sim before the LR scheduler")
    p.add_argument("--tag", type=str, default="", help="tag to append to wandb run name")
    p.add_argument("--unit-norm", action=argparse.BooleanOptionalAction, default=True,
                   help="L2-normalize projector output onto the unit sphere")
    p.add_argument("--val-every", type=int, default=5,
                   help="Evaluate on held-out split every N epochs (0 disables)")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay on >=2D weights (helps FiLMR_expm precision drift)")
    main_args = p.parse_args()
    print("Arguments:", vars(main_args))
    train(main_args)


if __name__ == "__main__":
    main()
