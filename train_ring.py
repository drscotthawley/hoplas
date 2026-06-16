#!/usr/bin/env python3
"""Ring-task training via four model variants. (dataset/models WIP)"""

import argparse
import configargparse
import os
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
from hoplas.data import LineDataset, EncodingsDataset
from hoplas.inference import make_class_ordered_images, make_viz_grids
from hoplas.models import Projector
from hoplas.ops import OpWrapper
from hoplas.losses import SIGReg, MomMatchLoss


def freeze_quaternion(ph_layer):
    """Fix ph_layer.a to the Hamilton quaternion multiplication table and freeze it."""
    assert ph_layer.n == 4, "freeze_quaternion requires --order 4"
    H = torch.tensor([
        [[ 1,  0,  0,  0], [ 0, -1,  0,  0], [ 0,  0, -1,  0], [ 0,  0,  0, -1]],
        [[ 0,  1,  0,  0], [ 1,  0,  0,  0], [ 0,  0,  0, -1], [ 0,  0,  1,  0]],
        [[ 0,  0,  1,  0], [ 0,  0,  0,  1], [ 1,  0,  0,  0], [ 0, -1,  0,  0]],
        [[ 0,  0,  0,  1], [ 0,  0, -1,  0], [ 0,  1,  0,  0], [ 1,  0,  0,  0]],
    ], dtype=ph_layer.a.dtype, device=ph_layer.a.device)
    ph_layer.a.data.copy_(H)
    ph_layer.a.requires_grad_(False)
from hoplas.vae import load_vae
from hoplas.viz import embedding_scatter3d


class WarmupThenPlateauWithReduction:
    """Linear LR warmup for the first `boundary` epochs, then ReduceLROnPlateau.

    SequentialLR can't hold ReduceLROnPlateau (it's metric-driven, not an LRScheduler),
    so this thin router does the phase switch instead. Call once per epoch:
        scheduler.step(metric, epoch)
    The warmup scheduler ignores `metric`; the plateau scheduler ignores `epoch`.
    """
    def __init__(self, warmup, plateau, boundary):
        self.warmup, self.plateau, self.boundary = warmup, plateau, boundary

    def step(self, metric, epoch):
        if self.warmup is not None and epoch <= self.boundary:
            self.warmup.step()
        else:
            self.plateau.step(metric)


@torch.no_grad()
def evaluate(loader, proj, trans_op, inv_proj, sim_fn, device, epoch, args, max_viz=0):
    """Run the same losses over a held-out loader (no grad) for generalization metrics.
    If max_viz > 0, also accumulates up to max_viz projected points for visualization."""
    proj.eval(); trans_op.eval(); inv_proj.eval()
    n = tot_loss = tot_sim = tot_mom = tot_sigreg = tot_recon = tot_va = tot_vb = 0.0
    viz_y, viz_xt, viz_yl, viz_xl, viz_n = [], [], [], [], 0
    for x, y in loader:
        xb, yb = x['data'].to(device), y['data'].to(device)
        xproj, yproj = proj(xb), proj(yb)
        xproj_t = trans_op(xproj)
        xprime, yprime = inv_proj(xproj), inv_proj(yproj)
        sim = sim_fn(xproj_t, yproj)
        mom, stats = MomMatchLoss(xproj_t, yproj, labels=x['label'].to(device),
                                  diag=args.mom_diag, cov_weight=args.mom_cov_weight, return_stats=True)
        sigreg = SIGReg(torch.cat([xproj_t, yproj], dim=0), global_step=epoch)
        recon = sim_fn(torch.cat([xprime, yprime]), torch.cat([xb, yb]))
        loss = (1 - args.lambd) * (args.lambda_sim * sim + args.lambda_mom * mom) + args.lambd * sigreg + args.lambda_recon * recon
        bs = xb.size(0)
        tot_loss += loss.item() * bs; tot_sim += sim.item() * bs; tot_mom += mom.item() * bs
        tot_sigreg += sigreg.item() * bs; tot_recon += recon.item() * bs
        tot_va += stats["var_a"] * bs; tot_vb += stats["var_b"] * bs; n += bs
        if max_viz > 0 and viz_n < max_viz:
            viz_y.append(yproj); viz_xt.append(xproj_t)
            viz_yl.append(y['label']); viz_xl.append(x['label'])
            viz_n += bs
    proj.train(); trans_op.train(); inv_proj.train()
    losses = (tot_loss / n, tot_sim / n, tot_mom / n, tot_sigreg / n, tot_recon / n)
    var_stats = {"var_xproj_t": tot_va / n, "var_yproj": tot_vb / n}
    viz = (torch.cat(viz_y), torch.cat(viz_xt), torch.cat(viz_yl), torch.cat(viz_xl)) if viz_y else None
    return losses, var_stats, viz


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if not args.cpu else torch.device("cpu")
    _PT_PATHS = {"mnist": "~/datasets/mnist_latents.pt", "cifar": "~/datasets/cifar_latents.pt"}
    if args.dataset == "line":
        dataset = LineDataset(nd=args.nd, npoints=args.npoints, noise=args.noise)
        val_dataset = LineDataset(nd=args.nd, npoints=args.npoints, noise=args.noise, debug=False, len=5000)
    else:
        pt = _PT_PATHS[args.dataset]
        dataset = EncodingsDataset(pt_path=pt, split="train")
        val_dataset = EncodingsDataset(pt_path=pt, split="test", debug=False)
        args.nd = dataset.nd
    if args.pnd is None:
        args.pnd = args.nd
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"device={device}  op={args.op}  dataset={args.dataset}  nd={args.nd}  pnd={args.pnd}")

    run_name = f"{args.op}_{args.order}" if args.op in ("ph", "quat") else args.op
    if args.op in ("filmr_expm", "filmr") and args.rank != 2:
        run_name = f"{args.op}_{args.rank}"
    run_name = f"{args.dataset}_{run_name}"
    run_name = f"{run_name}_{args.tag}" if args.tag else run_name
    project = {"line": "ring", "mnist": "ring-mnist", "cifar": "ring-cifar"}[args.dataset]
    if not args.no_wandb:
        wandb.init(project=project, name=run_name, config=vars(args))
        # index every logged metric/media by epoch, so panel sliders (incl. images) read in epochs, not steps
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    proj = Projector(nd=args.nd, pnd=args.pnd, n_hid=args.n_hid, n_layers=args.proj_layers, proj_resid=args.proj_resid, unit_norm=args.unit_norm).to(device)
    trans_op = OpWrapper(args.op, args.pnd, args.order, args.op_resid, args.rank, args.unit_norm).to(device)
    if args.op == "quat":
        freeze_quaternion(trans_op.op)
    # inverse projector: maps pnd back to nd (unit_norm=False: output isn't on the sphere)
    inv_proj = Projector(nd=args.pnd, pnd=args.nd, n_hid=args.n_hid, n_layers=args.proj_layers, unit_norm=False).to(device)

    n_proj, n_op, n_inv = (sum(p.numel() for p in m.parameters() if p.requires_grad) for m in [proj, trans_op, inv_proj])
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
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.lr / 20)
    warmup = (torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.warmup_start_lr / args.lr, total_iters=args.warmup)
        if args.warmup > 0 else None)  # constructing this immediately scales optimizer LRs down to the start
    scheduler = WarmupThenPlateauWithReduction(warmup, plateau, args.warmup)
    sim_fn = nn.MSELoss()
    sim_ema = None  # smoothed sim for the scheduler (sigreg flattens, so don't anneal on total)

    # load VAE + cache test grid once for periodic inference viz
    vae, viz_imgs = None, None
    if args.dataset in ("mnist", "cifar") and args.inf_every > 0 and wandb.run is not None:
        vae = load_vae(args.dataset, device=str(device))
        viz_imgs = make_class_ordered_images(dataset=args.dataset).to(device)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{run_name}.pt")
    best_val = float("inf")  # tracks best val_sim (sigreg is ~constant, sim is the real quality signal)

    try:
        for epoch in range(1, args.epochs + 1):
            totals = dict(loss=0.0, sim=0.0, mom=0.0, sigreg=0.0, recon=0.0)
            pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
            for x, y in pbar:
                xb, yb = x['data'].to(device), y['data'].to(device)
                optimizer.zero_grad()
                xproj, yproj = proj(xb), proj(yb)  # project into new space
                xproj_t = trans_op(xproj)          # transform/rotate
                xprime, yprime = inv_proj(xproj), inv_proj(yproj)  # reconstruct both from embedding
                sim_loss = sim_fn(xproj_t, yproj) # weak per-sample anchor: sets *where* the cloud goes
                mom_loss = MomMatchLoss(xproj_t, yproj, labels=x['label'].to(device),
                                        diag=args.mom_diag, cov_weight=args.mom_cov_weight)  # match cloud *shape*
                sigreg_loss = SIGReg( torch.cat([xproj_t, yproj], dim=0), global_step=epoch )  # pull distribution toward Gaussian
                recon_loss = sim_fn(torch.cat([xprime, yprime]), torch.cat([xb, yb]))  # inv_proj sees both x and y
                loss = (1 - args.lambd) * (args.lambda_sim * sim_loss + args.lambda_mom * mom_loss) + args.lambd * sigreg_loss + args.lambda_recon * recon_loss
                loss.backward()
                optimizer.step()
                bs = xb.size(0)
                for k, v in zip(totals, [loss, sim_loss, mom_loss, sigreg_loss, recon_loss]):
                    totals[k] += v.item() * bs
                pbar.set_postfix(loss=f"{loss.item():.6f}")
            avg = {k: v / len(dataset) for k, v in totals.items()}
            sim_ema = avg["sim"] if sim_ema is None else args.sim_ema * sim_ema + (1 - args.sim_ema) * avg["sim"]
            scheduler.step(sim_ema, epoch)  # warmup ramp, then anneal on smoothed sim (sigreg flattens by design)
            op_angle = trans_op.op.rotation_angle_deg() if args.op == "filmr_expm" else None
            angle_str = f"  angle={op_angle:.2f}deg" if op_angle is not None else ""
            print(f"epoch {epoch:4d}/{args.epochs}  loss={avg['loss']:.6f}  sim={avg['sim']:.6f}  mom={avg['mom']:.6f}  sigreg={avg['sigreg']:.6f}  recon={avg['recon']:.6f}{angle_str}")

            do_val = args.val_every and (epoch % args.val_every == 0 or epoch == 1)
            if do_val:
                (val_loss, val_sim, val_mom, val_sigreg, val_recon), val_vars, viz = evaluate(
                    val_loader, proj, trans_op, inv_proj, sim_fn, device, epoch, args,
                    max_viz=args.max_viz_points)
                print(f"      val  loss={val_loss:.6f}  sim={val_sim:.6f}  mom={val_mom:.6f}  sigreg={val_sigreg:.6f}  recon={val_recon:.6f}  var(xt/y)={val_vars['var_xproj_t']:.4f}/{val_vars['var_yproj']:.4f}")
                if val_sim < best_val and epoch >= args.min_ckpt_epoch:
                    best_val = val_sim
                    torch.save({"epoch": epoch, "val_sim_loss": val_sim, "args": vars(args),
                                "proj": proj.state_dict(), "inv_proj": inv_proj.state_dict(),
                                "trans_op": trans_op.state_dict()}, ckpt_path)

            if wandb.run is not None:
                log = {"epoch": epoch, "loss": avg["loss"], "lr": optimizer.param_groups[0]["lr"],
                       "sim_loss": avg["sim"], "sim_ema": sim_ema, "mom_loss": avg["mom"],
                       "sigreg_loss": avg["sigreg"], "recon_loss": avg["recon"]}
                if op_angle is not None:
                    log["op_angle_deg"] = op_angle
                if do_val:
                    log.update({"val_loss": val_loss, "val_sim_loss": val_sim,
                                "val_mom_loss": val_mom, "val_sigreg_loss": val_sigreg,
                                "val_recon_loss": val_recon, **val_vars})
                    if viz is not None:
                        vy, vxt, vyl, vxl = viz
                        log["embedding"] = embedding_scatter3d(
                            vy, vxt, epoch, args.op, args.order,
                            yproj_labels=vyl, xproj_t_labels=vxl,
                            max_points=args.max_viz_points)
                if vae is not None and args.inf_every > 0 and epoch % args.inf_every == 0:
                    for m in (proj, trans_op, inv_proj): m.eval()
                    imgs_in, imgs_recon, imgs_xform = make_viz_grids(vae, proj, trans_op, inv_proj, viz_imgs)
                    for m in (proj, trans_op, inv_proj): m.train()
                    def _wimg(t): return wandb.Image(make_grid(t, nrow=10).permute(1,2,0).numpy(), caption=f"epoch {epoch}")
                    log.update({f"{args.dataset}_input": _wimg(imgs_in), f"{args.dataset}_recon": _wimg(imgs_recon), f"{args.dataset}_transformed": _wimg(imgs_xform)})
                wandb.log(log)
    except KeyboardInterrupt:
        print("\ninterrupted — finishing run")
    finally:
        if wandb.run is not None:
            wandb.finish()


def main():
    p = configargparse.ArgumentParser(description="Ring task with different model variants.")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--config", is_config_file=True, help="path to a config file (keys = dest names with underscores)")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--dataset", choices=["line", "mnist", "cifar"], default="line",
                   help="line=synthetic ring; mnist/cifar=VAE encodings (nd forced by dataset)")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--min-ckpt-epoch", type=int, default=20,
                   help="Don't save checkpoints before this epoch (avoids locking in spuriously low early loss)")
    p.add_argument("--inf-every", type=int, default=20,
                   help="Log MNIST inference grids to W&B every N epochs (0 disables; mnist only)")
    p.add_argument("--lambd", type=float, default=0.01,
                   help="SIGReg weight: loss = (1-lambd)*sim + lambd*sigreg")
    p.add_argument("--lambda-recon", type=float, default=1.0,
                   help="Weight on inv_proj autoencoder reconstruction loss (0 disables)")
    p.add_argument("--lambda-mom", type=float, default=0.5,
                   help="Weight on MomMatch inside the non-sigreg group (0 disables; high values can merge classes)")
    p.add_argument("--lambda-sim", type=float, default=0.5,
                   help="Weight on MSE sim inside the non-sigreg group: (1-lambd)*(lambda_sim*sim + lambda_mom*mom)")
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--lr-patience", type=int, default=50, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--max-viz-points", type=int, default=1000,
                   help="Max points to accumulate per epoch for the W&B embedding scatter")
    p.add_argument("--mom-cov-weight", type=float, default=1.0,
                   help="Weight on the covariance term inside MomMatchLoss (mean term fixed at 1)")
    p.add_argument("--mom-diag", action="store_true",
                   help="Match per-dim variances only in MomMatchLoss (robust when pnd >> samples/class, e.g. cifar)")
    p.add_argument("--n-hid", type=int, default=32, help="Projector hidden dim")
    p.add_argument("--nd", type=int, default=3, help="Dimension of the space")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--noise", type=float, default=0.01, help="Jitter added to each point")
    p.add_argument("--npoints", type=int, default=12, help="Number of quantized points on the line")
    p.add_argument("--op", choices=["filmr", "filmr_expm", "matop", "matop_clip", "matop2", "ph", "quat", "kquat"], default="filmr_expm")
    p.add_argument("--op-lr", type=float, default=None,
                   help="Separate LR for trans_op (default: same as --lr). Lower it to slow the angle's climb.")
    p.add_argument("--op-resid", action=argparse.BooleanOptionalAction, default=True,
                   help="Wrap trans_op as x + op(x) (centers transform at identity; good for many ring points)")
    p.add_argument("--order", type=int, default=4,
                   help="Hypercomplex order n for PHMLinear (nd must be divisible by order)")
    p.add_argument("--pnd", type=int, default=None,
                   help="Projected space dimension (default: same as --nd)")
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
    p.add_argument("--val-every", type=int, default=4,
                   help="Evaluate on held-out split every N epochs (0 disables)")
    p.add_argument("--warmup", type=int, default=50,
                   help="Linear LR warmup epochs before ReduceLROnPlateau takes over (0 disables)")
    p.add_argument("--warmup-start-lr", type=float, default=1e-5,
                   help="LR at epoch 1 of warmup; ramps to --lr (and --op-lr) by epoch --warmup")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay on >=2D weights (helps FiLMR_expm precision drift)")
    main_args = p.parse_args()
    print("Arguments:", vars(main_args))
    train(main_args)


if __name__ == "__main__":
    main()
