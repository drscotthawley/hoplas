#!/usr/bin/env python3
"""Ring-task training via four model variants. (dataset/models WIP)"""

import argparse
import ast
import configargparse
import json
import os
import sys
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
from hoplas.viz import embedding_scatter3d, fit_pca, SECONDARY_SCALES


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
def evaluate(loader, proj, trans_op, inv_proj, sim_fn, device, epoch, args, max_viz=0,
             sec_heads=(), dataset=None):
    """Run the same losses over a held-out loader (no grad) for generalization metrics.
    If max_viz > 0, also accumulates up to max_viz projected points for visualization
    (primary op, plus each secondary head's output/target for its own scatter)."""
    proj.eval(); trans_op.eval(); inv_proj.eval()
    n = tot_loss = tot_sim = tot_mom = tot_sigreg = tot_recon = tot_va = tot_vb = 0.0
    viz_y, viz_xt, viz_yl, viz_xl, viz_n = [], [], [], [], 0
    sec_y = [[] for _ in sec_heads]; sec_xt = [[] for _ in sec_heads]
    for x, y in loader:
        xb, yb = x['data'].to(device), y['data'].to(device)
        xproj, yproj = proj(xb), proj(yb)
        xproj_t = trans_op(xproj)
        xprime, yprime = inv_proj(xproj), inv_proj(yproj)
        sim = sim_fn(xproj_t, yproj)
        mom, stats = MomMatchLoss(xproj_t, yproj, labels=x['label'].to(device),
                                  diag=args.mom_diag, cov_weight=args.mom_cov_weight, return_stats=True)
        sigreg = 0.5 * (SIGReg(xproj_t, global_step=epoch) + SIGReg(yproj, global_step=epoch))
        recon = sim_fn(torch.cat([xprime, yprime]), torch.cat([xb, yb]))
        loss = (1 - args.lambd) * (args.lambda_sim * sim + args.lambda_mom * mom) + args.lambd * sigreg + args.lambda_recon * recon
        bs = xb.size(0)
        tot_loss += loss.item() * bs; tot_sim += sim.item() * bs; tot_mom += mom.item() * bs
        tot_sigreg += sigreg.item() * bs; tot_recon += recon.item() * bs
        tot_va += stats["var_a"] * bs; tot_vb += stats["var_b"] * bs; n += bs
        if max_viz > 0 and viz_n < max_viz:
            viz_y.append(yproj); viz_xt.append(xproj_t)
            viz_yl.append(y['label']); viz_xl.append(x['label'])
            for hi, h in enumerate(sec_heads):
                sec_xt[hi].append(h["op"](xproj))
                sec_y[hi].append(proj(dataset.sample_target(x['label'].to(device), h["target"])))
            viz_n += bs
    proj.train(); trans_op.train(); inv_proj.train()
    losses = (tot_loss / n, tot_sim / n, tot_mom / n, tot_sigreg / n, tot_recon / n)
    var_stats = {"var_xproj_t": tot_va / n, "var_yproj": tot_vb / n}
    viz = (torch.cat(viz_y), torch.cat(viz_xt), torch.cat(viz_yl), torch.cat(viz_xl)) if viz_y else None
    # per secondary head: (y2proj, xproj_t2); labels reuse the source labels (viz_xl)
    sec_viz = [(torch.cat(sy), torch.cat(sx)) for sy, sx in zip(sec_y, sec_xt)] if viz_y else []
    return losses, var_stats, viz, sec_viz


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if not args.cpu else torch.device("cpu")
    _PT_PATHS = {"mnist": "~/datasets/mnist_latents.pt", "cifar": "~/datasets/cifar_latents.pt"}
    if args.dataset == "line":
        dataset = LineDataset(nd=args.nd, npoints=args.npoints, noise=args.noise, target=args.target)
        val_dataset = LineDataset(nd=args.nd, npoints=args.npoints, noise=args.noise, debug=False, len=5000, target=args.target)
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

    # op identifier: op[_order|_rank] + any secondary heads + tag
    op_part = f"{args.op}_{args.order}" if args.op in ("ph", "quat") else args.op
    if args.op in ("filmr_expm", "filmr") and args.rank != 2:
        op_part = f"{args.op}_{args.rank}"
    if args.op_list and len(args.op_list) > 1:
        op_part += "".join(f"+{s['op']}_{s['target']}" for s in args.op_list[1:])
    if args.tag:
        op_part = f"{op_part}_{args.tag}"
    if args.dataset == "line":
        # nd varies for line (3 vs 16, ...) so disambiguate. The project already conveys dataset+kind,
        # so omit that prefix from the wandb run name; keep it in the filename. Multi-head run = dihedral.
        op_part = f"nd{args.nd}_{op_part}"
        line_kind = "dihedral" if (args.op_list and len(args.op_list) > 1) else args.target
        wandb_name = op_part
        ckpt_name = f"line_{line_kind}_{op_part}"
        project = f"line-{line_kind}"
    else:
        wandb_name = ckpt_name = f"{args.dataset}_{op_part}"
        project = {"mnist": "ring-mnist", "cifar": "ring-cifar"}[args.dataset]
    if not args.no_wandb:
        wandb.init(project=project, name=wandb_name, config=vars(args))
        # index every logged metric/media by epoch, so panel sliders (incl. images) read in epochs, not steps
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    proj = Projector(nd=args.nd, pnd=args.pnd, n_hid=args.n_hid, n_layers=args.proj_layers, proj_resid=args.proj_resid, unit_norm=args.unit_norm).to(device)
    trans_op = OpWrapper(args.op, args.pnd, args.order, args.op_resid, args.rank, args.unit_norm).to(device)
    if args.op == "quat":
        freeze_quaternion(trans_op.op)
    # inverse projector: maps pnd back to nd (unit_norm=False: output isn't on the sphere)
    inv_proj = Projector(nd=args.pnd, pnd=args.nd, n_hid=args.n_hid, n_layers=args.proj_layers, unit_norm=False).to(device)

    # Frozen-geometry mode: load proj/inv_proj weights and freeze them, so only the op trains against
    # the fixed embedding (pure supervised). recon becomes constant. Geometry args were inherited as
    # defaults in main() (config still overrides), so the projector dims here match the checkpoint.
    if args.freeze_proj_from:
        ck = torch.load(os.path.expanduser(args.freeze_proj_from), map_location=device)
        proj.load_state_dict(ck["proj"]); inv_proj.load_state_dict(ck["inv_proj"])
        for m in (proj, inv_proj):
            for p in m.parameters():
                p.requires_grad_(False)
        print(f"froze proj+inv_proj from {args.freeze_proj_from} (epoch {ck.get('epoch')}); training op only")

    # Secondary op-heads (from --op-list entries 1+): each trains detached on its own target, riding
    # on the geometry the primary (trans_op) shapes, without back-propagating into proj/inv_proj.
    sec_heads = []
    for spec in (args.op_list[1:] if args.op_list else []):
        op = OpWrapper(spec["op"], args.pnd, spec.get("order", args.order),
                       spec.get("op_resid", args.op_resid), spec.get("rank", args.rank),
                       args.unit_norm).to(device)
        sec_heads.append({"op": op, "target": spec["target"],
                          "detach": spec.get("detach", True),
                          "name": f"{spec['op']}_{spec['target']}"})
    if sec_heads:
        print(f"secondary heads: {[h['name'] + ('(detach)' if h['detach'] else '') for h in sec_heads]}")

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
    op_params = list(trans_op.parameters()) + [p for h in sec_heads for p in h["op"].parameters()]
    split = lambda ps: ([p for p in ps if p.ndim >= 2], [p for p in ps if p.ndim < 2])
    other_decay, other_nodecay = split(other_params)
    op_decay, op_nodecay = split(op_params)
    # Two separate optimizers (GAN-style): the op(s) step --op-steps times per projector step, so the
    # transform can move faster than the geometry. opt_proj = proj+inv_proj; opt_op = trans_op+sec_heads.
    opt_proj = torch.optim.AdamW([
        {"params": other_decay,  "weight_decay": args.weight_decay},
        {"params": other_nodecay, "weight_decay": 0.0},
    ], lr=args.lr)
    opt_op = torch.optim.AdamW([
        {"params": op_decay,   "weight_decay": args.weight_decay},
        {"params": op_nodecay, "weight_decay": 0.0},
    ], lr=op_lr)

    def make_sched(opt, base_lr):
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=args.lr_patience, min_lr=base_lr / 20)
        warmup = (torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=args.warmup_start_lr / base_lr, total_iters=args.warmup)
            if args.warmup > 0 else None)
        return WarmupThenPlateauWithReduction(warmup, plateau, args.warmup)
    sched_proj = make_sched(opt_proj, args.lr)
    sched_op = make_sched(opt_op, op_lr)
    sim_fn = nn.MSELoss()
    sim_ema = None  # smoothed sim for the scheduler (sigreg flattens, so don't anneal on total)

    def compute_loss(xb, yb, labels, epoch):
        """Full forward + loss (primary terms + detached secondary heads). Returns (loss, comp-dict)."""
        xproj, yproj = proj(xb), proj(yb)                  # project into new space
        xproj_t = trans_op(xproj)                          # transform/rotate
        xprime, yprime = inv_proj(xproj), inv_proj(yproj)  # reconstruct both from embedding
        sim_loss = sim_fn(xproj_t, yproj)                  # weak per-sample anchor: where the cloud goes
        mom_loss = MomMatchLoss(xproj_t, yproj, labels=labels,
                                diag=args.mom_diag, cov_weight=args.mom_cov_weight)  # match cloud shape
        # SIGReg each cloud independently (not on the cat): the joint test lets a collapsed
        # xproj_t hide inside yproj's spread, halving the anti-collapse pressure
        sigreg_loss = 0.5 * (SIGReg(xproj_t, global_step=epoch) + SIGReg(yproj, global_step=epoch))
        recon_loss = sim_fn(torch.cat([xprime, yprime]), torch.cat([xb, yb]))  # inv_proj sees both x and y
        loss = (1 - args.lambd) * (args.lambda_sim * sim_loss + args.lambda_mom * mom_loss) + args.lambd * sigreg_loss + args.lambda_recon * recon_loss
        # secondary heads: supervised sim on their own target; detached so they don't reshape the geometry
        sec_sim = 0.0
        for h in sec_heads:
            src = xproj.detach() if h["detach"] else xproj
            h_tgt = proj(dataset.sample_target(labels, h["target"])).detach()
            sec_sim = sec_sim + sim_fn(h["op"](src), h_tgt)
        if sec_heads:
            loss = loss + (1 - args.lambd) * args.lambda_sim * sec_sim
        comp = {"loss": float(loss), "sim": float(sim_loss), "mom": float(mom_loss),
                "sigreg": float(sigreg_loss), "recon": float(recon_loss),
                "sec_sim": float(sec_sim) if sec_heads else 0.0}
        return loss, comp

    # load VAE + cache test grid once for periodic inference viz
    vae, viz_imgs = None, None
    if args.dataset in ("mnist", "cifar") and args.inf_every > 0 and wandb.run is not None:
        vae = load_vae(args.dataset, device=str(device))
        viz_imgs = make_class_ordered_images(dataset=args.dataset).to(device)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{ckpt_name}.pt")
    best_val = float("inf")  # tracks best val_sim (sigreg is ~constant, sim is the real quality signal)

    try:
        for epoch in range(1, args.epochs + 1):
            totals = dict(loss=0.0, sim=0.0, mom=0.0, sigreg=0.0, recon=0.0, sec_sim=0.0)
            pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
            for x, y in pbar:
                xb, yb = x['data'].to(device), y['data'].to(device)
                labels = x['label'].to(device)
                # op phase: --op-steps backprop steps that move only the transform(s) (fresh forward each)
                for _ in range(args.op_steps):
                    opt_proj.zero_grad(); opt_op.zero_grad()
                    loss, _ = compute_loss(xb, yb, labels, epoch)
                    loss.backward()
                    opt_op.step()
                # projector phase: one step moving only proj+inv_proj
                opt_proj.zero_grad(); opt_op.zero_grad()
                loss, comp = compute_loss(xb, yb, labels, epoch)
                loss.backward()
                opt_proj.step()
                bs = xb.size(0)
                for k in totals:
                    totals[k] += comp[k] * bs
                pbar.set_postfix(loss=f"{comp['loss']:.6f}")
            avg = {k: v / len(dataset) for k, v in totals.items()}
            sim_ema = avg["sim"] if sim_ema is None else args.sim_ema * sim_ema + (1 - args.sim_ema) * avg["sim"]
            sched_proj.step(sim_ema, epoch); sched_op.step(sim_ema, epoch)  # warmup ramp, then anneal on smoothed sim
            op_angle = trans_op.op.rotation_angle_deg() if args.op == "filmr_expm" else None
            angle_str = f"  angle={op_angle:.2f}deg" if op_angle is not None else ""
            print(f"epoch {epoch:4d}/{args.epochs}  loss={avg['loss']:.6f}  sim={avg['sim']:.6f}  mom={avg['mom']:.6f}  sigreg={avg['sigreg']:.6f}  recon={avg['recon']:.6f}{angle_str}")

            do_val = args.val_every and (epoch % args.val_every == 0 or epoch == 1)
            if do_val:
                (val_loss, val_sim, val_mom, val_sigreg, val_recon), val_vars, viz, sec_viz = evaluate(
                    val_loader, proj, trans_op, inv_proj, sim_fn, device, epoch, args,
                    max_viz=args.max_viz_points, sec_heads=sec_heads, dataset=val_dataset)
                print(f"      val  loss={val_loss:.6f}  sim={val_sim:.6f}  mom={val_mom:.6f}  sigreg={val_sigreg:.6f}  recon={val_recon:.6f}  var(xt/y)={val_vars['var_xproj_t']:.4f}/{val_vars['var_yproj']:.4f}")
                if val_sim < best_val and epoch >= args.warmup:
                    best_val = val_sim
                    torch.save({"epoch": epoch, "val_sim_loss": val_sim, "args": vars(args),
                                "proj": proj.state_dict(), "inv_proj": inv_proj.state_dict(),
                                "trans_op": trans_op.state_dict(),
                                "sec_ops": {h["name"]: h["op"].state_dict() for h in sec_heads}}, ckpt_path)

            if wandb.run is not None:
                log = {"epoch": epoch, "loss": avg["loss"], "lr": opt_proj.param_groups[0]["lr"],
                       "op_lr": opt_op.param_groups[0]["lr"],
                       "sim_loss": avg["sim"], "sim_ema": sim_ema, "mom_loss": avg["mom"],
                       "sigreg_loss": avg["sigreg"], "recon_loss": avg["recon"]}
                if sec_heads:
                    log["sec_sim_loss"] = avg["sec_sim"]
                if op_angle is not None:
                    log["op_angle_deg"] = op_angle
                if do_val:
                    log.update({"val_loss": val_loss, "val_sim_loss": val_sim,
                                "val_mom_loss": val_mom, "val_sigreg_loss": val_sigreg,
                                "val_recon_loss": val_recon, **val_vars})
                    if viz is not None:
                        vy, vxt, vyl, vxl = viz
                        # one shared PCA basis over all series so primary + secondary panels line up
                        pca = fit_pca([vy, vxt] + [a for sv in sec_viz for a in sv])
                        log["embedding"] = embedding_scatter3d(
                            vy, vxt, epoch, args.op, args.order,
                            s0_labels=vyl, s1_labels=vxl, pca=pca, max_points=args.max_viz_points)
                        for h, (s2y, s2xt) in zip(sec_heads, sec_viz):
                            # color both series by the source label i (same color = should overlap)
                            log[f"embedding_{h['name']}"] = embedding_scatter3d(
                                s2y, s2xt, epoch, h["name"], None,
                                s0_labels=vxl, s1_labels=vxl,
                                names=("y2proj", "xproj_t2"), scales=SECONDARY_SCALES,
                                pca=pca, max_points=args.max_viz_points)
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
    p.add_argument("--freeze-proj-from", type=str, default=None,
                   help="Load proj+inv_proj from this checkpoint and freeze them; train only the op "
                        "on the fixed geometry (supervised). Use with --target reflect, lambd=0, lambda_mom=0.")
    p.add_argument("--inf-every", type=int, default=20,
                   help="Log MNIST inference grids to W&B every N epochs (0 disables; mnist only)")
    p.add_argument("--lambd", type=float, default=0.01,
                   help="SIGReg weight: loss = (1-lambd)*sim + lambd*sigreg")
    p.add_argument("--lambda-mom", type=float, default=0.5,
                   help="Weight on MomMatch inside the non-sigreg group (0 disables; high values can merge classes)")
    p.add_argument("--lambda-recon", type=float, default=1.0,
                   help="Weight on inv_proj autoencoder reconstruction loss (0 disables)")
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
    p.add_argument("--op", choices=["filmr", "filmr_expm", "matop", "matop_clip", "matop2", "ph", "quat", "kquat", "kdualquat"], default="filmr_expm")
    p.add_argument("--op-list", nargs="+", default=None,
                   help='Op-head list (overrides --op), inline JSON. Entry 0 = primary head (shapes the '
                        'projector, uses --target); later entries = secondary heads on their own target '
                        '(detached by default). e.g. '
                        '[{"op":"ph","order":4,"target":"ring"}, {"op":"matop","target":"reflect","detach":true}]')
    p.add_argument("--op-lr", type=float, default=None,
                   help="Separate LR for trans_op (default: same as --lr). Lower it to slow the angle's climb.")
    p.add_argument("--op-resid", action=argparse.BooleanOptionalAction, default=True,
                   help="Wrap trans_op as x + op(x) (centers transform at identity; good for many ring points)")
    p.add_argument("--op-steps", type=int, default=2,
                   help="Op-optimizer backprop steps per projector step (GAN-style; 1 = alternate 1:1)")
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
    p.add_argument("--target", choices=["ring", "reflect"], default="ring",
                   help="LineDataset target: ring (cyclic T) or reflect (dihedral inversion I)")
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
    # Frozen-geometry mode: seed the projector-architecture args from the checkpoint as new *defaults*,
    # then re-parse so config/CLI still win for any key you do state. Omit these from the config and
    # they fall back to the checkpoint (so the frozen projector dims/behavior match the saved weights).
    if main_args.freeze_proj_from:
        ck_args = torch.load(os.path.expanduser(main_args.freeze_proj_from), map_location="cpu")["args"]
        p.set_defaults(**{k: ck_args[k] for k in ("nd", "pnd", "n_hid", "proj_layers", "unit_norm")
                          if k in ck_args})
        main_args = p.parse_args()
    # --op-list (nargs='+'): elements are dicts, JSON strings (CLI tokens), or Python-repr strings
    # (configargparse YAML-parses the config value to dicts, then str()'s each to feed the nargs arg).
    # Normalize to a flat list of head dicts; entry 0 folds into the single-op args below.
    if main_args.op_list:
        heads = []
        try:
            for it in main_args.op_list:
                if isinstance(it, dict):
                    heads.append(it)
                    continue
                try:
                    parsed = json.loads(it)          # CLI: JSON (double quotes, lowercase true)
                except (json.JSONDecodeError, ValueError):
                    parsed = ast.literal_eval(it)     # config: Python repr (single quotes, True)
                heads.extend(parsed) if isinstance(parsed, list) else heads.append(parsed)
        except (json.JSONDecodeError, ValueError, TypeError, SyntaxError) as e:
            sys.exit(f"--op-list: could not parse {main_args.op_list!r}: {e}")
        if not heads or not all(isinstance(h, dict) for h in heads):
            sys.exit(f"--op-list must be a non-empty list of op dicts; got: {heads!r}")
        main_args.op_list = heads
        p0 = main_args.op_list[0]
        main_args.op = p0["op"]
        main_args.order = p0.get("order", main_args.order)
        main_args.rank = p0.get("rank", main_args.rank)
        main_args.op_resid = p0.get("op_resid", main_args.op_resid)
        main_args.target = p0.get("target", main_args.target)
    print("Arguments:", vars(main_args))
    train(main_args)


if __name__ == "__main__":
    main()
