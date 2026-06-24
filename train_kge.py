#!/usr/bin/env python3
"""Knowledge-graph embedding as the ring task: relations are learnable group actions.

Mapping from train_ops.py (ring task) to KGE:
  ring:  y = trans_op(x);   loss = sim(op(x), y) + SIGReg(y)        # ONE global op
  KGE:   t = op_r(h);       loss = sim(op_r(E[h]), E[t]) + SIGReg(E)  # ONE op PER relation

Entities are a learnable embedding table (indexed by ID -- no raw coordinates to
project, unlike LineDataset). Each relation gets its own OpWrapper, so swapping
--op {ph,quat,kquat} is a direct PHM-vs-quaternion benchmark of the relation operator.
SIGReg on the entity cloud replaces negative sampling as the anti-collapse force.

Eval is our own filtered MRR/Hits@k (no PyKEEN evaluator). With inverse triples in the
eval split, ranking tails covers both head- and tail-prediction (the "both" metric).
"""
import argparse
import math
import os
import time

import configargparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from hoplas.data import KGTripleDataset
from hoplas.ops import OpWrapper
from hoplas.losses import SIGReg, MomMatchLoss
from hoplas.viz import fit_pca, embedding_scatter3d


def _hamilton_table(like):
    """Quaternion Hamilton multiplication table (4,4,4), matching `like`'s dtype/device."""
    return torch.tensor([
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]],
        [[0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
    ], dtype=like.dtype, device=like.device)


def freeze_quaternion(ph_layer):
    """Fix ph_layer.a to the Hamilton quaternion table and freeze it (-> raw quaternion)."""
    assert ph_layer.n == 4, "freeze_quaternion requires --order 4"
    ph_layer.a.data.copy_(_hamilton_table(ph_layer.a))
    ph_layer.a.requires_grad_(False)


def init_quaternion(ph_layer):
    """Warm-start ph_layer.a at the Hamilton table but keep it *trainable*, so the algebra
    starts as an exact quaternion and is free to deviate as it learns."""
    assert ph_layer.n == 4, "init_quaternion requires --order 4"
    ph_layer.a.data.copy_(_hamilton_table(ph_layer.a))  # requires_grad stays True


class KGEModel(nn.Module):
    """Learnable entity embeddings + one relation operator per relation id."""

    def __init__(self, num_entities, num_relations, nd, op, order,
                 op_resid=True, rank=2, unit_norm=False, quat_init=False):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, nd)
        nn.init.normal_(self.entity_emb.weight, std=1.0)  # ~N(0,1); SIGReg keeps Cov ~ I
        self.ops = nn.ModuleList([
            OpWrapper(op, nd, order, op_resid, rank, unit_norm) for _ in range(num_relations)
        ])
        if op == "quat":
            for o in self.ops:
                freeze_quaternion(o.op)          # frozen exact quaternion (raw baseline)
        elif op == "ph" and quat_init:
            for o in self.ops:
                init_quaternion(o.op)            # learnable, warm-started at exact quaternion

    def _apply_relation_loop(self, h_emb, r):
        """Reference: loop over the (<=Nr) relations present, apply each op to its rows.
        Correct for any op type, but O(#distinct relations) Python iterations per batch
        — the bottleneck on many-relation datasets (FB15k-237: 474, FB15k: 2690)."""
        out = h_emb.new_empty(h_emb.shape)
        for rid in r.unique().tolist():
            m = r == rid
            out[m] = self.ops[rid](h_emb[m])
        return out

    def _phm_stack_ok(self):
        """True iff every relation op is a PHMLinear (ph/quat): has a, s, bias, n."""
        return all(hasattr(o.op, "a") and hasattr(o.op, "s") and hasattr(o.op, "bias")
                   and hasattr(o.op, "n") for o in self.ops)

    def _apply_relation_vec(self, h_emb, r, chunk=2048):
        """Vectorized equivalent of the loop for PHMLinear ops, via the implicit-einsum
        math (PHMLinear_Implicit: no per-relation weight materialization). Stacks the
        per-relation algebra a (Nr,n,n,n), block weights s (Nr,n,do,di) and bias (Nr,nd),
        then for each BATCH CHUNK gathers per-sample by relation id and contracts in a
        memory-frugal order. Chunking + the explicit two-step contraction keep memory
        bounded (the naive single fused einsum blew up to ~18GB at nd512/bs8192).
        Handles op_resid / unit_norm uniformly from the (identical) OpWrappers."""
        op0 = self.ops[0]
        n = op0.op.n
        A = torch.stack([o.op.a for o in self.ops])      # (Nr, n, n, n)  [i, a, b]
        S = torch.stack([o.op.s for o in self.ops])      # (Nr, n, do, di) [i, j, k]
        Bk = torch.stack([o.op.bias for o in self.ops])  # (Nr, nd)
        B = h_emb.shape[0]
        out = h_emb.new_empty(B, h_emb.shape[1])
        for lo in range(0, B, chunk):
            sl = slice(lo, lo + chunk)
            hc, rc = h_emb[sl], r[sl]
            a_r, s_r, b_r = A[rc], S[rc], Bk[rc]          # gather per sample (this chunk)
            X = hc.reshape(hc.shape[0], n, -1)            # (p, b, k) = (p, n, di)
            T = torch.einsum("pijk,pbk->pijb", s_r, X)    # contract k -> (p, i, j, b) small
            Y = torch.einsum("piab,pijb->paj", a_r, T)    # contract i,b -> (p, a, j)=(p,n,do)
            opx = Y.reshape(hc.shape[0], -1) + b_r        # == PHMLinear(hc), per row
            oc = hc + opx if op0.op_resid else opx
            out[sl] = F.normalize(oc, dim=-1) if op0.unit_norm else oc
        return out

    def apply_relation(self, h_emb, r):
        """Apply each sample's relation operator. r: (B,) relation ids.
        apply_mode (set by --apply): 'loop' (default), 'vec' (fast, PHM only), or
        'check' (run both and assert they match — for verifying the vectorization)."""
        mode = getattr(self, "apply_mode", "loop")
        if mode in ("vec", "check") and self._phm_stack_ok():
            vec = self._apply_relation_vec(h_emb, r)
            if mode == "vec":
                return vec
            ref = self._apply_relation_loop(h_emb, r)
            d = (ref - vec).abs().max().item()
            print(f"[apply check] max|loop-vec|={d:.3e}", flush=True)
            assert d < 1e-3, f"vectorized apply_relation mismatch: {d}"
            return ref
        return self._apply_relation_loop(h_emb, r)

    def forward(self, h, r, t):
        return self.apply_relation(self.entity_emb(h), r), self.entity_emb(t)


@torch.no_grad()
def evaluate(model, eval_ds, hr2t, device, batch=512, score="l2"):
    """Filtered MRR / Hits@k over eval_ds (which includes inverse triples).

    score: ranking function over candidate tails.
      l2  -> -||pred - E||^2 (default; the original distance score)
      dot -> pred . E         (bilinear; drops the spurious -||E||^2 per-entity term)
      cos -> cosine(pred, E)  (dot, with pred and E row-normalized)
    """
    model.eval()
    E = model.entity_emb.weight                      # (Ne, nd)
    E_sq = (E ** 2).sum(-1)                           # (Ne,)
    E_norm = E / (E.norm(dim=-1, keepdim=True) + 1e-9)
    triples = eval_ds.triples
    ranks = []
    for i in range(0, len(triples), batch):
        chunk = triples[i:i + batch].to(device)
        h, r, t = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        pred = model.apply_relation(model.entity_emb(h), r)         # (B, nd)
        if score == "dot":
            scores = pred @ E.t()
        elif score == "cos":
            pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-9)
            scores = pred_n @ E_norm.t()
        else:  # "l2": -||pred - E||^2 via the matmul expansion (no (B,Ne,nd) tensor)
            scores = -((pred ** 2).sum(-1, keepdim=True) - 2 * pred @ E.t() + E_sq.unsqueeze(0))
        for b in range(chunk.size(0)):
            tgt = t[b].item()
            others = [x for x in hr2t[(h[b].item(), r[b].item())] if x != tgt]
            s = scores[b]
            if others:
                s[torch.tensor(others, device=device)] = float("-inf")  # filtered setting
            ranks.append(1 + int((s > s[tgt]).sum().item()))           # optimistic-free: strict >
    model.train()
    ranks = torch.tensor(ranks, dtype=torch.float)
    return dict(mrr=(1.0 / ranks).mean().item(), mr=ranks.mean().item(),
                h1=(ranks <= 1).float().mean().item(),
                h3=(ranks <= 3).float().mean().item(),
                h10=(ranks <= 10).float().mean().item())


def algebra_tensors(model):
    """Stacked per-relation algebra tensors (Nr, n, n, n), or None if the op has no `a`."""
    if not all(hasattr(o.op, "a") for o in model.ops):
        return None
    return torch.stack([o.op.a.detach().cpu() for o in model.ops])  # (Nr, n, n, n)


@torch.no_grad()
def algebra_metrics(model):
    """How close each relation's learned algebra is to the exact quaternion (op=ph/quat).

    NOTE: Frobenius distance to the *exact* Hamilton table -- not invariant to a change of
    basis / algebra isomorphism. A small distance => literally quaternion; a large distance
    is inconclusive (could be an isomorphic quaternion algebra). The saved checkpoints allow
    the deeper basis-invariant analysis offline.
    """
    A = algebra_tensors(model)
    if A is None or A.shape[1] != 4:
        return {}
    aq = _hamilton_table(A[0])                       # (4,4,4) on cpu
    dist = (A - aq).flatten(1).norm(dim=1)           # (Nr,) per-relation distance to quaternion
    norm = A.flatten(1).norm(dim=1)                  # (Nr,) algebra magnitude
    return {"algebra_dist_quat": dist.mean().item(),  # ||a_quat|| = 4.0 for reference
            "algebra_dist_quat_std": dist.std().item(),
            "algebra_norm": norm.mean().item()}


@torch.no_grad()
def embedding_viz(model, ds, op, order, epoch, max_points=1500):
    """wandb 3D PCA scatter: tail embeddings vs op_r(head) predictions, colored by relation.
    Shows whether relation-transformed heads land on their tails, and the cloud's spread."""
    n = min(max_points, len(ds.triples))
    idx = torch.randperm(len(ds.triples))[:n].to(model.entity_emb.weight.device)
    tr = ds.triples.to(idx.device)[idx]
    h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
    pred = model.apply_relation(model.entity_emb(h), r)
    t_emb = model.entity_emb(t)
    pca = fit_pca([t_emb, pred])
    return embedding_scatter3d(t_emb, pred, epoch, op, order, s0_labels=r, s1_labels=r,
                               names=("tail", "op(head)"), pca=pca, max_points=max_points)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_ds = KGTripleDataset(args.dataset, "train", create_inverse=True)
    valid_ds = KGTripleDataset(args.dataset, "valid", create_inverse=True)
    test_ds = KGTripleDataset(args.dataset, "test", create_inverse=True)
    hr2t = train_ds.true_tails([train_ds, valid_ds, test_ds])  # filter against all known positives
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"device={device}  op={args.op}  dataset={args.dataset}  nd={args.nd}  "
          f"Ne={train_ds.num_entities}  Nr={train_ds.num_relations}")

    model = KGEModel(train_ds.num_entities, train_ds.num_relations, args.nd, args.op,
                     args.order, args.op_resid, args.rank, args.unit_norm,
                     quat_init=args.quat_init).to(device)
    model.apply_mode = args.apply
    n_emb = model.entity_emb.weight.numel()
    n_op = sum(p.numel() for o in model.ops for p in o.parameters() if p.requires_grad)
    algebra = "frozen-quat" if args.op == "quat" else ("quat-init learnable" if args.quat_init else "random learnable")
    print(f"trainable params: entities={n_emb}  relation_ops={n_op}  | algebra: {algebra}")

    op_lr = args.op_lr if args.op_lr is not None else args.lr
    opt = torch.optim.AdamW([
        {"params": model.entity_emb.parameters(), "lr": args.lr, "weight_decay": 0.0},
        {"params": [p for o in model.ops for p in o.parameters()], "lr": op_lr,
         "weight_decay": args.weight_decay},
    ])
    # LR schedule. onecycle steps per-batch (cosine up-then-down to/from --max-lr);
    # warmup is a per-epoch linear ramp. Both param groups follow the same schedule.
    if args.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=len(loader))
        sched_per_batch = True
    elif args.scheduler == "warmup" and args.warmup > 0:
        sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=max(args.warmup_start_lr / args.lr, 1e-6), total_iters=args.warmup)
        sched_per_batch = False
    else:
        sched, sched_per_batch = None, False
    # Attraction loss: mse pulls pred onto t (L2 geometry); cos maximizes direction match
    # (consistent with --score cos; may sharpen rank-1 vs. MSE).
    if args.sim == "cos":
        sim_fn = lambda p, q: (1.0 - torch.cosine_similarity(p, q, dim=-1)).mean()
    else:
        sim_fn = nn.MSELoss()

    order_str = f"_{args.order}" if args.op not in ("trans",) else ""
    run_name = f"{args.dataset}_{args.op}{order_str}_nd{args.nd}_lambd{args.lambd}"
    if args.tag:
        run_name += f"_{args.tag}"
    if not args.no_wandb:
        # Per-dataset project so e.g. FB15k runs don't mix with WN18RR (override with --wandb-project).
        project = args.wandb_project or f"hoplas-kge-{args.dataset}"
        wandb.init(project=project, name=run_name, config=vars(args))
        wandb.define_metric("epoch"); wandb.define_metric("*", step_metric="epoch")
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{run_name}.pt")

    def save_ckpt(epoch, metrics, tag="final"):
        torch.save({"epoch": epoch, "tag": tag, "args": vars(args), "metrics": metrics,
                    "entity_emb": model.entity_emb.weight.detach().cpu(),
                    "algebra": algebra_tensors(model),  # (Nr, n, n, n) learned algebras, or None
                    "ops": model.ops.state_dict()}, ckpt_path.replace(".pt", f"_{tag}.pt"))

    R = train_ds.num_base_relations  # inverse of relation r is r+R (r<R) or r-R (r>=R)
    best_mrr = 0.0
    for epoch in range(1, args.epochs + 1):
        tot = dict(loss=0.0, sim=0.0, sigreg=0.0, mom=0.0, neg=0.0, inv=0.0)
        n = 0
        for chunk in loader:
            chunk = chunk.to(device)
            h, r, t = chunk[:, 0], chunk[:, 1], chunk[:, 2]
            pred, t_emb = model(h, r, t)
            sim = sim_fn(pred, t_emb)
            # SIGReg on a random sample of the entity table (anti-collapse, replaces negatives)
            idx = torch.randint(0, model.entity_emb.num_embeddings, (args.sigreg_n,), device=device)
            sigreg = SIGReg(model.entity_emb(idx), global_step=epoch)
            mom = MomMatchLoss(pred, t_emb, labels=r, diag=args.mom_diag) if args.lambda_mom > 0 else pred.new_zeros(())
            # Optional in-batch contrastive term (cosine InfoNCE): each pred should match its
            # own tail over the other tails in the batch -- a light discriminative pressure on
            # top of MSE+SIGReg (negatives, off by default to preserve the SIGReg-only regime).
            if args.lambda_neg > 0:
                pn = F.normalize(pred, dim=-1)
                tn = F.normalize(t_emb, dim=-1)
                logits = (pn @ tn.t()) / args.neg_temp
                neg = F.cross_entropy(logits, torch.arange(h.size(0), device=device))
            else:
                neg = pred.new_zeros(())
            # Optional explicit inverse-consistency term: the inverse relation's operator should
            # undo this relation's, i.e. op_{r_inv}(op_r(E[h])) ~ E[h]. The inverse relation
            # already trains independently on inverse triples; this ties the two ops as a true
            # round-trip identity (off by default).
            if args.lambda_inv > 0:
                r_inv = torch.where(r < R, r + R, r - R)
                back = model.apply_relation(pred, r_inv)
                inv = F.mse_loss(back, model.entity_emb(h))
            else:
                inv = pred.new_zeros(())
            loss = (1 - args.lambd) * (args.lambda_sim * sim + args.lambda_mom * mom
                                       + args.lambda_neg * neg + args.lambda_inv * inv) + args.lambd * sigreg
            opt.zero_grad(); loss.backward(); opt.step()
            if sched_per_batch:
                sched.step()
            bs = h.size(0); n += bs
            tot["loss"] += loss.item() * bs; tot["sim"] += sim.item() * bs
            tot["sigreg"] += sigreg.item() * bs; tot["mom"] += mom.item() * bs
            tot["neg"] += neg.item() * bs; tot["inv"] += inv.item() * bs
        if sched is not None and not sched_per_batch:
            sched.step()
        avg = {k: v / n for k, v in tot.items()}
        cur_lr = opt.param_groups[0]["lr"]
        msg = f"epoch {epoch:4d}/{args.epochs}  loss={avg['loss']:.5f}  sim={avg['sim']:.5f}  sigreg={avg['sigreg']:.5f}  mom={avg['mom']:.5f}"
        if args.lambda_inv > 0:
            msg += f"  inv={avg['inv']:.5f}"
        msg += f"  lr={cur_lr:.2e}"

        log = {"epoch": epoch, "lr": cur_lr, **{f"{k}_loss": v for k, v in avg.items()}}
        log.update(algebra_metrics(model))  # ||a - a_quat|| etc. (empty dict for non-ph/quat ops)
        if "algebra_dist_quat" in log:
            msg += f"  algΔquat={log['algebra_dist_quat']:.3f}"
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == 1):
            m = evaluate(model, valid_ds, hr2t, device, score=args.score)
            msg += f"  | val MRR={m['mrr']:.4f} H@10={m['h10']:.4f} H@1={m['h1']:.4f}"
            log.update({f"val_{k}": v for k, v in m.items()})
            if m["mrr"] > best_mrr:
                best_mrr = m["mrr"]
                save_ckpt(epoch, m, tag="best")
            if not args.no_wandb and args.viz_every and epoch % args.viz_every == 0:
                log["embedding"] = embedding_viz(model, valid_ds, args.op, args.order, epoch,
                                                 max_points=args.max_viz_points)
        print(msg, flush=True)
        if not args.no_wandb:
            wandb.log(log)

    # Report all three scoring functions at final test eval, so one run shows whether
    # dot/cos (which drop the spurious -||E||^2 per-entity term) beat l2 on H@1/MRR.
    test_scores = {sc: evaluate(model, test_ds, hr2t, device, score=sc)
                   for sc in ["l2", "dot", "cos"]}
    for sc in ["l2", "dot", "cos"]:
        m = test_scores[sc]
        print(f"TEST[{sc}]  MRR={m['mrr']:.4f}  MR={m['mr']:.1f}  "
              f"H@1={m['h1']:.4f}  H@3={m['h3']:.4f}  H@10={m['h10']:.4f}", flush=True)
    final = dict(test_scores[args.score])
    final.update(algebra_metrics(model))
    # Plain TEST line kept for scripts/results.sh compatibility (reflects --score, default l2).
    print(f"\nTEST  MRR={final['mrr']:.4f}  MR={final['mr']:.1f}  "
          f"H@1={final['h1']:.4f}  H@3={final['h3']:.4f}  H@10={final['h10']:.4f}"
          + (f"  algΔquat={final['algebra_dist_quat']:.3f}" if "algebra_dist_quat" in final else ""), flush=True)
    save_ckpt(args.epochs, final, tag="final")
    print(f"saved checkpoints: {ckpt_path.replace('.pt', '_best.pt')} , _final.pt")
    if not args.no_wandb:
        log = {"epoch": args.epochs}
        for sc, m in test_scores.items():
            log.update({f"test_{sc}_{k}": v for k, v in m.items()})
        log.update({f"test_{k}": v for k, v in final.items()})
        wandb.log(log)
        wandb.finish()
    return final


def main():
    p = configargparse.ArgumentParser(description="KGE as the ring task (relations = learnable group actions).")
    p.add_argument("--config", is_config_file=True, help="path to a config file (keys = dest names with underscores)")
    p.add_argument("--dataset", default="WN18RR")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--nd", type=int, default=200, help="entity embedding dim (divisible by --order)")
    p.add_argument("--op", choices=["filmr", "filmr_expm", "matop", "matop_clip", "matop2",
                                     "ph", "quat", "kquat", "kdualquat", "trans"], default="ph")
    p.add_argument("--order", type=int, default=4, help="hypercomplex order n (nd divisible by order)")
    p.add_argument("--op-resid", action=argparse.BooleanOptionalAction, default=True,
                   help="relation as x + op(x) (centers at identity)")
    p.add_argument("--unit-norm", action=argparse.BooleanOptionalAction, default=False,
                   help="L2-normalize op output (off for KGE: SIGReg wants Gaussian, not sphere)")
    p.add_argument("--rank", type=int, default=2, help="rotation-plane rank for filmr_expm")
    p.add_argument("--score", choices=["l2", "dot", "cos"], default="l2",
                   help="ranking score for val/checkpoint-selection (final test reports all three)")
    p.add_argument("--apply", choices=["loop", "vec", "check"], default="loop",
                   help="relation-application path: loop (reference), vec (fast einsum, PHM ops only), "
                        "check (run both and assert they match)")
    p.add_argument("--lr", type=float, default=0.01, help="entity-embedding LR")
    p.add_argument("--op-lr", type=float, default=None, help="relation-op LR (default: --lr)")
    p.add_argument("--weight-decay", type=float, default=0.0, help="weight decay on relation ops")
    p.add_argument("--lambd", type=float, default=0.05, help="SIGReg weight: (1-lambd)*sim + lambd*sigreg")
    p.add_argument("--lambda-sim", type=float, default=1.0)
    p.add_argument("--sim", choices=["mse", "cos"], default="mse",
                   help="attraction loss: mse (L2) or cos (1 - cosine_similarity); pair cos with --score cos")
    p.add_argument("--lambda-mom", type=float, default=0.0, help="MomMatch weight (0 disables)")
    p.add_argument("--lambda-neg", type=float, default=0.0,
                   help="in-batch cosine-InfoNCE contrastive weight (0 disables; ~0.01 balances MSE)")
    p.add_argument("--neg-temp", type=float, default=0.05, help="temperature for the contrastive term")
    p.add_argument("--lambda-inv", type=float, default=0.0,
                   help="explicit inverse-consistency weight: MSE(op_{r_inv}(op_r(E[h])), E[h]) (0 disables)")
    p.add_argument("--mom-diag", action="store_true")
    p.add_argument("--sigreg-n", type=int, default=4096, help="entities sampled per step for SIGReg")
    p.add_argument("--quat-init", action=argparse.BooleanOptionalAction, default=False,
                   help="(op=ph only) warm-start each relation's algebra at the exact quaternion, kept trainable")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--viz-every", type=int, default=0,
                   help="log a 3D-PCA embedding scatter (tail vs op(head)) to wandb every N epochs (0 disables)")
    p.add_argument("--max-viz-points", type=int, default=1500, help="points in the embedding scatter")
    p.add_argument("--scheduler", choices=["none", "warmup", "onecycle"], default="warmup",
                   help="LR schedule: warmup=linear ramp (per-epoch); onecycle=OneCycleLR (per-batch)")
    p.add_argument("--max-lr", type=float, default=0.1, help="peak LR for --scheduler onecycle")
    p.add_argument("--warmup", type=int, default=0, help="linear LR-ramp epochs (0 disables)")
    p.add_argument("--warmup-start-lr", type=float, default=1e-4, help="LR at epoch 1 of the ramp")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default=None,
                   help="wandb project name (default: hoplas-kge-<dataset>, keeps datasets separate)")
    args = p.parse_args()
    if args.op not in ("trans",):  # trans has no order requirement
        assert args.nd % args.order == 0, f"nd={args.nd} must be divisible by order={args.order}"
    print("Arguments:", vars(args))
    train(args)


if __name__ == "__main__":
    main()
