"""Operator constructors for the ring task.
  SEE ALSO the included ph_layers.py (by E. Grassucci) for the PH Layer definitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hoplas.filmr import FiLMR, FiLMR_expm, MatOp, MatOp2
from hoplas.ph_layers import PHMLinear
from hoplas.kingdon_layers import KingdonQuaternion, KingdonDualQuaternion


class Translation(nn.Module):
    """TransE-style per-relation translation: op(x) = b (independent of x). With op_resid
    (x + op(x)) this gives t ≈ h + b — a pure translation baseline for composability tests
    (translations accumulate error across hops in a bounded space; rotations/PHM should not)."""
    def __init__(self, nd: int):
        super().__init__()
        self.b = nn.Parameter(torch.randn(nd) * 0.1)

    def forward(self, x):
        return self.b.expand_as(x)


class OpWrapper(nn.Module):
    """Wraps a transform op, optionally as a residual x + op(x).
    op_resid centers the transform at identity (good for many ring points;
    redundant for filmr_expm, already near-identity via matrix_exp)."""
    def __init__(self, method: str, nd: int, order: int, op_resid: bool = False, rank: int = 2,
                 unit_norm: bool = True):
        super().__init__()
        self.op_resid = op_resid
        self.unit_norm = unit_norm
        if method == "filmr":
            self.op = FiLMR(nd=nd)
        elif method == "filmr_expm":
            self.op = FiLMR_expm(nd=nd, rank=rank)
        elif method == "matop":
            self.op = MatOp(nd=nd)
        elif method == "matop_clip":
            self.op = MatOp(nd=nd, spectral_clip=True)
        elif method == "matop2":
            self.op = MatOp2(nd=nd)
        elif method == "ph":
            if nd % order != 0:
                raise ValueError(f"nd={nd} must be divisible by order={order} for PHMLinear")
            self.op = PHMLinear(n=order, in_features=nd, out_features=nd)
        elif method == "quat":
            if nd % 4 != 0:
                raise ValueError(f"nd={nd} must be divisible by 4 for frozen quaternion")
            # rand_init_a=False so RNG consumption matches KingdonQuaternion (kquat),
            # letting the two share identical s/bias/inv_proj init under the same seed.
            self.op = PHMLinear(n=4, in_features=nd, out_features=nd, rand_init_a=False)
        elif method == "kquat":
            if nd % 4 != 0:
                raise ValueError(f"nd={nd} must be divisible by 4 for KingdonQuaternion")
            self.op = KingdonQuaternion(in_features=nd, out_features=nd)
        elif method == "kdualquat":
            if nd % 8 != 0:
                raise ValueError(f"nd={nd} must be divisible by 8 for KingdonDualQuaternion")
            self.op = KingdonDualQuaternion(in_features=nd, out_features=nd)
        elif method == "trans":
            self.op = Translation(nd)  # TransE-style x + b_r (composability control)
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, x):
        out = x + self.op(x) if self.op_resid else self.op(x)
        return F.normalize(out, dim=-1) if self.unit_norm else out





def algebra_tensors(ops):
    """Stacked per-relation algebra tensors (Nr, n, n, n), or None if the op has no `a`."""
    if not all(hasattr(o.op, "a") for o in ops):
        return None
    return torch.stack([o.op.a.detach().cpu() for o in ops])  # (Nr, n, n, n)



# Quaternion-specific ops: 
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



@torch.no_grad()
def algebra_metrics(ops):
    """How close each relation's learned algebra is to the exact quaternion (op=ph/quat).

    NOTE: Frobenius distance to the *exact* Hamilton table -- not invariant to a change of
    basis / algebra isomorphism. A small distance => literally quaternion; a large distance
    is inconclusive (could be an isomorphic quaternion algebra). The saved checkpoints allow
    the deeper basis-invariant analysis offline.
    """
    A = algebra_tensors(ops)
    if A is None or A.shape[1] != 4:
        return {}
    aq = _hamilton_table(A[0])                       # (4,4,4) on cpu
    dist = (A - aq).flatten(1).norm(dim=1)           # (Nr,) per-relation distance to quaternion
    norm = A.flatten(1).norm(dim=1)                  # (Nr,) algebra magnitude
    return {"algebra_dist_quat": dist.mean().item(),  # ||a_quat|| = 4.0 for reference
            "algebra_dist_quat_std": dist.std().item(),
            "algebra_norm": norm.mean().item()}
