"""Operator constructors for the ring task."""

import torch.nn as nn
import torch.nn.functional as F

from hoplas.filmr import FiLMR, FiLMR_expm, MatOp, MatOp2
from hoplas.ph_layers import PHMLinear


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
