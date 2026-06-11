#! /usr/bin/env python3

import torch
from hoplas.filmr import MatOp

torch.manual_seed(0)
nd, batch = 4, 8
m = MatOp(nd)
x = torch.randn(batch, nd)

out_matop = m(x)
out_ref   = x @ m.mat @ m.mat

assert torch.allclose(out_matop, out_ref, atol=1e-6), f"max diff: {(out_matop - out_ref).abs().max()}"
print("MatOp == x @ mat @ mat  ✓")

