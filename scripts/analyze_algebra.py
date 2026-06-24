#!/usr/bin/env python3
"""Classify the learned PHM relation algebra(s) in saved KGE checkpoints.

Each relation stores a structure tensor a of shape (n,n,n). a[i] is the left-multiplication
matrix L_i of basis element e_i:  (e_i . v)_p = sum_q a[i][p,q] v_q.

For n=2 (complex-sized) we (1) test whether the learned product is a unital / commutative /
associative 2D algebra (relative residuals, ~0 = yes), and (2) classify it basis-invariantly
via the discriminant of a generator's minimal polynomial:
    take generator g=e1 and unit u; write g^2 = a*u + b*g; D = a + (b/2)^2.
    D<0 -> isomorphic to C (complex);  D>0 -> split-complex;  D~0 -> dual numbers.
The SIGN of D is invariant to real change of basis, so it is the meaningful label.
"""
import sys
import glob
from collections import Counter

import torch


def classify_2d(a):
    a = a.double()
    L = [a[0], a[1]]                      # L_i[p,q] = (e_i . e_q)_p
    aF = a.norm().item() + 1e-12
    I = torch.eye(2, dtype=a.dtype)

    # unit element c: sum_i c_i L_i = I
    M = torch.stack([L[0].reshape(-1), L[1].reshape(-1)], dim=1)        # (4,2)
    c = torch.linalg.lstsq(M, I.reshape(-1)).solution
    unit_res = (M @ c - I.reshape(-1)).norm().item() / (2 ** 0.5)

    # commutativity: e_i e_j = col j of L_i ; e_j e_i = col i of L_j
    comm = 0.0
    for i in range(2):
        for j in range(2):
            comm += (L[i][:, j] - L[j][:, i]).pow(2).sum().item()
    comm_res = (comm ** 0.5) / aF

    # associativity: L_{e_i e_j} = L_i @ L_j, with e_i e_j = col j of L_i
    assoc = 0.0
    for i in range(2):
        for j in range(2):
            w = L[i][:, j]
            Lw = w[0] * L[0] + w[1] * L[1]
            assoc += (Lw - L[i] @ L[j]).pow(2).sum().item()
    assoc_res = (assoc ** 0.5) / (aF ** 2 / 2)   # ~quadratic in L, normalize by ~||L||^2

    # classify: generator g=e1, unit u=c; g^2 = alpha*u + beta*g
    u = c
    g = torch.tensor([0.0, 1.0], dtype=a.dtype)
    gg = L[1][:, 1]                                # coords of e1^2
    B = torch.stack([u, g], dim=1)                # (2,2)
    sol = torch.linalg.lstsq(B, gg).solution
    alpha, beta = sol[0].item(), sol[1].item()
    D = alpha + (beta / 2) ** 2
    scale = (a.pow(2).mean().sqrt()).item() + 1e-12
    Dn = D / (scale ** 2)
    cls = "C(complex)" if Dn < -1e-2 else ("split-complex" if Dn > 1e-2 else "dual~0")
    return dict(unit=unit_res, comm=comm_res, assoc=assoc_res, Dn=Dn, cls=cls)


def main(paths):
    for p in paths:
        ck = torch.load(p, map_location="cpu")
        A = ck.get("algebra")
        if A is None:
            print(f"\n=== {p}: no algebra tensor ==="); continue
        Nr, n = A.shape[0], A.shape[1]
        print(f"\n=== {p}  (Nr={Nr}, n={n}) ===")
        if n != 2:
            print(f"  n={n} (not complex-sized); 2D classification skipped"); continue
        cnt = Counter(); u = c = s = 0.0
        for r in range(Nr):
            res = classify_2d(A[r])
            cnt[res["cls"]] += 1
            u += res["unit"]; c += res["comm"]; s += res["assoc"]
            print(f"  rel {r:2d}: {res['cls']:13s} Dn={res['Dn']:+8.3f}  "
                  f"rel.resid[unit={res['unit']:.3f} comm={res['comm']:.3f} assoc={res['assoc']:.3f}]")
        N = max(Nr, 1)
        print(f"  SUMMARY: {dict(cnt)}  mean rel.resid["
              f"unit={u/N:.3f} comm={c/N:.3f} assoc={s/N:.3f}]")
        print("  (resid ~0 => genuinely a unital/commutative/associative algebra; "
              "Dn<0 => complex, >0 => split-complex, ~0 => dual)")


if __name__ == "__main__":
    paths = []
    for g in sys.argv[1:]:
        paths += sorted(glob.glob(g))
    if not paths:
        print("no checkpoints matched:", sys.argv[1:]); sys.exit(1)
    main(paths)
