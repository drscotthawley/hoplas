"""
Fixed-algebra linear layers derived from the kingdon geometric algebra library.
The algebra structure tensor is extracted from kingdon and registered as a frozen
buffer; only the s-matrices (scale weights) are trainable, same as PHMLinear
with a frozen a-tensor.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def _build_quat_structure_tensor():
    """Extract the 4x4x4 quaternion structure tensor from kingdon Algebra(0,2).
    Convention matches PHMLinear: a[k, row, col] = coeff of basis_k in (basis_col * basis_row).
    Verified to match the hand-coded Hamilton table in freeze_quaternion().
    """
    from kingdon import Algebra
    alg = Algebra(0, 2)
    names = list(alg.blades.keys())   # ['e', 'e1', 'e2', 'e12']
    basis = [alg.blades[n] for n in names]
    n = len(names)
    A = torch.zeros(n, n, n)
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis):
            prod = bi * bj
            for k, name in enumerate(names):
                A[k, j, i] = float(getattr(prod, name, 0))
    return A


class KingdonQuaternion(nn.Module):
    """Quaternion linear layer with algebra fixed by kingdon Algebra(0,2).

    Equivalent to PHMLinear(n=4) with a-tensor frozen to the Hamilton table,
    but the algebra is derived from kingdon rather than hand-coded.
    The s-matrices remain fully learnable.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        assert in_features % 4 == 0, "in_features must be divisible by 4"
        assert out_features % 4 == 0, "out_features must be divisible by 4"
        self.n = 4
        self.in_features = in_features
        self.out_features = out_features
        m_in = in_features // 4
        m_out = out_features // 4

        self.register_buffer("a", _build_quat_structure_tensor())

        self.s = nn.Parameter(
            init.xavier_uniform_(torch.zeros(4, m_out, m_in))
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x.to(dtype=self.s.dtype)
        X = x.reshape(x.shape[0], self.n, -1)
        Y = torch.einsum("iab,ijk,Bbk->Baj", self.a, self.s, X)
        return Y.reshape(x.shape[0], -1) + self.bias

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"


def _build_dual_quat_structure_tensor():
    """Extract the 8x8x8 dual quaternion structure tensor from kingdon Algebra(0,2,1).
    Basis: {e, e0, e1, e2, e01, e02, e12, e012}
    where e0 is the dual unit (e0^2=0), {e,e1,e2,e12} is the real quaternion subalgebra.
    Convention: a[k, row, col] = coeff of basis_k in (basis_col * basis_row).
    """
    from kingdon import Algebra
    alg = Algebra(0, 2, 1)
    names = list(alg.blades.keys())   # ['e','e0','e1','e2','e01','e02','e12','e012']
    basis = [alg.blades[n] for n in names]
    n = len(names)
    A = torch.zeros(n, n, n)
    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis):
            prod = bi * bj
            for k, name in enumerate(names):
                A[k, j, i] = float(getattr(prod, name, 0))
    return A


class KingdonDualQuaternion(nn.Module):
    """Dual quaternion linear layer with algebra fixed by kingdon Algebra(0,2,1).

    8-dimensional: real part {e,e1,e2,e12} (quaternions) plus dual part
    {e0,e01,e02,e012} where e0 is the dual unit (e0^2=0).
    Requires in_features and out_features divisible by 8.
    The s-matrices are fully learnable; the algebra is frozen.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        assert in_features % 8 == 0, "in_features must be divisible by 8"
        assert out_features % 8 == 0, "out_features must be divisible by 8"
        self.n = 8
        self.in_features = in_features
        self.out_features = out_features
        m_in = in_features // 8
        m_out = out_features // 8

        self.register_buffer("a", _build_dual_quat_structure_tensor())

        self.s = nn.Parameter(
            init.xavier_uniform_(torch.zeros(8, m_out, m_in))
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x.to(dtype=self.s.dtype)
        X = x.reshape(x.shape[0], self.n, -1)
        Y = torch.einsum("iab,ijk,Bbk->Baj", self.a, self.s, X)
        return Y.reshape(x.shape[0], -1) + self.bias

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"
