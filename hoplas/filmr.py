# code from original Oplas repo by Hawley

import torch
import torch.nn as nn
import math

# Numpy original from https://github.com/tulip-control/polytope/blob/main/polytope/polytope.py
# Converted to PyTorch by SHH]

def givens_rotation_matrix(i, j, theta, N, device=None, eye=None):
    """Return the Givens rotation matrix for an N-dimensional space."""
    R = eye.clone() if eye is not None else torch.eye(N, device=device)
    c = torch.cos(theta)
    s = torch.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def solve_rotation_ap(u, v, check_vecs=False, debug=True, eye=None):
    """Return the rotation matrix for the rotation in the plane defined by the
    vectors u and v across TWICE the angle between u and v.

    This algorithm uses the Aguilera-Perez Algorithm cite{Aguilera} (https://dspace5.zcu.cz/bitstream/11025/6178/1/N29.pdf)
    to generate the rotation matrix. The algorithm works basically as follows:

    Starting with the Nth component of u, rotate u towards the (N-1)th
    component until the Nth component is zero. Continue until u is parallel to
    the 0th basis vector. Next do the same with v until it only has none zero
    components in the first two dimensions. The result will be something like
    this:

    [[u0,  0, 0 ... 0],
     [v0, v1, 0 ... 0]]

    Now it is trivial to align u with v. Apply the inverse rotations to return
    to the original orientation.

    NOTE: The precision of this method is limited by sin, cos, and arctan
    functions.
    Also NOTE: Reversing order of u,v -> v,u yields R.T
    """
    # BTW: pretty safe to assume u & v have same dims, on same device
    device = u.device
    N = len(u)                       # the number of dimensions
    M = torch.eye(N, device=device)  # accumulates rotation matrix (don't use eye kwarg here)

    # optional: maybe save a bit of time for (anti-)parallel or zero u & v
    if check_vecs and u.norm() * v.norm() == torch.dot(u,v).abs():
        if debug:
            print(f"solve_rotation_ap: zero or (anti-)parallel u,v: 0 degree rotation")
        return M

    assert len(u.shape)==1, f"u ({list(u.shape)}) & v ({list(v.shape)}) should be single vectors, not batches"
    uv = torch.stack([u, v], axis=1)  # the plane of rotation
    # ensure u has positive basis0 component
    if uv[0, 0] < 0:
        M[0, 0] = -1
        M[1, 1] = -1
        uv = M.matmul(uv)
    # align uv plane with the basis01 plane and u with basis0.
    for c in range(2):
        for r in range(N - 1, c, -1):
            if uv[r, c] != 0:  # skip rotations when theta will be zero
                theta = torch.arctan2(uv[r, c], uv[r - 1, c])
                Mk = givens_rotation_matrix(r, r - 1, theta, N, device=device, eye=eye)
                uv = Mk.matmul(uv)
                M = Mk.matmul(M)
    # rotate u onto v
    theta = 2 * torch.arctan2(uv[1, 1], uv[0, 1])
    if debug:
        print(f"solve_rotation_ap: {180 * theta / math.pi:6.2f} degree rotation")
    R = givens_rotation_matrix(0, 1, theta, N, device=device, eye=eye)
    # perform M rotations in reverse order
    M_inverse = M.T
    R = M_inverse.matmul(R.matmul(M))
    return R


def rotate_batch(R, v_batch):
    "simple utility function used for other things"
    return v_batch @ R.T   # linear algebra FTW


def get_rot_2d(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

class FiLMR2d(nn.Module):
    "affine transformation plus rotation, in 2d. Only use for testing / sanity checks of general FILMR (below)"
    def __init__(self, nd=2,
                 beta_init_fac = 0.0001, # tiny beta for FILM is maybe cheating
                 theta_init_fac=6.28/100, # 2pi/n init is "cheating" a bit
                 ):
        super().__init__()
        self.gamma =  nn.Parameter(torch.ones((1)))
        self.beta = nn.Parameter(beta_init_fac * torch.randn((1)))
        self.theta = nn.Parameter( theta_init_fac * torch.ones((1)) )

    def forward(self, x):
        rot = get_rot_2d(self.theta)
        return (x * self.gamma + self.beta) @ rot.to(x.device)


def get_rot_nd(u, v, debug=False, eye=None):
    """Return the rotation matrix that rotates u onto v."""
    return solve_rotation_ap(u, v, debug=debug, eye=eye)

class FiLMR(nn.Module):
    "affine transformation plus rotation, in nd"
    def __init__(self, nd=3,
                 beta_init_fac = 0.0001, # tiny beta is maybe cheating
                 uv_diff_fac = 3.0, # difference scale between initial u and v
                 ):
        super().__init__()
        self.gamma =  nn.Parameter(torch.ones((1)))
        self.beta = nn.Parameter(beta_init_fac * torch.randn((1)))
        self.u = nn.Parameter( torch.randn((nd)) )
        self.v = nn.Parameter( self.u + uv_diff_fac*torch.randn((nd)) )
        self.register_buffer('eye', torch.eye(nd))

    def forward(self, x, debug=False):
        rot = get_rot_nd(self.u, self.v, debug=debug, eye=self.eye)
        return (x * self.gamma + self.beta) @ rot.to(x.device)


class FiLMR_expm(nn.Module):
    """FiLM + rotation via a low-rank skew-symmetric generator and matrix exp.

    The generator is built from k = rank//2 vector pairs (u_i, v_i):
        A = sum_i (u_i v_i^T - v_i u_i^T) = U^T V - V^T U   (skew, rank <= rank)
        R = expm(A)                                          (always in SO(nd))
    so `rank` caps how many independent rotation *planes* R can use:
      rank=2  -> a single-plane rotation (the right prior for the ring task),
      rank=nd -> recovers a general SO(nd) rotation.
    Gradients are well-conditioned everywhere (no arctan2 plateau near identity).
    """
    def __init__(self, nd=3, rank=2,
                 beta_init_fac = 0.001,   # tiny beta is maybe cheating
                 w_init_fac = 0.01,       # small generator => R starts near identity
                 ):
        super().__init__()
        assert rank % 2 == 0, f"rank must be even, got {rank}"
        assert rank <= nd, f"rank ({rank}) cannot exceed nd ({nd})"
        k = rank // 2                          # number of rotation planes
        self.gamma = nn.Parameter(torch.ones((1)))
        self.beta = nn.Parameter(beta_init_fac * torch.randn((1)))
        self.U = nn.Parameter(w_init_fac * torch.randn((k, nd)))
        self.V = nn.Parameter(w_init_fac * torch.randn((k, nd)))

    def forward(self, x):
        A = self.U.T @ self.V - self.V.T @ self.U   # rank-<=2k skew generator
        rot = torch.matrix_exp(A)                   # always in SO(nd)
        return (x * self.gamma + self.beta) @ rot

    @torch.no_grad()
    def rotation_angle_deg(self):
        """Total rotation angle (degrees) of the generator.

        For a single plane (rank=2) this is the exact rotation angle: the skew
        generator A has eigenvalues +-i*theta, and ||A||_F = sqrt(2)*theta.
        For higher rank it returns sqrt(sum of per-plane angles^2).
        """
        A = self.U.T @ self.V - self.V.T @ self.U
        theta = A.norm() / (2 ** 0.5)               # radians
        return float(theta * 180.0 / torch.pi)


####---- Baseline: Square Matrix Multiply

class MatOp(nn.Module):
    "just a square matrix operation"
    def __init__(self, nd=2, spectral_clip=False):
        super().__init__()
        self.mat = nn.Parameter(0.1 * torch.randn((nd,nd)))
        self.spectral_clip = spectral_clip

    def forward(self, x):
        W = self.mat.to(x.device)
        if self.spectral_clip:
            sigma = torch.linalg.matrix_norm(W, ord=2)
            W = W / sigma.clamp(min=1.0)
        x = W.T @ x.T
        x = x.T @ W
        return x


class MatOp2(nn.Module):
    "square matrix op applied once (x @ mat): cleaner equivalent if mat² was unintentional"
    def __init__(self, nd=2):
        super().__init__()
        self.mat = nn.Parameter(0.1 * torch.randn((nd,nd)))

    def forward(self, x):
        return x @ self.mat