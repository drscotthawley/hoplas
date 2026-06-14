import torch.nn as nn
import torch.nn.functional as F


def _norm(kind, dim):
    if kind == "batch": return nn.BatchNorm1d(dim)
    if kind == "layer": return nn.LayerNorm(dim)
    return nn.Identity()


class ResBlock(nn.Module):
    """n_hid -> n_hid block with a residual skip (dims match, so add is valid)."""
    def __init__(self, n_hid, norm="none"):
        super().__init__()
        self.fc = nn.Linear(n_hid, n_hid)
        self.norm = _norm(norm, n_hid)
        self.act = nn.GELU()

    def forward(self, z):
        return z + self.act(self.norm(self.fc(z)))


class Projector(nn.Module):
    """h(): a learned nonlinear map nd -> pnd (via n_hid).

    Input (nd->n_hid) and output (n_hid->pnd) layers change dims so they're plain;
    the middle n_hid->n_hid layers use residual skips.
    pnd defaults to nd (square map). proj_resid is disabled when nd != pnd.
    """
    def __init__(self, nd=64, pnd=None, n_hid=32, n_layers=3, norm="none", proj_resid=False, unit_norm=True):
        super().__init__()
        assert n_layers >= 2, "need at least an input and output layer"
        self.pnd = pnd if pnd is not None else nd
        self.proj_resid = proj_resid and (nd == self.pnd)  # skip only valid when dims match
        self.unit_norm = unit_norm
        self.in_proj = nn.Linear(nd, n_hid)
        self.in_norm = _norm(norm, n_hid)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList(ResBlock(n_hid, norm) for _ in range(n_layers - 2))
        self.out_proj = nn.Linear(n_hid, self.pnd)

    def forward(self, y):
        z = self.act(self.in_norm(self.in_proj(y)))
        for block in self.blocks:
            z = block(z)
        z = self.out_proj(z)
        z = y + z if self.proj_resid else z
        return F.normalize(z, dim=-1) if self.unit_norm else z
