import torch
import torch.nn.functional as F


def _cov(x):
    """Sample covariance (D, D) of x: (N, D) with N >= 2."""
    x = x - x.mean(0, keepdim=True)
    return (x.T @ x) / (x.size(0) - 1)


def MomMatchLoss(a, b, labels=None, diag=False, cov_weight=1.0, min_n=2, return_stats=False):
    """Class-conditional moment (mean + covariance) matching between a transformed
    source cloud `a` and a target cloud `b`.

    a, b: (N, D), index-aligned pairs (a[k] = transformed source of class i,
    b[k] = its target of class i+1). labels: (N,) source class per pair; both a and b
    are grouped by the *same* index, so grouping reproduces the (i, i+1) correspondence.
    labels=None pools all classes (weak: a ring rotation matches pooled moments trivially).

    Returns mean over groups of  ||mu_a - mu_b||^2 + cov_weight * ||Sigma_a - Sigma_b||_F^2.
    The covariance term is the load-bearing piece: it forbids the variance collapse that
    element-wise MSE induces, so a near-bijective op carries within-class diversity forward.
    diag=True matches per-dim variances only (robust when D is large vs. samples/class).
    return_stats=True also returns {"var_a", "var_b"}: mean within-class total variance,
    the collapse diagnostic (var_a << var_b means the op is contracting toward the centroid).
    """
    if labels is None:
        groups = [torch.ones(a.size(0), dtype=torch.bool, device=a.device)]
    else:
        groups = [labels == c for c in labels.unique()]
    total = a.new_zeros(())
    count = 0
    var_a = var_b = 0.0
    for m in groups:
        ag, bg = a[m], b[m]
        if ag.size(0) < min_n:
            continue
        l_mean = (ag.mean(0) - bg.mean(0)).pow(2).sum()
        if diag:
            l_cov = (ag.var(0, unbiased=False) - bg.var(0, unbiased=False)).pow(2).sum()
        else:
            l_cov = (_cov(ag) - _cov(bg)).pow(2).sum()
        total = total + l_mean + cov_weight * l_cov
        count += 1
        if return_stats:
            var_a += ag.var(0, unbiased=False).sum().item()
            var_b += bg.var(0, unbiased=False).sum().item()
    loss = total / max(count, 1)
    if return_stats:
        n = max(count, 1)
        return loss, {"var_a": var_a / n, "var_b": var_b / n}
    return loss


def SIGReg(x, global_step, num_slices=256, chunk_size=32):
    """SIGReg with Epps-Pulley statistic. x is (N, K) tensor.
       Chunked to reduce memory pressure -> More GPU utilization. :-)"""
    with torch.amp.autocast('cuda', enabled=False): # accum in float32
        x = x.float()
        device = x.device
        g = torch.Generator(device=device).manual_seed(global_step)
        A = torch.randn((x.size(1), num_slices), generator=g, device=device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-10)
        t = torch.linspace(-5, 5, 17, device=device)
        exp_f = torch.exp(-0.5 * t**2)
        T_total = torch.tensor(0.0, device=device)    # float32 accumulator
        if chunk_size < 1: chunk_size = num_slices    # < 1 Turns off chunking
        for i in range(0, num_slices, chunk_size):
            x_t = (x @ A[:, i:i+chunk_size]).unsqueeze(2) * t  # (N, chunk, T)
            ecf = (torch.exp(1j * x_t).mean(dim=0)).abs()
            diff = (ecf - exp_f).abs().square().mul(exp_f)
            T_total = T_total + torch.trapz(diff, t, dim=1).sum()
        return T_total


def InfoNCE(pred, tgt, temp=0.05):
    """In-batch cosine-InfoNCE negative-repulsion loss. pred, tgt: (N, D) index-aligned
    pairs; each pred must pick its own tgt (the diagonal) out of all N tgts in the batch,
    under cosine similarity at temperature `temp`. This is the in-batch negative term that
    lifted the KGE results; shared by train_kge.py (op(h) vs tail) and train_ops.py
    (op(x) vs target).

    NB: with few distinct classes (e.g. MNIST's 10) many in-batch tgts share a class and
    so act as (false) negatives — the term bites harder there than in the KGE entity table.
    """
    pn = F.normalize(pred, dim=-1)
    tn = F.normalize(tgt, dim=-1)
    logits = (pn @ tn.t()) / temp
    return F.cross_entropy(logits, torch.arange(pred.size(0), device=pred.device))