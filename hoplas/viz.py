"""Visualization helpers for hoplas (3D embedding scatter -> wandb)."""

import numpy as np
import plotly.graph_objects as go
import wandb


# (label, plotly 3d marker symbol, color)  -- symbols limited for Scatter3d
_SERIES_STYLE = [
    ("xproj",   "circle",  "#1f77b4"),  # blue circles
    ("yproj",   "square",  "#ff7f0e"),  # orange squares
    ("xproj_t", "diamond", "#2ca02c"),  # green diamonds
]


def _to_np(t):
    return t.detach().cpu().float().numpy()


def embedding_scatter3d(xproj, yproj, xproj_t, epoch, method, order=None,
                        max_points=300, seed=0):
    """Return a wandb.Html of a 3D PCA scatter of the three series.

    The three (B, nd) tensors are subsampled to the same rows, then projected
    to 3D with a single shared PCA so they're directly comparable. Log it with
    e.g. wandb.log({"embedding": embedding_scatter3d(...)}).
    """
    data = {
        "xproj": _to_np(xproj),
        "yproj": _to_np(yproj),
        "xproj_t": _to_np(xproj_t),
    }

    # subsample the same row indices across all three series
    n_avail = min(a.shape[0] for a in data.values())
    n = min(max_points, n_avail)
    idx = np.random.default_rng(seed).choice(n_avail, size=n, replace=False)
    data = {k: a[idx] for k, a in data.items()}

    # shared PCA to 3D, fit on all three series together
    stacked = np.concatenate(list(data.values()), axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    comps = np.linalg.svd(stacked - mean, full_matrices=False)[2][:3]  # (3, nd)
    proj = {k: (a - mean) @ comps.T for k, a in data.items()}          # (n, 3)

    fig = go.Figure()
    for label, symbol, color in _SERIES_STYLE:
        p = proj[label]
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2],
            mode="markers", name=label,
            marker=dict(size=3.5, symbol=symbol, color=color, opacity=0.75),
        ))

    title = f"epoch {epoch} — {method}"
    if method == "ph" and order is not None:
        title += f" (order={order})"
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(itemsizing="constant"),
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return wandb.Html(fig.to_html(full_html=True, include_plotlyjs="cdn"))
