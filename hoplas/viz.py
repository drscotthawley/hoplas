"""Visualization helpers for hoplas (3D embedding scatter -> wandb)."""

import numpy as np
import plotly.graph_objects as go
import wandb


# (series, plotly 3d marker symbol, colorscale)  -- points colored by digit label;
# symbols (limited set for Scatter3d) distinguish the two series.
_SERIES_STYLE = [
    ("yproj",   "square",  "Viridis"),
    ("xproj_t", "diamond", "Plasma"),
]


def _to_np(t):
    return t.detach().cpu().float().numpy()


def embedding_scatter3d(yproj, xproj_t, epoch, method, order=None,
                        yproj_labels=None, xproj_t_labels=None,
                        max_points=1000, seed=0):
    """Return a wandb.Html of a 3D PCA scatter of yproj and xproj_t.

    The two (B, nd) tensors are subsampled to the same rows, then projected
    to 3D with a single shared PCA so they're directly comparable. Optional
    per-point labels (ring indices) are rendered as text next to each marker.
    Log it with e.g. wandb.log({"embedding": embedding_scatter3d(...)}).
    """
    data = {
        "yproj": _to_np(yproj),
        "xproj_t": _to_np(xproj_t),
    }
    labels = {
        "yproj": _to_np(yproj_labels) if yproj_labels is not None else None,
        "xproj_t": _to_np(xproj_t_labels) if xproj_t_labels is not None else None,
    }

    # subsample the same row indices across both series
    n_avail = min(a.shape[0] for a in data.values())
    n = min(max_points, n_avail)
    idx = np.random.default_rng(seed).choice(n_avail, size=n, replace=False)
    data = {k: a[idx] for k, a in data.items()}
    labels = {k: (v[idx] if v is not None else None) for k, v in labels.items()}

    nd = next(iter(data.values())).shape[1]
    if nd == 3:
        # already 3D -- plot raw coords, no PCA
        proj = data
        axis_titles = ("dim0", "dim1", "dim2")
    else:
        # shared PCA to 3D, fit on both series together
        stacked = np.concatenate(list(data.values()), axis=0)
        mean = stacked.mean(axis=0, keepdims=True)
        comps = np.linalg.svd(stacked - mean, full_matrices=False)[2][:3]  # (3, nd)
        proj = {k: (a - mean) @ comps.T for k, a in data.items()}          # (n, 3)
        axis_titles = ("PC1", "PC2", "PC3")

    # shared label range so both colormaps span the same digits (measured, not hard-coded)
    present = [v for v in labels.values() if v is not None]
    cmin = min(float(v.min()) for v in present) if present else 0.0
    cmax = max(float(v.max()) for v in present) if present else 1.0

    fig = go.Figure()
    for name, symbol, cmap in _SERIES_STYLE:
        p = proj[name]
        lab = labels[name]
        hovertext = [f"{name} #{int(v)}" for v in lab] if lab is not None else None
        if lab is not None:
            marker = dict(size=3.5, symbol=symbol, opacity=0.8,
                          color=lab, colorscale=cmap, cmin=cmin, cmax=cmax, showscale=False)
        else:
            marker = dict(size=3.5, symbol=symbol, opacity=0.8, color="gray")
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2],
            mode="markers", name=name,
            hovertext=hovertext,  # shown alongside default x/y/z in hover
            marker=marker,
        ))

    title = f"epoch {epoch} — {method}"
    if method == "ph" and order is not None:
        title += f" (order={order})"
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(itemsizing="constant"),
        scene=dict(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1], zaxis_title=axis_titles[2],
                   aspectmode="data"),  # equal distance scale on every axis (no per-axis zoom)
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return wandb.Html(fig.to_html(full_html=True, include_plotlyjs="cdn"))
