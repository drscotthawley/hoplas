"""Visualization helpers for hoplas (3D embedding scatter -> wandb)."""

import numpy as np
import plotly.graph_objects as go
import wandb


# Custom colorscales with NO shared endpoint: a blue ramp and an orange ramp,
# both truncated so the light end is a tinted color (not white) and the dark end
# is saturated -- so the two series never collide on either end. Colorblind-safe.
_BLUE_SCALE = [[0.0, "#9ecae1"], [0.5, "#4292c6"], [1.0, "#084594"]]   # light blue -> navy
_ORANGE_SCALE = [[0.0, "#fdae6b"], [0.5, "#f16913"], [1.0, "#8c2d04"]]  # light orange -> dark orange
_PURPLE_SCALE = [[0.0, "#bcbddc"], [0.5, "#807dba"], [1.0, "#4a1486"]]  # light purple -> dark purple
_GREEN_SCALE = [[0.0, "#a1d99b"], [0.5, "#41ab5d"], [1.0, "#006d2c"]]   # light green -> dark green

# default colorscale pair for the primary (rotation) plot; secondary ops use purple/green
PRIMARY_SCALES = (_BLUE_SCALE, _ORANGE_SCALE)
SECONDARY_SCALES = (_PURPLE_SCALE, _GREEN_SCALE)


def _to_np(t):
    return t.detach().cpu().float().numpy()


def fit_pca(arrays):
    """Shared 3D PCA basis fit on the concatenation of (B_i, nd) arrays/tensors.
    Returns (mean, comps) for nd>3, or None for nd<=3 (raw coords are already shared)."""
    arrs = [_to_np(a) for a in arrays]
    if arrs[0].shape[1] <= 3:
        return None
    stacked = np.concatenate(arrs, axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    comps = np.linalg.svd(stacked - mean, full_matrices=False)[2][:3]  # (3, nd)
    return mean, comps


def embedding_scatter3d(s0, s1, epoch, method, order=None,
                        s0_labels=None, s1_labels=None,
                        names=("yproj", "xproj_t"), scales=PRIMARY_SCALES,
                        symbols=("square", "diamond"), pca=None, title_suffix="",
                        max_points=1000, seed=0):
    """Return a wandb.Html of a 3D scatter of two series (target s0 + op output s1).

    Two (B, nd) tensors are subsampled to the same rows then projected to 3D. Pass
    `pca=(mean, comps)` (from fit_pca) to project in a shared basis so multiple plots
    line up; otherwise each plot fits its own PCA (or pads raw coords when nd<=3).
    `names`/`scales`/`symbols` style the two series (purple/green for secondary ops).
    """
    data = {names[0]: _to_np(s0), names[1]: _to_np(s1)}
    labels = {names[0]: _to_np(s0_labels) if s0_labels is not None else None,
              names[1]: _to_np(s1_labels) if s1_labels is not None else None}

    # subsample the same row indices across both series
    n_avail = min(a.shape[0] for a in data.values())
    n = min(max_points, n_avail)
    idx = np.random.default_rng(seed).choice(n_avail, size=n, replace=False)
    data = {k: a[idx] for k, a in data.items()}
    labels = {k: (v[idx] if v is not None else None) for k, v in labels.items()}

    nd = next(iter(data.values())).shape[1]
    if pca is not None:
        # shared PCA basis passed in -> all plots line up in the same frame
        mean, comps = pca
        proj = {k: (a - mean) @ comps.T for k, a in data.items()}
        axis_titles = ("PC1", "PC2", "PC3")
    elif nd <= 3:
        # already <=3D -- plot raw coords (no PCA), zero-padding to 3 cols when nd<3
        proj = {k: np.pad(a, ((0, 0), (0, 3 - a.shape[1]))) for k, a in data.items()}
        axis_titles = ("dim0", "dim1", "dim2")
    else:
        # own PCA to 3D, fit on both series together
        stacked = np.concatenate(list(data.values()), axis=0)
        mean = stacked.mean(axis=0, keepdims=True)
        comps = np.linalg.svd(stacked - mean, full_matrices=False)[2][:3]  # (3, nd)
        proj = {k: (a - mean) @ comps.T for k, a in data.items()}          # (n, 3)
        axis_titles = ("PC1", "PC2", "PC3")

    # shared label range so both colormaps span the same digits (measured, not hard-coded)
    present = [v for v in labels.values() if v is not None]
    cmin = min(float(v.min()) for v in present) if present else 0.0
    cmax = max(float(v.max()) for v in present) if present else 1.0

    series_style = list(zip(names, symbols, scales))
    fig = go.Figure()
    for i, (name, symbol, cmap) in enumerate(series_style):
        p = proj[name]
        lab = labels[name]
        hovertext = [f"{name} #{int(v)}" for v in lab] if lab is not None else None
        if lab is not None:
            marker = dict(size=3.5, symbol=symbol, opacity=0.8,
                          color=lab, colorscale=cmap, cmin=cmin, cmax=cmax,
                          showscale=True,  # one colorbar per series, offset so they don't overlap
                          colorbar=dict(title=dict(text=name, side="right"),
                                        x=1.02 + 0.14 * i, thickness=14, len=0.75))
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
    title += title_suffix
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(itemsizing="constant"),
        scene=dict(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1], zaxis_title=axis_titles[2],
                   aspectmode="data"),  # equal distance scale on every axis (no per-axis zoom)
        margin=dict(l=0, r=110, t=40, b=0),  # right room for the two colorbars
    )
    return wandb.Html(fig.to_html(full_html=True, include_plotlyjs="cdn"))
