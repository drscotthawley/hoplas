#!/usr/bin/env python
"""Pull ring/dihedral (train_ops.py) results from W&B with the columns that actually
gate quality on the ops task:

  sec_sim  reflect/secondary-head latent fit (lower better)
  sim      primary-op latent fit
  recon    projector->inv_proj autoencoder round-trip (OOD canary #1)
  ratio    var(xproj_t) / var(yproj) -- transform-cloud spread vs target (OOD canary #2;
           ~1 is healthy, >>1 = over-spread off-manifold, <<1 = diversity collapse)

`sec_sim` alone can look fine while the *decoded* transform is garbage (see the
lambda-neg over-spread failure). This puller flags that case: a run with a good
sec_sim but a bad recon or a spread ratio far from 1 gets a warning tag, so
"looks fine on paper, bad images" runs don't slip through.

Invoke via the wrapper (pre-approved `bash scripts/*.sh` lane):
    bash scripts/ops_results.sh [--project dihedral-mnist] [--name-contains STR]
                                [--sort sec_sim|recon|ratio|sim|epoch] [--md]
"""
import argparse
import sys

DEFAULT_ENTITY = "drscotthawley"
DEFAULT_PROJECT = "dihedral-mnist"

# thresholds for the likely-bad-decode (OOD) warning -- fired on the canaries
# directly, so an over-spread run is flagged even when its sec_sim also degraded.
RECON_BAD = 0.035           # recon round-trip this high -> decode likely OOD (R)
RATIO_HI, RATIO_LO = 1.4, 0.6   # transform-cloud over-spread / collapsed vs target (S)


def _api():
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed (pip install wandb).")
    return wandb.Api(timeout=30)


def _get(summary, k):
    try:
        v = dict(summary).get(k)
        return v if isinstance(v, (int, float)) else None
    except Exception:  # noqa: BLE001
        return None


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def _print_table(rows, headers, md=False):
    if not rows:
        print("(no runs matched)")
        return
    cells = [[_fmt(r.get(h, "")) for h in headers] for r in rows]
    if md:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for c in cells:
            print("| " + " | ".join(c) + " |")
    else:
        w = [max(len(headers[i]), *(len(c[i]) for c in cells)) for i in range(len(headers))]
        print("  ".join(headers[i].ljust(w[i]) for i in range(len(headers))))
        print("  ".join("-" * w[i] for i in range(len(headers))))
        for c in cells:
            print("  ".join(c[i].ljust(w[i]) for i in range(len(headers))))


def main():
    p = argparse.ArgumentParser(description="Ring/dihedral results with recon + spread-ratio OOD canaries.")
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT, help=f"W&B project (default: {DEFAULT_PROJECT})")
    p.add_argument("--name-contains", default=None, help="client-side substring filter on run name")
    p.add_argument("--state", default=None, help="filter by state (finished/running/...)")
    p.add_argument("--sort", default="sec_sim", help="column to sort ascending (sec_sim/recon/ratio/sim/epoch)")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--md", action="store_true", help="emit a markdown table")
    args = p.parse_args()

    api = _api()
    path = f"{args.entity}/{args.project}"
    try:
        runs = api.runs(path, filters=({"state": args.state} if args.state else None))
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not query runs at '{path}': {e}")

    rows = []
    for run in runs:
        if args.name_contains and args.name_contains.lower() not in run.name.lower():
            continue
        s = run.summary
        sec = _get(s, "sec_sim_loss")
        recon = _get(s, "recon_loss")
        vxt, vy = _get(s, "var_xproj_t"), _get(s, "var_yproj")
        ratio = (vxt / vy) if (vxt is not None and vy) else None
        flag = ""
        if recon is not None and recon > RECON_BAD:
            flag += "R"
        if ratio is not None and (ratio > RATIO_HI or ratio < RATIO_LO):
            flag += "S"
        rows.append({"name": run.name, "state": run.state, "epoch": _get(s, "epoch"),
                     "sim": _get(s, "sim_loss"), "sec_sim": sec, "recon": recon,
                     "ratio": ratio, "flag": flag or "-"})
        if len(rows) >= args.limit:
            break

    rows.sort(key=lambda r: (r.get(args.sort) is None, r.get(args.sort) if r.get(args.sort) is not None else 0))
    _print_table(rows, ["name", "state", "epoch", "sim", "sec_sim", "recon", "ratio", "flag"], md=args.md)
    print("\nflag: R=recon round-trip high, S=transform over/under-spread (var ratio off) "
          "-> decode likely OOD even if sec_sim looks fine", file=sys.stderr)
    print(f"{len(rows)} run(s) from {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
