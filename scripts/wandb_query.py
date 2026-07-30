#!/usr/bin/env python
"""Query Weights & Biases: search runs and pull their data.

Auth: uses ~/.netrc (api.wandb.ai entry) or $WANDB_API_KEY. Runs locally against
the W&B cloud — no remote host needed.

Invoke via the wrappers so it stays in the pre-approved `bash scripts/*.sh` lane:
    bash scripts/wandb_search.sh [opts]
    bash scripts/wandb_pull.sh   [opts]

Subcommands
-----------
search : list/filter runs in a project (or list projects if --project omitted)
pull   : dump a single run's summary / config / metric history

Examples
--------
    # what projects exist for the entity?
    bash scripts/wandb_search.sh --entity drscotthawley

    # list ring runs, show a few config + summary columns, markdown table
    bash scripts/wandb_search.sh --project ring \\
        --config-cols op,order,nd --metrics val_loss,recon_loss --md

    # filter by tag/state/name and sort
    bash scripts/wandb_search.sh --project ring-mnist --tag champ \\
        --state finished --sort val_loss --limit 20

    # pull one run's summary + config
    bash scripts/wandb_pull.sh --project ring --run abc123 --summary --config

    # pull a metric history to CSV
    bash scripts/wandb_pull.sh --project ring --run abc123 \\
        --history epoch,val_loss,recon_loss --out /tmp/run.csv
"""
import argparse
import csv
import json
import sys

DEFAULT_ENTITY = "drscotthawley"


def _api():
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed in this environment (pip install wandb).")
    try:
        return wandb.Api(timeout=30)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not init wandb API (check ~/.netrc / WANDB_API_KEY): {e}")


def _flatten(d, prefix=""):
    """Flatten nested dicts with dotted keys; leave scalars as-is."""
    out = {}
    for k, v in dict(d).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


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
        widths = [max(len(headers[i]), *(len(c[i]) for c in cells)) for i in range(len(headers))]
        line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
        print(line)
        print("  ".join("-" * widths[i] for i in range(len(headers))))
        for c in cells:
            print("  ".join(c[i].ljust(widths[i]) for i in range(len(headers))))


def cmd_search(args):
    api = _api()

    if not args.project:
        print(f"Projects for entity '{args.entity}':")
        try:
            for p in api.projects(args.entity):
                print(f"  {p.name}")
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Could not list projects: {e}")
        return

    # Build a server-side filter where cheap; refine client-side otherwise.
    filters = {}
    if args.state:
        filters["state"] = args.state
    if args.tag:
        filters["tags"] = {"$in": [args.tag]}
    for kv in args.config or []:
        k, _, v = kv.partition("=")
        filters[f"config.{k}.value"] = _coerce(v)

    path = f"{args.entity}/{args.project}"
    try:
        runs = api.runs(path, filters=filters or None, order=args.order)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not query runs at '{path}': {e}")

    cfg_cols = [c for c in (args.config_cols.split(",") if args.config_cols else []) if c]
    met_cols = [m for m in (args.metrics.split(",") if args.metrics else []) if m]
    headers = ["name", "id", "state"] + cfg_cols + met_cols + (["tags"] if args.show_tags else [])

    rows = []
    for run in runs:
        if args.name_contains and args.name_contains.lower() not in run.name.lower():
            continue
        cfg = _flatten(run.config)
        summ = _flatten(run.summary)
        row = {"name": run.name, "id": run.id, "state": run.state}
        for c in cfg_cols:
            row[c] = cfg.get(c)
        for m in met_cols:
            row[m] = summ.get(m)
        if args.show_tags:
            row["tags"] = ",".join(run.tags)
        rows.append(row)
        if len(rows) >= args.limit:
            break

    if args.sort:
        rows.sort(key=lambda r: (r.get(args.sort) is None, r.get(args.sort)), reverse=args.desc)

    _print_table(rows, headers, md=args.md)
    print(f"\n{len(rows)} run(s) shown from {path}", file=sys.stderr)


def _coerce(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def _resolve_run(api, args):
    """Find a run by id, or by display name within the project."""
    path = f"{args.entity}/{args.project}"
    if args.run:
        try:
            return api.run(f"{path}/{args.run}")
        except Exception:  # noqa: BLE001
            pass  # not an id; fall through to name search
    target = args.run or args.name
    for run in api.runs(path):
        if run.name == target or run.id == target:
            return run
    sys.exit(f"No run matching '{target}' in {path}")


def cmd_pull(args):
    api = _api()
    run = _resolve_run(api, args)
    print(f"# run: {run.name}  (id={run.id}, state={run.state})  {run.url}", file=sys.stderr)

    if args.config:
        print("## config")
        print(json.dumps(_flatten(run.config), indent=2, default=str))
    if args.summary:
        print("## summary")
        print(json.dumps(_flatten(run.summary), indent=2, default=str))

    if args.history:
        keys = [k for k in args.history.split(",") if k]
        rows = list(run.scan_history(keys=keys))
        if not rows:
            print("(no history rows for those keys)", file=sys.stderr)
            return
        out = open(args.out, "w", newline="") if args.out else sys.stdout
        writer = csv.DictWriter(out, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        if args.out:
            out.close()
            print(f"wrote {len(rows)} rows × {len(keys)} cols -> {args.out}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Search and pull data from Weights & Biases.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--entity", default=DEFAULT_ENTITY, help=f"W&B entity (default: {DEFAULT_ENTITY})")
    common.add_argument("--project", default=None, help="W&B project (omit in search to list projects)")

    s = sub.add_parser("search", parents=[common], help="list/filter runs in a project")
    s.add_argument("--config-cols", default="", help="comma-list of config keys to show as columns")
    s.add_argument("--metrics", default="", help="comma-list of summary metric keys to show as columns")
    s.add_argument("--tag", default=None, help="only runs with this tag")
    s.add_argument("--state", default=None, help="filter by state (finished/running/crashed/failed)")
    s.add_argument("--config", action="append", help="config filter key=value (repeatable)")
    s.add_argument("--name-contains", default=None, help="client-side substring filter on run name")
    s.add_argument("--sort", default=None, help="column to sort by (config/metric/name)")
    s.add_argument("--desc", action="store_true", help="sort descending")
    s.add_argument("--order", default="-created_at", help="server-side order (default: -created_at)")
    s.add_argument("--limit", type=int, default=50, help="max runs to show")
    s.add_argument("--show-tags", action="store_true", help="add a tags column")
    s.add_argument("--md", action="store_true", help="emit a markdown table")
    s.set_defaults(func=cmd_search)

    g = sub.add_parser("pull", parents=[common], help="dump one run's summary/config/history")
    g.add_argument("--run", default=None, help="run id (or display name)")
    g.add_argument("--name", default=None, help="run display name (alternative to --run)")
    g.add_argument("--summary", action="store_true", help="print the run summary dict")
    g.add_argument("--config", action="store_true", help="print the run config dict")
    g.add_argument("--history", default=None, help="comma-list of metric keys to export as CSV")
    g.add_argument("--out", default=None, help="write history CSV here (default: stdout)")
    g.set_defaults(func=cmd_pull)

    args = p.parse_args()
    if args.cmd == "pull" and not args.project:
        sys.exit("pull requires --project")
    args.func(args)


if __name__ == "__main__":
    main()
