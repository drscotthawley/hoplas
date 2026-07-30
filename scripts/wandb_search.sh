#!/bin/bash
# Search/list W&B runs (or list projects if --project omitted).
# Thin wrapper around scripts/wandb_query.py `search`. See that file for options.
#
#   bash scripts/wandb_search.sh --entity drscotthawley           # list projects
#   bash scripts/wandb_search.sh --project ring --config-cols op,order,nd \
#       --metrics val_loss,recon_loss --md
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/wandb_query.py" search "$@"
