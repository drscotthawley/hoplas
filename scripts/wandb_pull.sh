#!/bin/bash
# Pull one W&B run's summary / config / metric history.
# Thin wrapper around scripts/wandb_query.py `pull`. See that file for options.
#
#   bash scripts/wandb_pull.sh --project ring --run abc123 --summary --config
#   bash scripts/wandb_pull.sh --project ring --run abc123 \
#       --history epoch,val_loss,recon_loss --out /tmp/run.csv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/wandb_query.py" pull "$@"
