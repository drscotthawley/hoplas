#!/bin/bash
# Fast CPU smoke test of a config on the remote host: rsync source, then run 2 tiny
# epochs (nd=8, cpu, no wandb) to catch wiring/code-change bugs before a real launch.
#
# Usage:
#   ./scripts/smoke.sh <host> <config_file>
#
# Env:
#   HOPLAS_REMOTE_REPO   repo path on host (relative to $HOME, or absolute); default github/hoplas
#   HOPLAS_REMOTE_ENV    venv path on host  (relative to $HOME, or absolute); default envs/hoplas

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOST="${1:?Usage: $0 <host> <config_file>}"
CONFIG="${2:?Usage: $0 <host> <config_file>}"
CONFIG_NAME=$(basename "$CONFIG" .cfg)
[[ "$CONFIG_NAME" == kge_* ]] && TRAIN="train_kge.py" || TRAIN="train_ops.py"

REPO_ARG="${HOPLAS_REMOTE_REPO:-github/hoplas}"
ENV_ARG="${HOPLAS_REMOTE_ENV:-envs/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

echo "Syncing source to ${HOST}:${REPO_ARG} ..."
rsync -az --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
    --exclude='wandb' --exclude='checkpoints' --exclude='*.pt' \
    "${REPO_DIR}/hoplas/" "${HOST}:${REPO_ARG}/hoplas/"
rsync -az "${REPO_DIR}/train_ops.py" "${REPO_DIR}/train_kge.py" "${HOST}:${REPO_ARG}/"
rsync -az "${REPO_DIR}/configs/" "${HOST}:${REPO_ARG}/configs/"

echo "Running smoke test (${TRAIN}, ${CONFIG_NAME}) ..."
$SSH "${HOST}" bash -s -- "$CONFIG_NAME" "$TRAIN" "$REPO_ARG" "$ENV_ARG" << 'ENDSSH'
CFG="$1"; TRAIN="$2"
case "$3" in /*) REPO="$3";; *) REPO="$HOME/$3";; esac
case "$4" in /*) ENV="$4";;  *) ENV="$HOME/$4";; esac
source "$ENV/bin/activate"
cd "$REPO"
if [ "$TRAIN" = "train_kge.py" ]; then
    python "$TRAIN" --config "configs/${CFG}.cfg" \
        --epochs 2 --nd 8 --batch-size 8192 --cpu --no-wandb --eval-every 0
else
    # ring task (train_ops.py): --val-every 1 exercises the new closure/planarity metrics;
    # --warmup 0 lets the best-checkpoint save path run too. Uses the config's nd.
    python "$TRAIN" --config "configs/${CFG}.cfg" \
        --epochs 2 --batch-size 8192 --cpu --no-wandb --val-every 1 --warmup 0
fi
echo "smoke exit code: $?"
ENDSSH
