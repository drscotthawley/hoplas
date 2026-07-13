#!/bin/bash
# Start a self-refilling training queue ON the remote host (survives laptop sleep).
# Syncs source + configs, ships remote_runner.sh, and nohups it on the host. The runner
# keeps up to --par jobs going (global cap over all training procs), refilling from the
# given config list as slots free.
#
# Usage:
#   ./scripts/remote_queue.sh <host> [--par N] [--gpu ID] [--poll S] <config_files...>
#
# Monitor:  ./scripts/gpu.sh <host> ; ./scripts/results.sh <host> ;
#           ./scripts/status.sh <host> _remote_queue   (runner log)
# Stop:     ./scripts/kill.sh <host> <runner_pid>      (PID is printed below)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOST="${1:?Usage: $0 <host> [--par N] [--gpu ID] [--poll S] <configs...>}"
shift
PAR=3; GPU=0; POLL=60
while [[ $# -gt 0 ]]; do
    case "$1" in
        --par)  PAR="$2";  shift 2 ;;
        --gpu)  GPU="$2";  shift 2 ;;
        --poll) POLL="$2"; shift 2 ;;
        *) break ;;
    esac
done
[[ $# -gt 0 ]] || { echo "No config files given."; exit 1; }
CONFIGS=()
for f in "$@"; do CONFIGS+=("$(basename "$f" .cfg)"); done

REPO_ARG="${HOPLAS_REMOTE_REPO:-github/hoplas}"
ENV_ARG="${HOPLAS_REMOTE_ENV:-envs/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

echo "Queue (${#CONFIGS[@]}): ${CONFIGS[*]}"
echo "par=${PAR} gpu=${GPU} poll=${POLL}s  host=${HOST}"
echo "Syncing source to ${HOST}:${REPO_ARG} ..."
rsync -az --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
    --exclude='wandb' --exclude='checkpoints' --exclude='*.pt' \
    "${REPO_DIR}/hoplas/" "${HOST}:${REPO_ARG}/hoplas/"
rsync -az "${REPO_DIR}/train_ops.py" "${REPO_DIR}/train_kge.py" "${HOST}:${REPO_ARG}/"
rsync -az "${REPO_DIR}/configs/" "${HOST}:${REPO_ARG}/configs/"
rsync -az "${REPO_DIR}/scripts/remote_runner.sh" "${HOST}:${REPO_ARG}/remote_runner.sh"

$SSH "${HOST}" bash -s -- "$PAR" "$GPU" "$POLL" "$REPO_ARG" "$ENV_ARG" "${CONFIGS[@]}" << 'ENDSSH'
PAR="$1"; GPU="$2"; POLL="$3"; RA="$4"; EA="$5"; shift 5
case "$RA" in /*) REPO="$RA";; *) REPO="$HOME/$RA";; esac
mkdir -p "$REPO/logs" "$REPO/checkpoints"
LOG="$REPO/logs/_remote_queue.log"
chmod +x "$REPO/remote_runner.sh"
nohup bash "$REPO/remote_runner.sh" "$PAR" "$GPU" "$POLL" "$RA" "$EA" "$@" >> "$LOG" 2>&1 &
echo "remote_queue runner PID $! ; log: $LOG"
ENDSSH
