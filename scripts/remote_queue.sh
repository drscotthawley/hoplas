#!/bin/bash
# Start a self-refilling training queue ON the remote host (survives laptop sleep).
# Syncs source + configs, ships remote_runner.sh, and nohups it on the host. The runner
# keeps up to --par jobs going (global cap over all training procs), refilling from the
# given config list as slots free.
#
# Usage:
#   ./scripts/remote_queue.sh <host> [--par N] [--gpu ID] [--poll S] <items...>
#
# Each <item> is one of:
#   configs/x.cfg (or a glob)  -> train_ops.py  (train_kge.py for kge_* configs)
#   vae:<dataset>              -> scripts/train_vae.py --dataset <dataset>  (cifar10|fashion|mnist)
#   clf:<dataset>              -> scripts/train_classifier.py --dataset <dataset>  (mnist|fashion|cifar10)
#   recon:<dataset>            -> scripts/score_recon.py --dataset <dataset>  (k=0 recon-ceiling eval)
# (This replaces the retired launch.sh / launch_queue.sh: a single config is just a 1-item queue.)
#
# Anything after a literal "--" is passed through to every launched job (validated to plain
# CLI flags only), e.g.:  ./scripts/remote_queue.sh lecun vae:mnist -- --fresh
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
# Split args on a literal "--": items before it, pass-through flags after it (applied to every
# launched job; harmless where a script ignores them, e.g. --fresh on cifar/fashion).
ITEMS=(); EXTRA=(); seen_dashdash=0
for a in "$@"; do
    if [[ $seen_dashdash -eq 0 && "$a" == "--" ]]; then seen_dashdash=1; continue; fi
    if [[ $seen_dashdash -eq 1 ]]; then EXTRA+=("$a"); else ITEMS+=("$a"); fi
done
[[ ${#ITEMS[@]} -gt 0 ]] || { echo "No items given."; exit 1; }
# Safety: pass-through args are re-parsed by the remote shell, so allow only plain CLI
# flags/values (letters, digits, . _ / = -). Reject anything with spaces or shell metacharacters
# -- this permits --flag and --flag=value, but never arbitrary commands.
for a in "${EXTRA[@]}"; do
    [[ "$a" =~ ^[A-Za-z0-9._/=-]+$ ]] || { echo "Refusing unsafe pass-through arg: '$a'"; exit 1; }
done
CONFIGS=()
for f in "${ITEMS[@]}"; do CONFIGS+=("$(basename "$f" .cfg)"); done

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
$SSH "${HOST}" "mkdir -p ${REPO_ARG}/scripts"
rsync -az "${REPO_DIR}/scripts/train_vae.py" "${HOST}:${REPO_ARG}/scripts/train_vae.py"
rsync -az "${REPO_DIR}/scripts/train_classifier.py" "${HOST}:${REPO_ARG}/scripts/train_classifier.py"
rsync -az "${REPO_DIR}/scripts/score_recon.py" "${HOST}:${REPO_ARG}/scripts/score_recon.py"

$SSH "${HOST}" bash -s -- "$PAR" "$GPU" "$POLL" "$REPO_ARG" "$ENV_ARG" "${CONFIGS[@]}" -- "${EXTRA[@]}" << 'ENDSSH'
PAR="$1"; GPU="$2"; POLL="$3"; RA="$4"; EA="$5"; shift 5
case "$RA" in /*) REPO="$RA";; *) REPO="$HOME/$RA";; esac
mkdir -p "$REPO/logs" "$REPO/checkpoints"
LOG="$REPO/logs/_remote_queue.log"
chmod +x "$REPO/remote_runner.sh"
nohup bash "$REPO/remote_runner.sh" "$PAR" "$GPU" "$POLL" "$RA" "$EA" "$@" >> "$LOG" 2>&1 &
echo "remote_queue runner PID $! ; log: $LOG"
ENDSSH
