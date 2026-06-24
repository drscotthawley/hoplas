#!/bin/bash
# Analyze the learned PHM relation algebra(s) in saved checkpoints on the host (no GPU):
# classify each relation's n=2 algebra as complex / split-complex / dual, with
# unit/commutativity/associativity residuals. Ships analyze_algebra.py and runs it remotely.
#
# Usage:
#   ./scripts/analyze.sh <host> [ckpt_glob]
#   default glob: checkpoints/WN18RR_ph_2_nd512_lambd0.1_champ_*_best.pt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOST="${1:?Usage: $0 <host> [ckpt_glob]}"
GLOB="${2:-checkpoints/WN18RR_ph_2_nd512_lambd0.1_champ_*_best.pt}"
REPO_ARG="${HOPLAS_REMOTE_REPO:-github/hoplas}"
ENV_ARG="${HOPLAS_REMOTE_ENV:-envs/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

rsync -az "${REPO_DIR}/scripts/analyze_algebra.py" "${HOST}:${REPO_ARG}/analyze_algebra.py"

$SSH "${HOST}" bash -s -- "$GLOB" "$REPO_ARG" "$ENV_ARG" << 'ENDSSH'
GLOB="$1"
case "$2" in /*) REPO="$2";; *) REPO="$HOME/$2";; esac
case "$3" in /*) ENV="$3";;  *) ENV="$HOME/$3";; esac
source "$ENV/bin/activate"
cd "$REPO"
python analyze_algebra.py $GLOB
ENDSSH
