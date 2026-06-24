#!/bin/bash
# Show the multi-score final-eval lines (TEST[l2] / TEST[dot] / TEST[cos]) from run logs,
# for runs trained with the multi-score build of train_kge.py.
#
# Usage:
#   ./scripts/scores.sh <host> [config_glob]   # glob (no .log) defaults to 'kge_*'

HOST="${1:?Usage: $0 <host> [config_glob]}"
GLOB="${2:-kge_*}"
REPO_ARG="${HOPLAS_REMOTE_REPO:-github/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

$SSH "${HOST}" bash -s -- "$GLOB" "$REPO_ARG" << 'ENDSSH'
GLOB="$1"
case "$2" in /*) REPO="$2";; *) REPO="$HOME/$2";; esac
shopt -s nullglob
for L in "$REPO/logs"/$GLOB.log; do
    name=$(basename "$L" .log)
    lines=$(tr '\r' '\n' < "$L" | grep -E '^TEST\[')
    [ -z "$lines" ] && continue
    echo "=== $name ==="
    echo "$lines"
done
ENDSSH
