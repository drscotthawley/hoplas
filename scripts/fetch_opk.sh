#!/bin/bash
# Fetch op_k_*.csv transition-curve files (written by score_transitions.py) from a
# remote host into a local directory, for building the paper's op^k figures.
#
# Usage: ./scripts/fetch_opk.sh <host> <local_dest_dir>

HOST="${1:?Usage: $0 <host> <local_dest_dir>}"
DEST="${2:?Usage: $0 <host> <local_dest_dir>}"
REMOTE_REPO="${HOPLAS_REMOTE_REPO:-~/github/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

mkdir -p "$DEST"
echo "--- op_k csv files on ${HOST} ---"
$SSH "${HOST}" "ls -la ${REMOTE_REPO}/op_k_*.csv 2>/dev/null || echo 'none in repo root'; ls ${REMOTE_REPO}/scores/op_k_*.csv 2>/dev/null || true"
echo "--- copying ---"
scp -q "${HOST}:${REMOTE_REPO}/op_k_*.csv" "$DEST/" 2>/dev/null
scp -q "${HOST}:${REMOTE_REPO}/scores/op_k_*.csv" "$DEST/" 2>/dev/null
ls -la "$DEST"
