#!/bin/bash
# Dump the tail of a run log on the host (for diagnosing errors/OOM/stalls, or checking the
# remote_runner via config name "_remote_queue"). Cleans \r->\n. Optional grep pattern.
#
# Usage:
#   ./scripts/log.sh <host> <config> [N] [pattern]
#   e.g. ./scripts/log.sh lecun kge_fb237_nd512_neg05_1k 40
#        ./scripts/log.sh lecun _remote_queue 30
#        ./scripts/log.sh lecun kge_fb15k_champ 200 "Error|Traceback|OOM|epoch"

HOST="${1:?Usage: $0 <host> <config> [N] [pattern]}"
CONFIG="${2:?Usage: $0 <host> <config> [N] [pattern]}"
N="${3:-40}"
PATTERN="${4:-}"
REPO_ARG="${HOPLAS_REMOTE_REPO:-github/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

$SSH "${HOST}" bash -s -- "$CONFIG" "$N" "$REPO_ARG" "$PATTERN" << 'ENDSSH'
CONFIG="$1"; N="$2"; PATTERN="$4"
case "$3" in /*) REPO="$3";; *) REPO="$HOME/$3";; esac
LOG="$REPO/logs/$CONFIG.log"
if [ ! -f "$LOG" ]; then echo "No log: $LOG"; exit 1; fi
echo "Log: $LOG  ($(wc -l < "$LOG") lines)"
if [ -n "$PATTERN" ]; then
    tr '\r' '\n' < "$LOG" | grep -E "$PATTERN" | tail -n "$N"
else
    tr '\r' '\n' < "$LOG" | tail -n "$N"
fi
ENDSSH
