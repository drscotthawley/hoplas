#!/bin/bash
# Check the status of a hoplas training run on a remote host.
#
# Usage:
#   ./scripts/status.sh <host>              # most recently modified log
#   ./scripts/status.sh <host> <config>     # e.g. kge_ph_nd400 (no .log)
#
# Env:
#   HOPLAS_REMOTE_REPO   repo path on host (relative to $HOME, or absolute); default github/hoplas

HOST="${1:?Usage: $0 <host> [config_name]}"
CONFIG="${2:-}"
REPO_ARG="${HOPLAS_REMOTE_REPO:-github/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

# Quoted heredoc: nothing expands locally. Values are passed as positional args so $HOME
# (and the repo path) expand on the *remote* host -- avoids the old quoted-tilde bug.
$SSH "${HOST}" bash -s -- "$CONFIG" "$REPO_ARG" << 'ENDSSH'
CONFIG="$1"
case "$2" in /*) REPO="$2";; *) REPO="$HOME/$2";; esac
LOGS_DIR="$REPO/logs"

if [ -n "$CONFIG" ]; then
    LOG="${LOGS_DIR}/${CONFIG}.log"
else
    LOG=$(ls -t "${LOGS_DIR}"/*.log 2>/dev/null | head -1)
fi
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    echo "No log found (looked in ${LOGS_DIR})"
    exit 1
fi
echo "Log: $LOG"
echo ""

# Is the log growing?
LOG_GROWING=0
SIZE1=$(wc -c < "$LOG")
sleep 2
SIZE2=$(wc -c < "$LOG")
[ "$SIZE2" -gt "$SIZE1" ] && LOG_GROWING=1

# Match this run's training process (train_kge or train_ops) by its config file.
MAIN_PID=$(ps -eo pid,args | grep -E "[t]rain_(kge|ops)\.py" | grep -F "${CONFIG}.cfg" | awk '{print $1}' | sort -n | head -1)
[ -z "$MAIN_PID" ] && MAIN_PID=$(ps -eo pid,args | grep -E "[t]rain_(kge|ops)\.py" | awk '{print $1}' | sort -n | head -1)

if [ "$LOG_GROWING" -eq 1 ]; then
    echo "Status: RUNNING (log growing$([ -n "$MAIN_PID" ] && echo ", PID $MAIN_PID"))"
    [ -n "$MAIN_PID" ] && echo "  $(ps -p $MAIN_PID -o args=)"
elif [ -n "$MAIN_PID" ]; then
    echo "Status: RUNNING (PID $MAIN_PID, log not growing)"
    echo "  $(ps -p $MAIN_PID -o args=)"
else
    echo "Status: NOT RUNNING"
fi
echo ""

echo "--- Last 5 lines ---"
tr '\r' '\n' < "$LOG" | grep -vE '[0-9]+it/s|%\|' | grep -v '^[[:space:]]*$' | tail -5
ENDSSH
echo ""
