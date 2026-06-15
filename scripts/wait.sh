#!/bin/bash
# Block until a hoplas training run on a remote host finishes.
# Polls status every INTERVAL seconds.
#
# Usage:
#   ./scripts/wait.sh <host>
#   ./scripts/wait.sh <host> <config_name>       # e.g. mnist_ph_4_nonorm
#   ./scripts/wait.sh <host> "" <interval>       # custom poll interval (seconds)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${1:?Usage: $0 <host> [config_name] [interval_seconds]}"
CONFIG="${2:-}"
INTERVAL="${3:-120}"
MAX_FAILURES=5

failures=0
seen_running=0
while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    output=$(bash "${SCRIPT_DIR}/status.sh" "${HOST}" "${CONFIG}" 2>&1)
    status=$?
    echo "$output"
    if [[ $status -ne 0 ]] || ! echo "$output" | grep -q "Status:"; then
        failures=$((failures + 1))
        echo "[warn] SSH/status failed (attempt ${failures}/${MAX_FAILURES}); will retry..."
        [[ $failures -ge $MAX_FAILURES ]] && { echo "[error] ${MAX_FAILURES} consecutive failures — giving up."; exit 1; }
    else
        failures=0
        if echo "$output" | grep -q "Status: RUNNING"; then
            seen_running=1
        elif [[ $seen_running -eq 1 ]]; then
            break
        fi
    fi
    echo "(next check in ${INTERVAL}s...)"
    sleep "${INTERVAL}"
done
