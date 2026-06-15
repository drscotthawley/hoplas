#!/bin/bash
# Check the status of a hoplas training run on a remote host.
#
# Usage:
#   ./scripts/status.sh <host>              # most recently modified log
#   ./scripts/status.sh <host> <config>     # e.g. mnist_ph_4_nonorm (no .log)

HOST="${1:?Usage: $0 <host> [config_name]}"
CONFIG="${2:-}"
REMOTE_REPO="${HOPLAS_REMOTE_REPO:-~/github/hoplas}"
LOGS_DIR="${REMOTE_REPO}/logs"

ssh -o ClearAllForwardings=yes "${HOST}" bash << ENDSSH
if [ -n "${CONFIG}" ]; then
    LOG="${LOGS_DIR}/${CONFIG}.log"
else
    LOG=\$(ls -t ${LOGS_DIR}/*.log 2>/dev/null | head -1)
fi
if [ -z "\$LOG" ] || [ ! -f "\$LOG" ]; then
    echo "No log found (looked in ${LOGS_DIR})"
    exit 1
fi
echo "Log: \$LOG"
echo ""

# Check if log is growing
LOG_GROWING=0
SIZE1=\$(wc -c < "\$LOG")
sleep 2
SIZE2=\$(wc -c < "\$LOG")
[ "\$SIZE2" -gt "\$SIZE1" ] && LOG_GROWING=1

MAIN_PID=\$(ps aux | grep '[p]ython.*train_ring' | awk '{print \$2}' | sort -n | head -1)
if [ "\$LOG_GROWING" -eq 1 ]; then
    echo "Status: RUNNING (log growing\$([ -n "\$MAIN_PID" ] && echo ", PID \$MAIN_PID"))"
    [ -n "\$MAIN_PID" ] && echo "  \$(ps -p \$MAIN_PID -o cmd=)"
elif [ -n "\$MAIN_PID" ]; then
    echo "Status: RUNNING (PID \$MAIN_PID, log not growing)"
    echo "  \$(ps -p \$MAIN_PID -o cmd=)"
else
    echo "Status: NOT RUNNING"
fi
echo ""

echo "--- Last 5 lines ---"
tr '\r' '\n' < "\$LOG" | grep -vE '[0-9]+it/s|%\|' | grep -v '^\s*$' | tail -5
ENDSSH
echo ""
