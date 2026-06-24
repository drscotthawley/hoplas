#!/bin/bash
# Kill a process on the remote host by PID (e.g. the remote_queue runner).
# Use responsibly -- do not kill training jobs without authorization.
#
# Usage:
#   ./scripts/kill.sh <host> <pid>

HOST="${1:?Usage: $0 <host> <pid>}"
PID="${2:?Usage: $0 <host> <pid>}"
SSH="ssh -o ClearAllForwardings=yes"
$SSH "${HOST}" "kill ${PID} 2>/dev/null && echo 'killed ${PID}' || echo 'no such pid ${PID}'"
