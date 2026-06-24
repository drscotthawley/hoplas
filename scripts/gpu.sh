#!/bin/bash
# Fleet / slot overview for hoplas training on a remote GPU host.
# Shows GPU memory/util, compute apps, and running train_kge/train_ops processes,
# so we can respect the parallel-run cap and detect when a slot frees.
#
# Usage:
#   ./scripts/gpu.sh <host>

HOST="${1:?Usage: $0 <host>}"
SSH="ssh -o ClearAllForwardings=yes"

$SSH "${HOST}" bash -s << 'ENDSSH'
echo "=== $(hostname)  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
echo "--- compute apps (pid, mem) ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
echo "--- training processes ---"
ps -eo pid,etime,args | grep -E '[t]rain_(kge|ops)\.py' || echo "(none)"
n=$(ps -eo args | grep -cE '[t]rain_(kge|ops)\.py')
echo "active runs: ${n}"
ENDSSH
