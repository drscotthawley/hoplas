#!/bin/bash
# Launch configs on a remote host, keeping MAX_PAR slots busy.
# Polls by PID; when a slot frees, launches the next config in the queue.
#
# Usage:
#   ./scripts/launch_queue.sh <host> [max_parallel] [gpu_id] [config_files...]
#
# Examples:
#   ./scripts/launch_queue.sh lecun              # all configs/mnist_*.cfg, 2 parallel, GPU 0
#   ./scripts/launch_queue.sh lecun 2 0 configs/mnist_filmr_expm_rank*.cfg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOST="${1:?Usage: $0 <host> [max_parallel] [gpu_id] [config_files...]}"
MAX_PAR="${2:-2}"
GPU="${3:-0}"
shift 3 2>/dev/null || shift $#   # remaining args are optional config files
POLL_INTERVAL=60   # seconds between status checks
LAUNCH_GRACE=10    # seconds to wait after launch before polling

SSH="ssh -o ClearAllForwardings=yes"
REMOTE_REPO="${HOPLAS_REMOTE_REPO:-~/github/hoplas}"
REMOTE_ENV="${HOPLAS_REMOTE_ENV:-~/envs/hoplas}"

# Build queue: from explicit args or default glob
if [[ $# -gt 0 ]]; then
    QUEUE=($(for f in "$@"; do basename "${f}" .cfg; done))
else
    QUEUE=($(ls "${REPO_DIR}"/configs/mnist_*.cfg 2>/dev/null | xargs -n1 basename | sed 's/\.cfg$//'))
fi
[[ ${#QUEUE[@]} -eq 0 ]] && { echo "No configs/mnist_*.cfg found."; exit 1; }

echo "Queue (${#QUEUE[@]} configs): ${QUEUE[*]}"
echo "Max parallel: ${MAX_PAR}  GPU: ${GPU}  Poll interval: ${POLL_INTERVAL}s"
echo ""

# Sync source once upfront
echo "Syncing source to ${HOST}:${REMOTE_REPO}..."
rsync -az --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
    --exclude='wandb' --exclude='checkpoints' --exclude='*.pt' \
    "${REPO_DIR}/hoplas/" "${HOST}:${REMOTE_REPO}/hoplas/"
rsync -az "${REPO_DIR}/train_ring.py" "${HOST}:${REMOTE_REPO}/"
rsync -az "${REPO_DIR}/configs/" "${HOST}:${REMOTE_REPO}/configs/"
$SSH "${HOST}" "mkdir -p ${REMOTE_REPO}/logs ${REMOTE_REPO}/checkpoints"
echo ""

QUEUE_IDX=0
ACTIVE_PIDS=()
ACTIVE_NAMES=()

launch_next() {
    [[ $QUEUE_IDX -ge ${#QUEUE[@]} ]] && return 1
    local config="${QUEUE[$QUEUE_IDX]}"
    QUEUE_IDX=$((QUEUE_IDX + 1))
    local remote_config="${REMOTE_REPO}/configs/${config}.cfg"
    local log="${REMOTE_REPO}/logs/${config}.log"

    cat > /tmp/hoplas_run.sh << EOF
#!/bin/bash
source ${REMOTE_ENV}/bin/activate
cd ${REMOTE_REPO}
CUDA_VISIBLE_DEVICES=${GPU} nohup python train_ring.py --config ${remote_config} \
    > ${log} 2>&1 &
echo \$!
EOF
    scp -q /tmp/hoplas_run.sh "${HOST}:/tmp/hoplas_run_${config}.sh"
    local pid
    pid=$($SSH "${HOST}" "bash /tmp/hoplas_run_${config}.sh")
    echo "[$(date '+%H:%M:%S')] Launched ${config}  PID=${pid}  log=${log}"
    ACTIVE_PIDS+=("$pid")
    ACTIVE_NAMES+=("$config")
    return 0
}

is_pid_running() {
    $SSH "${HOST}" "kill -0 ${1} 2>/dev/null && echo yes || echo no" 2>/dev/null | grep -q "^yes$"
}

# Fill initial slots
while [[ ${#ACTIVE_PIDS[@]} -lt $MAX_PAR ]] && [[ $QUEUE_IDX -lt ${#QUEUE[@]} ]]; do
    launch_next
done
echo "(grace period ${LAUNCH_GRACE}s before first poll...)"
sleep "$LAUNCH_GRACE"

# Main loop
while [[ ${#ACTIVE_PIDS[@]} -gt 0 ]]; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S')  active=${#ACTIVE_PIDS[@]}  queued=$((${#QUEUE[@]} - QUEUE_IDX)) ==="

    new_pids=()
    new_names=()
    freed=0
    for i in "${!ACTIVE_PIDS[@]}"; do
        pid="${ACTIVE_PIDS[$i]}"
        name="${ACTIVE_NAMES[$i]}"
        if is_pid_running "$pid"; then
            echo "  RUNNING  ${name}  (PID ${pid})"
            new_pids+=("$pid")
            new_names+=("$name")
        else
            echo "  DONE     ${name}  (PID ${pid})"
            freed=$((freed + 1))
        fi
    done
    ACTIVE_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    ACTIVE_NAMES=("${new_names[@]+"${new_names[@]}"}")

    # Refill freed slots
    while [[ $freed -gt 0 ]] && [[ $QUEUE_IDX -lt ${#QUEUE[@]} ]]; do
        launch_next && freed=$((freed - 1)) || break
        sleep "$LAUNCH_GRACE"
    done

    [[ ${#ACTIVE_PIDS[@]} -gt 0 ]] && sleep "$POLL_INTERVAL"
done

echo ""
echo "=== All ${#QUEUE[@]} configs complete. ==="
