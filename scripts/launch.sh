#!/bin/bash
# Launch hoplas ring-task training on a remote machine via SSH.
#
# Usage:
#   ./scripts/launch.sh <host> <config_file> [gpu_id]
#   ./scripts/launch.sh <host> all [gpu_id]   # launches all configs/mnist_*.cfg
#
# Examples:
#   ./scripts/launch.sh lelio configs/mnist_ph_4_nonorm.cfg 2
#   ./scripts/launch.sh lelio all 2
#
# host must be configured in ~/.ssh/config.
# Remote repo path is set via REMOTE_REPO below (default: /mnt/media/scott/github/hoplas).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOST="${1:?Usage: $0 <host> <config_file|all> [gpu_id]}"
CONFIG_ARG="${2:?Usage: $0 <host> <config_file|all> [gpu_id]}"
GPU="${3:-0}"

SSH="ssh -o ClearAllForwardings=yes"
REMOTE_REPO="${HOPLAS_REMOTE_REPO:-~/github/hoplas}"
REMOTE_ENV="${HOPLAS_REMOTE_ENV:-~/envs/hoplas}"

# Sync source to remote (never overwrites checkpoints or datasets)
echo "Syncing source to ${HOST}:${REMOTE_REPO}..."
rsync -az --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
    --exclude='wandb' --exclude='checkpoints' --exclude='*.pt' \
    "${REPO_DIR}/hoplas/" "${HOST}:${REMOTE_REPO}/hoplas/"
rsync -az "${REPO_DIR}/train_ops.py" "${HOST}:${REMOTE_REPO}/"
rsync -az "${REPO_DIR}/configs/" "${HOST}:${REMOTE_REPO}/configs/"
$SSH "${HOST}" "mkdir -p ${REMOTE_REPO}/logs ${REMOTE_REPO}/checkpoints"

# Build list of configs
if [[ "$CONFIG_ARG" == "all" ]]; then
    CONFIGS=($(ls "${REPO_DIR}"/configs/mnist_*.cfg 2>/dev/null))
    [[ ${#CONFIGS[@]} -eq 0 ]] && { echo "No configs/mnist_*.cfg found."; exit 1; }
    echo "Found ${#CONFIGS[@]} configs to launch."
else
    CONFIGS=("${CONFIG_ARG}")
fi

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "${CONFIG}" .cfg)
    REMOTE_CONFIG="${REMOTE_REPO}/configs/$(basename "${CONFIG}")"
    # inject nd<N> (read from inside the config) into the log name so nd3 vs nd16 runs are distinguishable
    ND=$(grep -E '^[[:space:]]*nd[[:space:]]*=' "${CONFIG}" 2>/dev/null | head -1 | sed -E 's/.*=[[:space:]]*([0-9]+).*/\1/')
    LOG_NAME="${CONFIG_NAME}"
    [[ -n "$ND" ]] && LOG_NAME=$(echo "${CONFIG_NAME}" | sed -E "s/^(line_[a-z]+_)/\\1nd${ND}_/")
    LOG="${REMOTE_REPO}/logs/${LOG_NAME}.log"

    cat > /tmp/hoplas_run.sh << EOF
#!/bin/bash
source ${REMOTE_ENV}/bin/activate
cd ${REMOTE_REPO}
CUDA_VISIBLE_DEVICES=${GPU} nohup python train_ops.py --config ${REMOTE_CONFIG} \
    > ${LOG} 2>&1 &
echo \$!
EOF

    scp -q /tmp/hoplas_run.sh "${HOST}:/tmp/hoplas_run_${CONFIG_NAME}.sh"
    echo "Launching ${CONFIG_NAME} on ${HOST} (GPU=${GPU})..."
    PID=$($SSH "${HOST}" "bash /tmp/hoplas_run_${CONFIG_NAME}.sh")
    echo "  PID ${PID} → ${LOG}"
done

echo "Done."
