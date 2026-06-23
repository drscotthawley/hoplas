#!/bin/bash
# Run eval_kge.py against a checkpoint on a remote host.
#
# Usage:
#   ./scripts/launch_eval_kge.sh <host> <checkpoint_filename> [extra_args...]
#
# checkpoint_filename: just the filename (no path), looked up in
#   <remote_repo>/checkpoints/, e.g.:
#   WN18RR_ph_2_nd512_lambd0.1_champ_s1_best.pt
#
# Extra args are passed directly to eval_kge.py, e.g.:
#   --score cos --max-k 3 --n-queries 2000 --tests all
#
# Env:
#   HOPLAS_REMOTE_REPO   repo path on host (relative to $HOME, or absolute); default github/hoplas
#   HOPLAS_REMOTE_ENV    venv path on host  (relative to $HOME, or absolute); default envs/hoplas

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOST="${1:?Usage: $0 <host> <checkpoint_filename> [extra_args...]}"
CKPT_NAME="${2:?Usage: $0 <host> <checkpoint_filename> [extra_args...]}"
shift 2

REPO_ARG="${HOPLAS_REMOTE_REPO:-github/hoplas}"
ENV_ARG="${HOPLAS_REMOTE_ENV:-envs/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

echo "Syncing source to ${HOST}:${REPO_ARG} ..."
rsync -az --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
    --exclude='wandb' --exclude='checkpoints' --exclude='*.pt' \
    "${REPO_DIR}/hoplas/" "${HOST}:${REPO_ARG}/hoplas/"
rsync -az "${REPO_DIR}/train_kge.py" "${REPO_DIR}/eval_kge.py" "${HOST}:${REPO_ARG}/"

echo "Running eval_kge.py on ${HOST}: checkpoints/${CKPT_NAME} $*"

# Write a temp runner (variables expand locally into the script body;
# no arg-passing across the SSH boundary, so quoting is unambiguous).
cat > /tmp/hoplas_eval_run.sh << EOF
#!/bin/bash
case "${REPO_ARG}" in /*) REPO="${REPO_ARG}";; *) REPO="\$HOME/${REPO_ARG}";; esac
case "${ENV_ARG}" in /*) ENV="${ENV_ARG}";; *) ENV="\$HOME/${ENV_ARG}";; esac
source "\$ENV/bin/activate"
cd "\$REPO"
python eval_kge.py "checkpoints/${CKPT_NAME}" $*
EOF

scp -q /tmp/hoplas_eval_run.sh "${HOST}:/tmp/hoplas_eval_run.sh"
$SSH "${HOST}" "bash /tmp/hoplas_eval_run.sh"
