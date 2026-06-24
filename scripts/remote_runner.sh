#!/bin/bash
# Self-refilling training queue that runs ON the GPU host (nohup'd), so it survives the
# control laptop sleeping/disconnecting. Keeps up to PAR train_kge/train_ops jobs running,
# counting ALL training processes on the box (a true global cap), refilling from a config
# queue as slots free. Exits once the queue is exhausted (launched jobs keep running).
#
# Invoked by scripts/remote_queue.sh -- not meant to be called directly.
# Args: PAR GPU POLL REPO_ARG ENV_ARG <config_basenames...>

PAR="$1"; GPU="$2"; POLL="$3"
case "$4" in /*) REPO="$4";; *) REPO="$HOME/$4";; esac
case "$5" in /*) ENV="$5";;  *) ENV="$HOME/$5";; esac
shift 5
QUEUE=("$@")

source "$ENV/bin/activate"
cd "$REPO" || exit 1

n_running()   { ps -eo args | grep -cE '[t]rain_(kge|ops)\.py'; }
is_running()  { ps -eo args | grep -E '[t]rain_(kge|ops)\.py' | grep -qF "/$1.cfg"; }
done_already(){ local L="$REPO/logs/$1.log"; [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -qE '^TEST '; }

launch() {
    local cfg="$1" train
    [[ "$cfg" == kge_* ]] && train=train_kge.py || train=train_ops.py
    CUDA_VISIBLE_DEVICES="$GPU" nohup python "$train" --config "configs/$cfg.cfg" \
        > "$REPO/logs/$cfg.log" 2>&1 &
    echo "[$(date '+%F %T')] launched $cfg ($train) pid $!"
}

echo "[$(date '+%F %T')] remote_runner start: par=$PAR gpu=$GPU poll=$POLL queue=${QUEUE[*]}"
idx=0
while :; do
    while [ "$(n_running)" -lt "$PAR" ] && [ "$idx" -lt "${#QUEUE[@]}" ]; do
        cfg="${QUEUE[$idx]}"; idx=$((idx + 1))
        if is_running "$cfg";   then echo "[$(date '+%F %T')] skip $cfg (already running)"; continue; fi
        if done_already "$cfg"; then echo "[$(date '+%F %T')] skip $cfg (already has TEST)"; continue; fi
        launch "$cfg"
        sleep 8
    done
    if [ "$idx" -ge "${#QUEUE[@]}" ]; then
        echo "[$(date '+%F %T')] queue exhausted ($(n_running) still running); runner exiting"
        break
    fi
    sleep "$POLL"
done
