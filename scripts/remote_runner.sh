#!/bin/bash
# Self-refilling training queue that runs ON the GPU host (nohup'd), so it survives the
# control laptop sleeping/disconnecting. Keeps up to PAR train_kge/train_ops/train_vae jobs
# running, counting ALL training processes on the box (a true global cap), refilling from a
# queue as slots free. Exits once the queue is exhausted (launched jobs keep running).
#
# Invoked by scripts/remote_queue.sh -- not meant to be called directly.
# Args: PAR GPU POLL REPO_ARG ENV_ARG <items...>
# Each item is a config basename (-> train_ops/train_kge) or "vae:<dataset>" (-> train_vae.py).

PAR="$1"; GPU="$2"; POLL="$3"
case "$4" in /*) REPO="$4";; *) REPO="$HOME/$4";; esac
case "$5" in /*) ENV="$5";;  *) ENV="$HOME/$5";; esac
shift 5
# Remaining args: queue items, then a literal "--", then pass-through flags for every job.
QUEUE=(); EXTRA=(); seen_dashdash=0
for a in "$@"; do
    if [[ $seen_dashdash -eq 0 && "$a" == "--" ]]; then seen_dashdash=1; continue; fi
    if [[ $seen_dashdash -eq 1 ]]; then EXTRA+=("$a"); else QUEUE+=("$a"); fi
done

source "$ENV/bin/activate"
cd "$REPO" || exit 1

n_running()   {
    # Count main training procs only, excluding forked DataLoader workers: a worker's parent is
    # another train proc, a real job's parent is the shell/nohup. (train_vae.py uses num_workers>0,
    # so a plain `ps|grep -c` counts ~5 procs per job and would blow past --par.)
    ps -eo pid,ppid,args | awk '
        /[t]rain_(kge|ops|vae|classifier)\.py/ { pid[$1]=1; par[$1]=$2 }
        END { for (p in pid) if (!(par[p] in pid)) c++; print c+0 }'
}
is_running()  {
    case "$1" in
        vae:*) ps -eo args | grep -E '[t]rain_vae\.py' | grep -qF -- "--dataset ${1#vae:}" ;;
        clf:*) ps -eo args | grep -E '[t]rain_classifier\.py' | grep -qF -- "--dataset ${1#clf:}" ;;
        *)     ps -eo args | grep -E '[t]rain_(kge|ops)\.py' | grep -qF "/$1.cfg" ;;
    esac
}
done_already(){ local L="$REPO/logs/$1.log"; [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -qE '^TEST '; }

launch() {
    local item="$1"
    if [[ "$item" == vae:* ]]; then
        # vae:<dataset> or vae:<dataset>:<tag>; the tag only distinguishes the log name so
        # multiple configs of one dataset can run concurrently without clobbering each other.
        local rest="${item#vae:}" ds tag logname
        ds="${rest%%:*}"; tag="${rest#*:}"; [[ "$tag" == "$rest" ]] && tag=""
        logname="vae_$ds"; [[ -n "$tag" ]] && logname="vae_${ds}_$tag"
        CUDA_VISIBLE_DEVICES="$GPU" nohup python scripts/train_vae.py --dataset "$ds" "${EXTRA[@]}" \
            > "$REPO/logs/$logname.log" 2>&1 &
        echo "[$(date '+%F %T')] launched vae:$rest (train_vae.py) pid $! -> logs/$logname.log"
        return
    fi
    if [[ "$item" == clf:* ]]; then
        local ds="${item#clf:}"
        CUDA_VISIBLE_DEVICES="$GPU" nohup python scripts/train_classifier.py --dataset "$ds" "${EXTRA[@]}" \
            > "$REPO/logs/clf_$ds.log" 2>&1 &
        echo "[$(date '+%F %T')] launched clf:$ds (train_classifier.py) pid $! -> logs/clf_$ds.log"
        return
    fi
    if [[ "$item" == recon:* ]]; then
        local ds="${item#recon:}"
        CUDA_VISIBLE_DEVICES="$GPU" nohup python scripts/score_recon.py --dataset "$ds" "${EXTRA[@]}" \
            > "$REPO/logs/recon_$ds.log" 2>&1 &
        echo "[$(date '+%F %T')] launched recon:$ds (score_recon.py) pid $! -> logs/recon_$ds.log"
        return
    fi
    local cfg="$item" train logname="$item"
    [[ "$cfg" == kge_* ]] && train=train_kge.py || train=train_ops.py
    # nd-logname: line_* configs without _nd in the name get nd<N> (read from the config) injected,
    # so 4-dim vs 8-dim vs 16-dim runs are distinguishable in the logs.
    if [[ "$cfg" == line_* && "$cfg" != *_nd* ]]; then
        local nd; nd=$(grep -E '^[[:space:]]*nd[[:space:]]*=' "configs/$cfg.cfg" 2>/dev/null | head -1 | sed -E 's/.*=[[:space:]]*([0-9]+).*/\1/')
        [[ -n "$nd" ]] && logname=$(echo "$cfg" | sed -E "s/^(line_[a-z]+_)/\\1nd${nd}_/")
    fi
    CUDA_VISIBLE_DEVICES="$GPU" nohup python "$train" --config "configs/$cfg.cfg" "${EXTRA[@]}" \
        > "$REPO/logs/$logname.log" 2>&1 &
    echo "[$(date '+%F %T')] launched $cfg ($train) pid $! -> logs/$logname.log"
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
