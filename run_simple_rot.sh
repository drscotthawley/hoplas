#!/usr/bin/env bash
# Sweep all methods (and PH orders) over a set of corr_nd values, log each to
# wandb, then print a final-loss table (rows=method, cols=corr_nd).
# Sample usage: 
# Run: ./run_simple_rot.sh (corr=1.0) or ./run_simple_rot.sh 0.5 to override strength.

set -euo pipefail

cd "$(dirname "$0")"

# correlation strength (default 0.9); extra args forwarded to the script
CORR="${1:-0.9}"
[[ $# -gt 0 ]] && shift

CORR_NDS=(1 2 4 8 16)

# method specs: "method" or "ph:ORDER"
RUNS=(
    "filmr"
    "filmr_expm"
    "matop"
    "matop2"
    "ph:2"
    "ph:4"
    "ph:8"
    "ph:16"
)

LOGDIR="$(mktemp -d)"
declare -a LABELS
declare -A LOSSES   # key "label|corr_nd" -> final loss

for cnd in "${CORR_NDS[@]}"; do
    for spec in "${RUNS[@]}"; do
        if [[ "$spec" == ph:* ]]; then
            label="ph_${spec##*:}"
            args=(--method ph --order "${spec##*:}")
        else
            label="$spec"
            args=(--method "$spec")
        fi
        [[ " ${LABELS[*]:-} " == *" $label "* ]] || LABELS+=("$label")

        echo "=== running $label (corr=$CORR corr_nd=$cnd) ==="
        logfile="$LOGDIR/${label}_${cnd}.log"
        ./train_simple_rot.py "${args[@]}" --corr "$CORR" --corr-nd "$cnd" "$@" 2>&1 | tee "$logfile"

        # final loss = last "loss=..." value printed
        LOSSES["$label|$cnd"]="$(grep -oE 'loss=[0-9.eE+-]+' "$logfile" | tail -1 | cut -d= -f2)"
    done
done

echo
echo "==========  final loss (corr=$CORR), cols=corr_nd  =========="
printf "%-14s" "run"
for cnd in "${CORR_NDS[@]}"; do printf "%12s" "nd=$cnd"; done
printf "\n"
for label in "${LABELS[@]}"; do
    printf "%-14s" "$label"
    for cnd in "${CORR_NDS[@]}"; do printf "%12s" "${LOSSES[$label|$cnd]:-—}"; done
    printf "\n"
done
