#!/usr/bin/env bash
# Run all rotation-learning methods (and PH orders), log each to wandb,
# then print a summary table of final losses.
set -euo pipefail

cd "$(dirname "$0")"

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
declare -a LOSSES

for spec in "${RUNS[@]}"; do
    method="${spec%%:*}"
    if [[ "$spec" == ph:* ]]; then
        order="${spec##*:}"
        label="ph_${order}"
        args=(--method ph --order "$order")
    else
        label="$method"
        args=(--method "$method")
    fi

    echo "=== running $label ==="
    logfile="$LOGDIR/$label.log"
    ./train_simple_rot.py "${args[@]}" "$@" 2>&1 | tee "$logfile"

    # final loss = last "loss=..." value printed
    final_loss="$(grep -oE 'loss=[0-9.eE+-]+' "$logfile" | tail -1 | cut -d= -f2)"
    LABELS+=("$label")
    LOSSES+=("$final_loss")
done

echo
echo "==================  final loss summary  =================="
printf "%-14s %s\n" "run" "final_loss"
printf -- "---------------------------------\n"
for i in "${!LABELS[@]}"; do
    printf "%-14s %s\n" "${LABELS[$i]}" "${LOSSES[$i]}"
done
