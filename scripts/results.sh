#!/bin/bash
# Harvest val / TEST ranking metrics from hoplas KGE run logs into a comparison table.
# Best val = the eval line with the highest val MRR; TEST = the final test-split line.
#
# Usage:
#   ./scripts/results.sh <host> [config_glob]   # glob (no .log) defaults to 'kge_*'
#
# Env:
#   HOPLAS_REMOTE_REPO   repo path on host (relative to $HOME, or absolute); default github/hoplas

HOST="${1:?Usage: $0 <host> [config_glob]}"
GLOB="${2:-kge_*}"
REPO_ARG="${HOPLAS_REMOTE_REPO:-github/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

$SSH "${HOST}" bash -s -- "$GLOB" "$REPO_ARG" << 'ENDSSH'
GLOB="$1"
case "$2" in /*) REPO="$2";; *) REPO="$HOME/$2";; esac
LOGS_DIR="$REPO/logs"

printf '%-36s %8s %7s %7s | %7s %7s %7s %7s %7s\n' \
       RUN vMRR vH1 vH10 tMRR tMR tH1 tH3 tH10
printf '%.0s-' {1..96}; echo

shopt -s nullglob
for L in "$LOGS_DIR"/$GLOB.log; do
    name=$(basename "$L" .log)
    clean=$(tr '\r' '\n' < "$L")

    bv=$(echo "$clean" | grep -oE 'val MRR=[0-9.]+ H@10=[0-9.]+ H@1=[0-9.]+' \
         | sort -t= -k2 -gr | head -1)
    vmrr=$(echo "$bv" | sed -nE 's/.*val MRR=([0-9.]+).*/\1/p')
    vh10=$(echo "$bv" | sed -nE 's/.*H@10=([0-9.]+).*/\1/p')
    vh1=$(echo "$bv"  | sed -nE 's/.* H@1=([0-9.]+).*/\1/p')

    t=$(echo "$clean" | grep -E '^TEST ' | tail -1)
    tmrr=$(echo "$t" | sed -nE 's/.*MRR=([0-9.]+).*/\1/p')
    tmr=$(echo "$t"  | sed -nE 's/.*MR=([0-9.]+).*/\1/p')
    th1=$(echo "$t"  | sed -nE 's/.*H@1=([0-9.]+).*/\1/p')
    th3=$(echo "$t"  | sed -nE 's/.*H@3=([0-9.]+).*/\1/p')
    th10=$(echo "$t" | sed -nE 's/.*H@10=([0-9.]+).*/\1/p')

    printf '%-36s %8s %7s %7s | %7s %7s %7s %7s %7s\n' \
        "$name" "${vmrr:--}" "${vh1:--}" "${vh10:--}" \
        "${tmrr:--}" "${tmr:--}" "${th1:--}" "${th3:--}" "${th10:--}"
done
ENDSSH
