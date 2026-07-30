#!/bin/bash
# nd=16 ring NOISE-LEVEL sweep for the fan-out figure. "Noise" here is really class spread
# (variance) -- a data choice, not signal noise. Sweep it across {0.0, 0.01, 0.03, 0.05} for
# all operators x {norm, nonorm}, everything else fixed. Story: at 0.0 (single points) the
# operators should converge; as spread grows they fan apart (filmr/matop fail, PH holds).
# 0.01 is the visual sweet spot (clusters distinct, no bleed); 0.1 mushes them -- do NOT use.
# One project (hoplas-ring-noise); tag + filename encode the level (n00/n01/n03/n05) so runs
# are distinct. Files: configs/ringnoise<NN>_<op[_order]>_nd16_<norm>.cfg (-> train_ops.py).
#
# Usage: ./scripts/gen_ring_noise_configs.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_DIR="$(dirname "$SCRIPT_DIR")/configs"
PROJECT="hoplas-ring-noise"
EPOCHS=500
ND=16
LEVELS="0.0:00 0.01:01 0.03:03 0.05:05"   # <noise>:<tag/filename suffix>

emit() {  # emit <op> <order|""> <unit_norm> <noise> <nn>
    local op="$1" order="$2" unitnorm="$3" noise="$4" nn="$5"
    local normtag; [ "$unitnorm" = "true" ] && normtag="norm" || normtag="nonorm"
    local opname="$op"; [ -n "$order" ] && opname="${op}_${order}"
    local f="$CFG_DIR/ringnoise${nn}_${opname}_nd${ND}_${normtag}.cfg"
    {
        echo "dataset = line"
        echo "target = ring"
        echo "noise = $noise"
        echo "npoints = 12"
        echo "nd = $ND"
        echo "batch-size = 8192"
        echo "op = $op"
        [ -n "$order" ] && echo "order = $order"
        echo "op-resid = false"
        echo "unit-norm = $unitnorm"
        echo "epochs = $EPOCHS"
        echo "wandb-project = $PROJECT"
        echo "tag = n${nn}_${normtag}"
    } > "$f"
}

gen_ops() {  # gen_ops <unit_norm> <noise> <nn>
    local un="$1" noise="$2" nn="$3"
    emit filmr_expm "" "$un" "$noise" "$nn"
    emit matop      "" "$un" "$noise" "$nn"
    for k in 2 4 8 16; do emit ph "$k" "$un" "$noise" "$nn"; done
    emit quat 4 "$un" "$noise" "$nn"
    emit kdualquat "" "$un" "$noise" "$nn"
}

rm -f "$CFG_DIR"/ringnoise[0-9][0-9]_*.cfg
for pair in $LEVELS; do
    noise="${pair%%:*}"; nn="${pair##*:}"
    gen_ops true  "$noise" "$nn"
    gen_ops false "$noise" "$nn"
done
echo "total noise-sweep configs: $(ls "$CFG_DIR"/ringnoise[0-9][0-9]_*.cfg | wc -l)"
