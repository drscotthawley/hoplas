#!/bin/bash
# Generate the ~50-cell ring-task sweep: distinct operators x dimension x {norm-primary,
# nonorm-subset}, controlled (noise=0, npoints=12, target=ring, op-resid=false, 1000 epochs).
# Only divisibility-valid (op,nd) cells. Files: configs/ring_<op[_order]>_nd<N>_<norm>.cfg
# (ring_ prefix -> train_ops.py). Deletes any existing ring_*.cfg first (all are generated).
#
# Usage: ./scripts/gen_ring_configs.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_DIR="$(dirname "$SCRIPT_DIR")/configs"
PROJECT="hoplas-ring"
EPOCHS=500   # ring runs converge well before this; 500 is ample (was 1000)
NORM_DIMS="2 4 8 16 64 256"      # full norm grid
NONORM_DIMS="4 64"               # nonorm only on a representative subset

emit() {  # emit <op> <order|""> <nd> <unit_norm:true|false>
    local op="$1" order="$2" nd="$3" unitnorm="$4"
    local normtag; [ "$unitnorm" = "true" ] && normtag="norm" || normtag="nonorm"
    local opname="$op"; [ -n "$order" ] && opname="${op}_${order}"
    local f="$CFG_DIR/ring_${opname}_nd${nd}_${normtag}.cfg"
    {
        echo "dataset = line"
        echo "target = ring"
        echo "noise = 0.0"
        echo "npoints = 12"
        echo "nd = $nd"
        echo "batch-size = 8192"
        echo "op = $op"
        [ -n "$order" ] && echo "order = $order"
        echo "op-resid = false"
        echo "unit-norm = $unitnorm"
        echo "epochs = $EPOCHS"
        echo "wandb-project = $PROJECT"
        echo "tag = $normtag"
    } > "$f"
}

gen_for_dims() {  # gen_for_dims <unit_norm> <dims...>
    local unitnorm="$1"; shift
    for nd in "$@"; do
        emit filmr_expm "" "$nd" "$unitnorm"                       # rotation via matrix-exp
        emit matop "" "$nd" "$unitnorm"                            # unstructured control
        for k in 2 4 8 16; do                                      # PH order axis
            [ $((nd % k)) -eq 0 ] && emit ph "$k" "$nd" "$unitnorm"
        done
        [ $((nd % 4)) -eq 0 ] && emit quat 4 "$nd" "$unitnorm"     # frozen Hamilton quaternion
        [ $((nd % 8)) -eq 0 ] && emit kdualquat "" "$nd" "$unitnorm"  # dual quaternion
    done
}

rm -f "$CFG_DIR"/ring_*.cfg
gen_for_dims true  $NORM_DIMS
gen_for_dims false $NONORM_DIMS
echo "total ring_ configs: $(ls "$CFG_DIR"/ring_*.cfg | wc -l)"
echo "norm:   $(ls "$CFG_DIR"/ring_*_norm.cfg 2>/dev/null | wc -l)   nonorm: $(ls "$CFG_DIR"/ring_*_nonorm.cfg 2>/dev/null | wc -l)"
