#!/bin/bash
# Ring/dihedral (train_ops.py) results from W&B with the OOD canaries:
# sec_sim (reflect fit), sim, recon, and var(xproj_t)/var(yproj) spread ratio.
# Flags runs where a good sec_sim hides a bad recon or over/under-spread.
# Thin wrapper around scripts/ops_results.py. See that file for options.
#
#   bash scripts/ops_results.sh                                  # dihedral-mnist, by sec_sim
#   bash scripts/ops_results.sh --name-contains _neg --md
#   bash scripts/ops_results.sh --project ring-mnist --sort recon
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/ops_results.py" "$@"
