#!/bin/bash
# Download an OpenKE-format dataset (train2id/valid2id/test2id/entity2id/relation2id) into
# the host's local data dir (<repo>/data/<dataset>), for datasets whose pykeen download is
# broken (WN18, FB15K). Default source: the QuatE paper's own benchmarks (apples-to-apples).
# KGTripleDataset auto-loads from <repo>/data/<dataset> when present (HOPLAS_DATA).
#
# Usage:
#   ./scripts/fetch_openke.sh <host> <dataset> [src_base_url]
#   e.g. ./scripts/fetch_openke.sh lecun WN18
#        ./scripts/fetch_openke.sh lecun FB15K

HOST="${1:?Usage: $0 <host> <dataset> [src_base_url]}"
DS="${2:?Usage: $0 <host> <dataset> [src_base_url]}"
BASE="${3:-https://raw.githubusercontent.com/cheungdaven/QuatE/master/benchmarks}"
REPO_ARG="${HOPLAS_REMOTE_REPO:-github/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

$SSH "${HOST}" bash -s -- "$DS" "$BASE" "$REPO_ARG" << 'ENDSSH'
DS="$1"; BASE="$2"
case "$3" in /*) REPO="$3";; *) REPO="$HOME/$3";; esac
DEST="$REPO/data/$DS"
mkdir -p "$DEST"
ok=1
for f in train2id.txt valid2id.txt test2id.txt entity2id.txt relation2id.txt; do
    url="$BASE/$DS/$f"
    echo "fetching $url"
    if curl -sSL --fail --max-time 180 -o "$DEST/$f" "$url"; then
        echo "  -> $(wc -l < "$DEST/$f") lines; line1: $(head -1 "$DEST/$f")"
    else
        echo "  FAILED: $f"; ok=0
    fi
done
[ "$ok" = 1 ] && echo "OK fetched $DS -> $DEST" || echo "INCOMPLETE $DS"
ls -la "$DEST"
ENDSSH
