#!/bin/bash
# Diagnose and (re)download a pykeen dataset whose cached archive is corrupt.
# Shows the offending cached file (type + first bytes -> HTML interstitial vs truncated
# binary), then (action=fix) removes that dataset's pykeen cache and re-fetches it.
# No GPU. Safe to run alongside training (CPU/network only).
#
# Usage:
#   ./scripts/redownload.sh <host> <dataset> [diagnose|fix]   # default: fix
#   e.g. ./scripts/redownload.sh lecun WN18

HOST="${1:?Usage: $0 <host> <dataset> [diagnose|fix]}"
DS="${2:?Usage: $0 <host> <dataset> [diagnose|fix]}"
ACTION="${3:-fix}"
ENV_ARG="${HOPLAS_REMOTE_ENV:-envs/hoplas}"
SSH="ssh -o ClearAllForwardings=yes"

$SSH "${HOST}" bash -s -- "$DS" "$ACTION" "$ENV_ARG" "${URL:-}" << 'ENDSSH'
DS="$1"; ACTION="$2"; URL="$4"
case "$3" in /*) ENV="$3";; *) ENV="$HOME/$3";; esac
source "$ENV/bin/activate"

PK=$(python -c 'import pystow; print(pystow.join("pykeen","datasets"))' 2>/dev/null)
LC=$(echo "$DS" | tr 'A-Z' 'a-z')
DSDIR="$PK/$LC"
echo "pykeen datasets dir: $PK"
echo "--- dataset dir: $DSDIR ---"
ls -laR "$DSDIR" 2>/dev/null | head -40
echo "--- inspecting cached files (type + first 160 bytes) ---"
find "$DSDIR" -type f 2>/dev/null | while read -r f; do
    echo ">> $f"; file "$f"; head -c 160 "$f" | tr -d '\0'; echo; echo "----"
done

if [ "$ACTION" = "probe" ]; then
    echo "=== pykeen source URL(s) for $DS (grep installed package) ==="
    PKGDIR=$(python -c 'import pykeen, os; print(os.path.dirname(pykeen.__file__))')
    grep -rInE "https?://[^\"' ]+" "$PKGDIR/datasets/" 2>/dev/null \
        | grep -iE "$DS|freebase|wordnet|fb15|wn18|drive.google|dropbox|github" | head -20
    echo "--- resolving the class's url attribute ---"
    python - "$DS" << 'PY'
import sys
from pykeen.datasets import dataset_resolver
ds = sys.argv[1]
cls = dataset_resolver.lookup(ds)
print("class:", cls)
for attr in ("url", "URL"):
    print(attr, "=", getattr(cls, attr, None))
import inspect
src = inspect.getsource(cls)
for line in src.splitlines():
    if "http" in line.lower() or "url" in line.lower():
        print("  ", line.strip()[:160])
PY
    echo "--- curl headers + first bytes of the url (set URL env to test a specific one) ---"
    if [ -n "$URL" ]; then
        echo "probing: $URL"
        curl -sSIL --max-time 25 "$URL" 2>&1 | head -20
        echo "[curl exit: $?]"
    fi
fi

if [ "$ACTION" = "fix" ]; then
    if [ -n "$PK" ] && [ -d "$DSDIR" ]; then
        echo "=== removing $DSDIR ==="
        rm -rf "$DSDIR"
    fi
    echo "=== re-fetching $DS via pykeen.get_dataset ==="
    python -c "from pykeen.datasets import get_dataset; t=get_dataset(dataset='$DS').training; print('REFETCH OK:', '$DS', 'entities=', t.num_entities, 'relations=', t.num_relations)"
fi
ENDSSH
