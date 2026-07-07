#!/bin/bash
# Syntax-check (byte-compile) Python files without running them -- catches syntax /
# indentation errors after an edit, without importing torch or executing anything.
#
# Usage:
#   bash scripts/compile_test.sh                       # check all top-level *.py + hoplas/*.py
#   bash scripts/compile_test.sh train_kge.py eval_kge.py   # check only the given files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [[ $# -gt 0 ]]; then
    FILES=()
    for f in "$@"; do
        case "$f" in
            /*) FILES+=("$f") ;;             # absolute path: use as-is
            *)  FILES+=("$REPO_DIR/$f") ;;   # relative: resolve against repo root, not CWD
        esac
    done
else
    FILES=("$REPO_DIR"/*.py "$REPO_DIR"/hoplas/*.py "$SCRIPT_DIR"/*.py)
fi

python -m py_compile "${FILES[@]}" && echo "OK: py_compile passed (${#FILES[@]} files)"
