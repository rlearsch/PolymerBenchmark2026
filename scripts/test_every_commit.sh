#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -x "$REPO_ROOT/Datasets/Dataset_construction_scripts/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/Datasets/Dataset_construction_scripts/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "ERROR: Python 3 was not found." >&2
  exit 1
fi

cd "$REPO_ROOT"
STATUS_BEFORE="$(git status --porcelain=v1 --untracked-files=all)"
shopt -s nullglob
bash -n Datasets/Dataset_construction_scripts/*.sh scripts/*.sh Models/*/*.sh
PYTHON_SOURCES=(
  Datasets/Dataset_construction_scripts/*.py
  Models/RDKit_RF/*.py
  Models/polyBERT/*.py
  scripts/*.py
  tests/*.py
)
"$PYTHON" -m compileall -q "${PYTHON_SOURCES[@]}"
"$PYTHON" -m unittest discover -s tests -v
git diff --check
STATUS_AFTER="$(git status --porcelain=v1 --untracked-files=all)"
if [[ "$STATUS_BEFORE" != "$STATUS_AFTER" ]]; then
  echo "ERROR: tests changed the working tree." >&2
  diff <(printf '%s\n' "$STATUS_BEFORE") <(printf '%s\n' "$STATUS_AFTER") || true
  exit 1
fi
