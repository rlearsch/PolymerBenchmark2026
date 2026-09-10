#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash generate_scaling_datasets.sh [--overwrite] [--dry-run]

Generate the exact, representation-aligned datasets used by the PolyBench26
dataset-size scaling experiments. Generated CSVs and manifests are written to
Datasets/scaling_splits/ and are intentionally excluded from Git.

Prerequisites:
  1. bash generate_basic_datasets.sh
  2. bash generate_polybert_datasets.sh

Options:
  --overwrite  Replace existing split files.
  --dry-run    Print the generation commands without running them.
  -h, --help   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: dataset environment not found: $PYTHON" >&2
  echo "Run: bash $SCRIPT_DIR/setup_environments.sh" >&2
  exit 1
fi

GENERATOR="$SCRIPT_DIR/create_scaling_splits.py"
COMMON_ARGS=(--num-folds 5 --seed 42)
if [[ "$OVERWRITE" -eq 1 ]]; then
  COMMON_ARGS+=(--overwrite)
fi

run_experiment() {
  local dataset="$1"
  local property="$2"
  local filename="$3"
  shift 3
  local command=(
    "$PYTHON" "$GENERATOR"
    --dataset "$dataset"
    --property "$property"
    --file "$filename"
    --train-sizes "$@"
    "${COMMON_ARGS[@]}"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    echo "Generating $dataset/$property/$filename (train sizes: $*)"
    "${command[@]}"
  fi
}

# MD_5000 alternating copolymers: enough rows for the full scaling curve.
for property in Cp density refractive_index Rg; do
  run_experiment MD_5000 "$property" "alternating_${property}.csv" 300 1000 3000 10000
done

# MD_5000 homopolymers: approximately 1,700-1,800 total rows.
for property in Cp Cv density refractive_index Rg; do
  run_experiment MD_5000 "$property" "homopolymer_${property}.csv" 300 1000
done

# Experimental glass-transition-temperature data.
run_experiment PolyMetriX Tg Tg.csv 300 1000 3000

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Scaling datasets written under $REPO_ROOT/Datasets/scaling_splits"
fi
