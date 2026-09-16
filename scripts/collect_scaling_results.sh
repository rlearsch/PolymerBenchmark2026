#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/collect_scaling_results.sh [options]

Options:
  --split-root PATH        Scaling split root (default: Datasets/scaling_splits)
  --out-root PATH          Summary output root (default: scaling_results)
  --seed-name NAME         Seed directory to collect (default: seed_42)
  --only PATTERN           Only summarize experiments whose relative path contains PATTERN
  -h, --help               Show this help

This collects completed outputs from:
  Models/polyBERT/scaling_results
  Models/RDKit_RF/scaling_results
  Models/polymer_chemprop/scaling_results
  Models/polymer_periodic_graph/scaling_results

It writes per-experiment summaries to:
  scaling_results/<dataset>/<property>/<file>/<seed_name>/

And one combined summary to:
  scaling_results/all_<seed_name>/
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPLIT_ROOT="$REPO_ROOT/Datasets/scaling_splits"
OUT_ROOT="$REPO_ROOT/scaling_results"
SEED_NAME="seed_42"
ONLY_PATTERN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split-root)
      SPLIT_ROOT="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --seed-name)
      SEED_NAME="$2"
      shift 2
      ;;
    --only)
      ONLY_PATTERN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$SPLIT_ROOT" ]]; then
  echo "ERROR: split root not found: $SPLIT_ROOT" >&2
  exit 1
fi

if [[ -x "$REPO_ROOT/Models/RDKit_RF/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/Models/RDKit_RF/.venv/bin/python"
elif [[ -x "$REPO_ROOT/Datasets/Dataset_construction_scripts/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/Datasets/Dataset_construction_scripts/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "ERROR: no Python interpreter found." >&2
  exit 1
fi

MODEL_RESULT_ROOTS=(
  "$REPO_ROOT/Models/polyBERT/scaling_results"
  "$REPO_ROOT/Models/RDKit_RF/scaling_results"
  "$REPO_ROOT/Models/polymer_chemprop/scaling_results"
  "$REPO_ROOT/Models/polymer_periodic_graph/scaling_results"
)

all_roots=()
summarized=0

while IFS= read -r -d '' manifest; do
  rel="${manifest#"$SPLIT_ROOT/"}"
  rel="${rel%/manifest.json}"

  if [[ "$rel" != *"/$SEED_NAME" ]]; then
    continue
  fi
  if [[ -n "$ONLY_PATTERN" && "$rel" != *"$ONLY_PATTERN"* ]]; then
    continue
  fi

  roots=()
  for model_root in "${MODEL_RESULT_ROOTS[@]}"; do
    candidate="$model_root/$rel"
    if [[ -d "$candidate" ]]; then
      roots+=("$candidate")
      all_roots+=("$candidate")
    fi
  done

  if [[ "${#roots[@]}" -eq 0 ]]; then
    echo "Skipping $rel: no completed model result directories found."
    continue
  fi

  echo "Summarizing $rel"
  "$PYTHON" "$REPO_ROOT/scripts/summarize_scaling_results.py" \
    --result-roots "${roots[@]}" \
    --out-dir "$OUT_ROOT/$rel"
  summarized=$((summarized + 1))
done < <(find "$SPLIT_ROOT" -name manifest.json -print0 | sort -z)

if [[ -n "$ONLY_PATTERN" ]]; then
  echo "Filtered collection requested; not overwriting $OUT_ROOT/all_$SEED_NAME."
elif [[ "${#all_roots[@]}" -gt 0 ]]; then
  echo "Summarizing all completed experiments for $SEED_NAME"
  "$PYTHON" "$REPO_ROOT/scripts/summarize_scaling_results.py" \
    --result-roots "${all_roots[@]}" \
    --out-dir "$OUT_ROOT/all_$SEED_NAME"
fi

echo "Summarized $summarized experiment(s)."
