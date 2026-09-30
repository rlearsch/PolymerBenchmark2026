#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash Datasets/task4_architecture_transfer/train_models.sh <model> [options]

Models: rdkit_rf, polybert, chemprop, periodic_graph, or both_graph

Options:
  --split-root PATH  Shared Task 4 split root
  --epochs N         Graph-model epochs (default: 50)
  --seed N           Model seed (default: 42)
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

MODEL="$1"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SPLIT_ROOT="$SCRIPT_DIR/generated/shared_splits"
EPOCHS=50
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split-root) SPLIT_ROOT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -d "$SPLIT_ROOT" ]]; then
  echo "ERROR: Task 4 split root not found: $SPLIT_ROOT" >&2
  exit 1
fi

case "$MODEL" in
  rdkit_rf)
    exec "$REPO_ROOT/Models/RDKit_RF/.venv/bin/python" "$REPO_ROOT/Models/RDKit_RF/train_rf_scaling.py" \
      --split-root "$SPLIT_ROOT" \
      --save-root "$REPO_ROOT/Models/RDKit_RF/task4-architecture-transfer" \
      --seed "$SEED"
    ;;
  polybert)
    exec "$REPO_ROOT/Models/polyBERT/.venv/bin/python" "$REPO_ROOT/Models/polyBERT/train_pBERT_scaling.py" \
      --split-root "$SPLIT_ROOT" \
      --save-root "$REPO_ROOT/Models/polyBERT/task4-architecture-transfer" \
      --seed "$SEED"
    ;;
  chemprop|periodic_graph)
    graph_model="polymer_chemprop"
    [[ "$MODEL" == "periodic_graph" ]] && graph_model="polymer_periodic_graph"
    exec bash "$REPO_ROOT/scripts/train_graph_scaling.sh" "$graph_model" \
      --split-root "$SPLIT_ROOT" --result-dir task4-architecture-transfer \
      --epochs "$EPOCHS" --seed "$SEED"
    ;;
  both_graph)
    exec bash "$REPO_ROOT/scripts/train_graph_scaling.sh" both \
      --split-root "$SPLIT_ROOT" --result-dir task4-architecture-transfer \
      --epochs "$EPOCHS" --seed "$SEED"
    ;;
  *)
    echo "Unknown model: $MODEL" >&2
    usage >&2
    exit 1
    ;;
esac
