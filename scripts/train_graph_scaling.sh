#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_graph_scaling.sh <model> [options]

Models:
  polymer_chemprop         Train polymer-chemprop on wPSMILES scaling splits
  polymer_periodic_graph   Train polymer-periodic-graph on PSMILES scaling splits
  both                     Run both models concurrently in background

Options:
  --split-root PATH        Scaling split root (default: Datasets/scaling_splits)
  --epochs N               Epochs per run (default: 50)
  --seed N                 Chemprop seed and PyTorch seed (default: 0)
  --only PATTERN           Only run split roots whose relative path contains PATTERN
  --dry-run                Print commands without running them
  --no-quiet               Do not pass --quiet to trainers
  -h, --help               Show this help

Examples:
  # Recommended parallel usage: run these in two terminals.
  bash scripts/train_graph_scaling.sh polymer_chemprop
  bash scripts/train_graph_scaling.sh polymer_periodic_graph

  # Or start both from one shell and wait for both to finish.
  bash scripts/train_graph_scaling.sh both

  # Run only the large alternating density condition.
  bash scripts/train_graph_scaling.sh polymer_chemprop --only MD_5000/density/alternating_density
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODEL="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPLIT_ROOT="$REPO_ROOT/Datasets/scaling_splits"
EPOCHS=50
SEED=0
ONLY_PATTERN=""
DRY_RUN=0
QUIET=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split-root)
      SPLIT_ROOT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --only)
      ONLY_PATTERN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-quiet)
      QUIET=0
      shift
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

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

quiet_args=()
if [[ "$QUIET" -eq 1 ]]; then
  quiet_args+=(--quiet)
fi

discover_split_roots() {
  local manifest rel
  while IFS= read -r -d '' manifest; do
    rel="${manifest#"$SPLIT_ROOT/"}"
    rel="${rel%/manifest.json}"
    if [[ -n "$ONLY_PATTERN" && "$rel" != *"$ONLY_PATTERN"* ]]; then
      continue
    fi
    printf '%s\0' "$(dirname "$manifest")"
  done < <(find "$SPLIT_ROOT" -name manifest.json -print0 | sort -z)
}

train_polymer_chemprop() {
  local model_dir="$REPO_ROOT/Models/polymer_chemprop"
  local train_bin="$model_dir/.venv/bin/chemprop_train"

  if [[ ! -x "$train_bin" ]]; then
    echo "ERROR: chemprop_train not found or not executable: $train_bin" >&2
    echo "Run: cd Models/polymer_chemprop && bash setup_environment.sh" >&2
    exit 1
  fi

  local split_root rel fold_dir train_dir train_size fold train_csv val_csv test_csv save_dir log_path
  while IFS= read -r -d '' split_root; do
    rel="${split_root#"$SPLIT_ROOT/"}"
    while IFS= read -r -d '' train_dir; do
      fold_dir="$(dirname "$train_dir")"
      fold="$(basename "$fold_dir")"
      train_size="$(basename "$train_dir")"
      train_csv="$train_dir/wPSMILES/train.csv"
      val_csv="$train_dir/wPSMILES/val.csv"
      test_csv="$train_dir/wPSMILES/test.csv"
      save_dir="$model_dir/scaling_results/$rel/$train_size/$fold"
      log_path="$save_dir/train.log"

      if [[ ! -f "$train_csv" || ! -f "$val_csv" || ! -f "$test_csv" ]]; then
        echo "ERROR: missing wPSMILES split CSVs under $train_dir" >&2
        exit 1
      fi

      mkdir -p "$save_dir"
      echo "[polymer_chemprop] $rel $fold $train_size"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        run_cmd "$train_bin" --data_path "$train_csv" --separate_val_path "$val_csv" --separate_test_path "$test_csv" --dataset_type regression --pytorch_seed "$SEED" --seed "$SEED" --save_dir "$save_dir" "${quiet_args[@]}" --epochs "$EPOCHS" --num_folds 1 --save_smiles_splits --polymer --num_workers 0
      else
        run_cmd "$train_bin" --data_path "$train_csv" --separate_val_path "$val_csv" --separate_test_path "$test_csv" --dataset_type regression --pytorch_seed "$SEED" --seed "$SEED" --save_dir "$save_dir" "${quiet_args[@]}" --epochs "$EPOCHS" --num_folds 1 --save_smiles_splits --polymer --num_workers 0 > "$log_path" 2>&1
      fi
    done < <(find "$split_root" -mindepth 2 -maxdepth 2 -type d -name 'train_*' -print0 | sort -z)
  done < <(discover_split_roots)
}

train_polymer_periodic_graph() {
  local model_dir="$REPO_ROOT/Models/polymer_periodic_graph"
  local train_py="$model_dir/Polymer_Model/train.py"
  local python_bin="$model_dir/.venv/bin/python"

  if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: python venv not found or not executable: $python_bin" >&2
    echo "Run: cd Models/polymer_periodic_graph && bash setup_environment.sh" >&2
    exit 1
  fi
  if [[ ! -f "$train_py" ]]; then
    echo "ERROR: train.py not found: $train_py" >&2
    exit 1
  fi

  local split_root rel fold_dir train_dir train_size fold train_csv val_csv test_csv save_dir log_path
  while IFS= read -r -d '' split_root; do
    rel="${split_root#"$SPLIT_ROOT/"}"
    while IFS= read -r -d '' train_dir; do
      fold_dir="$(dirname "$train_dir")"
      fold="$(basename "$fold_dir")"
      train_size="$(basename "$train_dir")"
      train_csv="$train_dir/PSMILES/train.csv"
      val_csv="$train_dir/PSMILES/val.csv"
      test_csv="$train_dir/PSMILES/test.csv"
      save_dir="$model_dir/scaling_results/$rel/$train_size/$fold"
      log_path="$save_dir/train.log"

      if [[ ! -f "$train_csv" || ! -f "$val_csv" || ! -f "$test_csv" ]]; then
        echo "ERROR: missing PSMILES split CSVs under $train_dir" >&2
        exit 1
      fi

      mkdir -p "$save_dir"
      echo "[polymer_periodic_graph] $rel $fold $train_size"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        run_cmd "$python_bin" "$train_py" --data_path "$train_csv" --separate_val_path "$val_csv" --separate_test_path "$test_csv" --dataset_type regression --pytorch_seed "$SEED" --seed "$SEED" --save_dir "$save_dir" "${quiet_args[@]}" --epochs "$EPOCHS" --num_folds 1 --num_workers 0
      else
        (
          cd "$model_dir"
          run_cmd "$python_bin" "$train_py" --data_path "$train_csv" --separate_val_path "$val_csv" --separate_test_path "$test_csv" --dataset_type regression --pytorch_seed "$SEED" --seed "$SEED" --save_dir "$save_dir" "${quiet_args[@]}" --epochs "$EPOCHS" --num_folds 1 --num_workers 0 > "$log_path" 2>&1
        )
      fi
    done < <(find "$split_root" -mindepth 2 -maxdepth 2 -type d -name 'train_*' -print0 | sort -z)
  done < <(discover_split_roots)
}

case "$MODEL" in
  polymer_chemprop)
    train_polymer_chemprop
    ;;
  polymer_periodic_graph)
    train_polymer_periodic_graph
    ;;
  both)
    echo "Starting both graph model batches in parallel."
    train_polymer_chemprop &
    pcp_pid=$!
    train_polymer_periodic_graph &
    ppg_pid=$!
    wait "$pcp_pid"
    wait "$ppg_pid"
    ;;
  *)
    echo "Unknown model: $MODEL" >&2
    usage
    exit 1
    ;;
esac

echo "Done."
