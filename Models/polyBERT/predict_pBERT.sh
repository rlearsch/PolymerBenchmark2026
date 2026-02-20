#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash predict_pBERT.sh <checkpoint_dir> <test_data.csv> [output.csv]"
    echo "Example: bash predict_pBERT.sh ./MD_300/density/homopolymer_density/random_split ../../Datasets/polyBERT/test.csv predictions.csv"
    echo ""
    echo "Note: checkpoint_dir should contain all fold_* subdirectories with checkpoint_best.pt for ensemble prediction"
    exit 1
fi

CHECKPOINT_DIR="$1"
TEST_DATA="$2"
OUTPUT_PATH="${3:-predictions.csv}"

# Validation checks
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Test data not found: $TEST_DATA"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run: bash setup_environment.sh"
    exit 1
fi

echo "================================================"
echo "Running polyBERT ensemble predictions"
echo "================================================"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Test data: $TEST_DATA"
echo "Output: $OUTPUT_PATH"
echo ""

# Activate environment
source .venv/bin/activate

# Run prediction with glob to find all checkpoint_best.pt files
START_TIME=$(date +%s)

python pBERT_predict_ensemble.py \
  --data_csv "$TEST_DATA" \
  --ckpt_glob "${CHECKPOINT_DIR}/**/checkpoint_best.pt" \
  --out_csv "$OUTPUT_PATH" \
  --x_cols 600 \
  --y_col 600 \
  --batch_size 2048 \
  --device cpu

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "================================================"
echo "Predictions complete!"
echo "================================================"
echo "Time taken: ${DURATION}s"
echo "Predictions saved to: $OUTPUT_PATH"
echo ""

deactivate

