#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash predict_rf.sh <checkpoint_dir> <test_data.csv> [output.csv]"
    echo "Example: bash predict_rf.sh ./MD_300/density/homopolymer_density/random_split ../../Datasets/RDKit_descriptors/test.csv predictions.csv"
    echo ""
    echo "Note: checkpoint_dir should contain all fold_* subdirectories for ensemble prediction"
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
echo "Running Random Forest ensemble predictions"
echo "================================================"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Test data: $TEST_DATA"
echo "Output: $OUTPUT_PATH"
echo ""

# Activate environment
source .venv/bin/activate

# Run prediction
START_TIME=$(date +%s)

python predict_rf_ensemble.py \
  --fold-root "$CHECKPOINT_DIR" \
  --data "$TEST_DATA" \
  --out "$OUTPUT_PATH"

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
