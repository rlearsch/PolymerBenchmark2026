#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -ne 3 ]; then
    echo "Usage: bash predict_ppg.sh <checkpoint_dir> <test_data.csv> <output.csv>"
    echo "Example: bash predict_ppg.sh ./MD_300/density/homopolymer_density/random_split/ test.csv predictions.csv"
    exit 1
fi

CHECKPOINT_DIR="$1"
TEST_DATA="$2"
OUTPUT_PATH="$3"

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found at: $CHECKPOINT_DIR"
    exit 1
fi

# Check if test data exists
if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Test data file not found at: $TEST_DATA"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run: bash setup_environment.sh"
    exit 1
fi

echo "================================================"
echo "Making predictions with polymer_periodic_graph model"
echo "================================================"
echo "Test data: $TEST_DATA"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_PATH"
echo ""

# Activate environment
source .venv/bin/activate

# Run prediction
START_TIME=$(date +%s)

python Polymer_Model/predict.py \
  --test_path "$TEST_DATA" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --preds_path "$OUTPUT_PATH" \
  --num_workers 0

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "================================================"
echo "Prediction complete!"
echo "================================================"
echo "Time taken: ${DURATION}s"
echo "Predictions saved to: $OUTPUT_PATH"
echo ""

deactivate