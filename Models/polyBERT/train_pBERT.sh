#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: bash train_pBERT.sh <path_to_dataset.csv>"
    echo "Example: bash train_pBERT.sh ../../Datasets/polyBERT/MD_300/density/homopolymer_density.csv"
    exit 1
fi

DATA_PATH="$1"

# Check if dataset exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Dataset not found at: $DATA_PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run: bash setup_environment.sh"
    exit 1
fi

echo "================================================"
echo "Training polyBERT model"
echo "================================================"
echo "Dataset: $DATA_PATH"

# Activate environment
source .venv/bin/activate

# Extract save directory from dataset path
# From: ../../Datasets/polyBERT/MD_300/density/homopolymer_density.csv
# To: MD_300/density/homopolymer_density
SUBPATH=$(echo "$DATA_PATH" | sed -E 's|.*polyBERT/([^/]+/[^/]+/)([^/_]+)(_[^/]*)?\.csv$|\1\2|')
SAVE_DIR="./${SUBPATH}/random_split/"

echo "Output directory: $SAVE_DIR"
echo ""

# Run training
START_TIME=$(date +%s)

python train_pBERT.py \
  --data_path "$DATA_PATH" \
  --save_dir "$SAVE_DIR" \
  --num_folds 5 \
  --epochs 50 \
  --batch_size 50

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "================================================"
echo "Training complete!"
echo "================================================"
echo "Time taken: ${DURATION}s"
echo "Results saved to: $SAVE_DIR"
echo ""

deactivate
