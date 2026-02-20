#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: bash train_ppg.sh <path_to_dataset.csv>"
    echo "Example: bash train_ppg.sh ../../Datasets/PSMILES/MD_300/density/homopolymer_density.csv"
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
echo "Training polymer_periodic_graph model"
echo "================================================"
echo "Dataset: $DATA_PATH"

# Activate environment
source .venv/bin/activate

# Extract save directory from dataset path
# From: ../../Datasets/PSMILES/MD_300/density/homopolymer_density.csv
# To: MD_300/density/homopolymer_density
SUBPATH=$(echo "$DATA_PATH" | sed -E 's|.*PSMILES/([^/]+/[^/]+/)([^/_]+)(_[^/]*)?\.csv$|\1\2|')
SAVE_DIR="./${SUBPATH}/random_split/"

echo "Output directory: $SAVE_DIR"
echo ""

# Run training
START_TIME=$(date +%s)

python Polymer_Model/train.py \
  --data_path "$DATA_PATH" \
  --dataset_type regression \
  --pytorch_seed 0 \
  --save_dir "$SAVE_DIR" \
  --quiet \
  --epochs 50 \
  --num_folds 5 \
  --num_workers 0

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