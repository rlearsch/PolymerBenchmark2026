#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "Generating polyBERT Datasets"
echo "================================================"
echo ""
echo "WARNING: This process:"
echo "  - Requires ~16GB RAM"
echo "  - Takes several hours depending on system"
echo "  - Generates ~7.3GB of data"
echo "  - Requires polyBERT model at ../../Models/polyBERT/"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Check if polyBERT env exists
if [ ! -d "polyBERT_env" ]; then
    echo "ERROR: polyBERT environment not found."
    echo "Please run: bash setup_environments.sh"
    exit 1
fi

# Check if PSMILES datasets exist
if [ ! -d "../../Datasets/PSMILES" ]; then
    echo "ERROR: PSMILES datasets not found."
    echo "Please run: bash generate_basic_datasets.sh first"
    exit 1
fi

# Check if polyBERT model exists
if [ ! -d "../../Models/polyBERT" ]; then
    echo "ERROR: polyBERT model not found at ../../Models/polyBERT/"
    echo "Please download the polyBERT model first:"
    echo "  https://huggingface.co/kuelumbus/polyBERT"
    exit 1
fi

echo "✓ Prerequisites verified"
echo ""

# Activate environment
source polyBERT_env/bin/activate

# Create output directory
echo "Creating output directory..."
mkdir -p ../../Datasets/polyBERT
echo "✓ Directory created"
echo ""

# Run scripts
START_TIME=$(date +%s)

echo "[1/2] Creating/updating polyBERT dictionary..."
echo "  (This may take a long time for the first run)"
python create_PSMILES_pBERT_dictionary.py
echo "✓ Dictionary updated"
echo ""

echo "[2/2] Converting PSMILES to polyBERT format..."
python PSMILES_to_pBERT.py
echo "✓ Conversion complete"
echo ""

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo "================================================"
echo "polyBERT dataset generation complete!"
echo "Time taken: ${MINUTES}m ${SECONDS}s"
echo "================================================"
echo ""
echo "Generated datasets:"
echo "  - ../../Datasets/polyBERT/ (~7.3GB)"
echo ""

deactivate
