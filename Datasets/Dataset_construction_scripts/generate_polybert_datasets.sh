#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_MODEL_PATH="$SCRIPT_DIR/../../../polyBERT"
MODEL_PATH="${POLYBERT_MODEL_PATH:-$DEFAULT_MODEL_PATH}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash generate_polybert_datasets.sh [--model-path PATH]"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            exit 2
            ;;
    esac
done
MODEL_PATH="$(cd "$MODEL_PATH" 2>/dev/null && pwd || true)"

echo "================================================"
echo "Generating polyBERT Datasets"
echo "================================================"
echo ""
echo "WARNING: This process:"
echo "  - Requires ~16GB RAM"
echo "  - Takes several hours depending on system"
echo "  - Generates ~7.3GB of data"
echo "  - Requires a local polyBERT model (default: ../../../polyBERT/)"
echo ""
if [[ ! -x "polyBERT_env/bin/python" ]]; then
    echo "ERROR: polyBERT environment not found."
    echo "Please run: bash setup_environments.sh"
    exit 1
fi

# Check if PSMILES datasets exist
if [ ! -d "./../../Datasets/PSMILES" ]; then
    echo "ERROR: PSMILES datasets not found."
    echo "Please run: bash generate_basic_datasets.sh first"
    exit 1
fi

# Check if polyBERT model exists
if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: polyBERT model not found at ${POLYBERT_MODEL_PATH:-$DEFAULT_MODEL_PATH}"
    echo "Please download the polyBERT model first:"
    echo "  https://huggingface.co/kuelumbus/polyBERT"
    exit 1
fi

echo "✓ Prerequisites verified"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

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
python create_PSMILES_pBERT_dictionary.py --model-path "$MODEL_PATH"
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
