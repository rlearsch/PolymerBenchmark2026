#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "Generating RDKit Molecular Descriptor Datasets"
echo "================================================"
echo ""
echo "WARNING: This process:"
echo "  - Requires significant time for large datasets"
echo "  - Calculates 200+ molecular descriptors per polymer"
echo "  - May take 30+ minutes for full dataset"
echo "  - Generates descriptor files in RDKit_descriptors/"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run: bash setup_environments.sh"
    exit 1
fi

# Check if PSMILES datasets exist
if [ ! -d "../../PSMILES" ]; then
    echo "ERROR: PSMILES datasets not found."
    echo "Please run: bash generate_basic_datasets.sh first"
    exit 1
fi

echo "✓ Prerequisites verified"
echo ""

# Activate environment
source .venv/bin/activate

# Create output directory
echo "Creating output directory..."
mkdir -p ../../RDKit_descriptors
echo "✓ Directory created"
echo ""

# Run script
START_TIME=$(date +%s)

echo "Calculating RDKit descriptors..."
echo "  (This may take a while - progress will be shown)"
echo ""
python PSMILES_to_RDKit_descriptors.py

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "================================================"
echo "RDKit descriptor generation complete!"
echo "Time taken: ${MINUTES}m ${SECONDS}s"
echo "================================================"
echo ""
echo "Generated datasets:"
echo "  - ../RDKit_descriptors/"
echo ""
echo "These descriptors can be used with traditional ML models"
echo "like Random Forests, SVMs, etc."
echo ""

deactivate
