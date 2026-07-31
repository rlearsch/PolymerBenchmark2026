#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "Generating PSMILES and wPSMILES Datasets"
echo "================================================"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run: bash setup_environments.sh"
    exit 1
fi

# Check source files
echo "Checking source files..."
if [ ! -f "files/Cleaned_OMersBench.jsonl" ]; then
    echo "ERROR: files/Cleaned_OMersBench.jsonl not found"
    exit 1
fi
if [ ! -d "files/polymer-chemprop-data" ]; then
    echo "ERROR: files/polymer-chemprop-data/ not found"
    exit 1
fi
echo "✓ Source files found"
echo ""

# Activate environment
source .venv/bin/activate

# Create output directories
echo "Creating output directories..."
mkdir -p ../../Datasets/PSMILES/Vipea/{EA,IP}
mkdir -p ../../Datasets/PSMILES/MD_300/{Cp,Cv,density,refractive_index,Rg}
mkdir -p ../../Datasets/PSMILES/MD_5000/{Cp,Cv,density,refractive_index,Rg}
mkdir -p ../../Datasets/wPSMILES/Vipea/{EA,IP}
mkdir -p ../../Datasets/wPSMILES/MD_300/{Cp,Cv,density,refractive_index,Rg}
mkdir -p ../../Datasets/wPSMILES/MD_5000/{Cp,Cv,density,refractive_index,Rg}
echo "✓ Directories created"
echo ""

# Run scripts
START_TIME=$(date +%s)

echo "[1/3] Processing Vipea data..."
python process_Vipea_data.py
echo "✓ Vipea data processed"
echo ""

echo "[2/3] Converting OMersBench data..."
echo "  (This may take several minutes)"
python OMers_convert_jsonl.py
echo "✓ OMersBench data converted"
echo ""

echo "[3/3] Converting web-sourced datasets..."
echo "  (Polymer_Genome, OpenPoly_2025, PolyMetriX)"
python convert_web_datasets.py
echo "✓ Web datasets converted"
echo ""

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo "================================================"
echo "Basic dataset generation complete!"
echo "Time taken: ${MINUTES}m ${SECONDS}s"
echo "================================================"
echo ""
echo "Generated datasets:"
echo "  - ../../Datasets/PSMILES/"
echo "  - ../../Datasets/wPSMILES/"
echo ""
echo "Optional next steps:"
echo "  - Combine datasets: See README.md for combine_alternating_homopolymer.py usage"
echo "  - Generate polyBERT: bash generate_polybert_datasets.sh"
echo ""

deactivate
