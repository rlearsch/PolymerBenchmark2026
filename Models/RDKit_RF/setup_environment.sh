#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "Setting up RDKit_RF environment"
echo "================================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

echo "Detected Python version: $PYTHON_VERSION"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "ERROR: Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

# Create or repair the virtual environment. Reinstalling requirements on every
# run makes retries safe after an interrupted dependency installation.
if [ ! -x ".venv/bin/python" ]; then
    if [ -e ".venv" ]; then
        echo "Removing incomplete virtual environment..."
        rm -rf .venv
    fi
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip

echo "Installing RDKit_RF dependencies..."
echo "This may take several minutes..."
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import joblib, numpy, pandas, rdkit, sklearn"

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "To activate the environment:"
echo "  cd $(pwd)"
echo "  source .venv/bin/activate"
echo ""
echo "To train a model:"
echo "  bash train_rf.sh ../../Datasets/RDKit_descriptors/MD_300/density/homopolymer_density.csv"
echo ""
