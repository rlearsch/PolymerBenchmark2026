#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "Dataset Construction Environment Setup"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_MAJOR=3
REQUIRED_MINOR=10

CURRENT_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
CURRENT_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$CURRENT_MAJOR" -lt "$REQUIRED_MAJOR" ] || ([ "$CURRENT_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$CURRENT_MINOR" -lt "$REQUIRED_MINOR" ]); then
    echo "ERROR: Python 3.10 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION detected"
echo ""

# Setup basic environment
echo "Setting up basic dataset generation environment (.venv)..."
if [ -d ".venv" ]; then
    echo "  .venv already exists, skipping..."
else
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    echo "✓ Basic environment created"
fi
echo ""

# Setup polyBERT environment
echo "Setting up polyBERT environment (polyBERT_env)..."
if [ -d "polyBERT_env" ]; then
    echo "  polyBERT_env already exists, skipping..."
else
    python3 -m venv polyBERT_env
    source polyBERT_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements_polybert.txt
    deactivate
    echo "✓ polyBERT environment created"
fi
echo ""

echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Verify source data files exist in ./files/"
echo "  2. Run: bash generate_basic_datasets.sh"
echo "  3. (Optional) For polyBERT: bash generate_polybert_datasets.sh"
echo ""
