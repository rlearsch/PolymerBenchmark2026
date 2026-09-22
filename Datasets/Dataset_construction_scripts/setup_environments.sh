#!/bin/bash
set -euo pipefail

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

WITH_POLYBERT=false
if [[ "${1:-}" == "--with-polybert" ]]; then
    WITH_POLYBERT=true
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash setup_environments.sh [--with-polybert]"
    exit 0
elif [[ $# -gt 0 ]]; then
    echo "ERROR: Unknown option: $1" >&2
    exit 2
fi

setup_environment() {
    local environment="$1"
    local requirements="$2"
    local imports="$3"

    if [[ ! -x "$environment/bin/python" ]]; then
        if [[ -e "$environment" ]]; then
            echo "  Removing incomplete $environment..."
            rm -rf "$environment"
        fi
        python3 -m venv "$environment"
    fi

    "$environment/bin/python" -m pip install --upgrade pip
    "$environment/bin/python" -m pip install -r "$requirements"
    "$environment/bin/python" -c "$imports"
}

echo "Setting up basic dataset generation environment (.venv)..."
setup_environment ".venv" "requirements.txt" "import numpy, pandas, rdkit"
echo "✓ Basic environment is ready"
echo ""

if [[ "$WITH_POLYBERT" == true ]]; then
    echo "Setting up polyBERT environment (polyBERT_env)..."
    setup_environment "polyBERT_env" "requirements_polybert.txt" "import numpy, pandas, sentence_transformers, torch"
    echo "✓ polyBERT environment is ready"
    echo ""
fi
echo ""

echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Verify source data files exist in ./files/"
echo "  2. Run: bash generate_basic_datasets.sh"
echo "  3. (Optional) For polyBERT: bash setup_environments.sh --with-polybert"
echo "     then bash generate_polybert_datasets.sh"
echo ""
