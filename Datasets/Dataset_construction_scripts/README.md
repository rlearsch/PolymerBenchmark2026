# Dataset Construction Scripts

This directory contains scripts to generate polymer property datasets from source data files.

## Overview

These scripts convert source data into four dataset formats:
- **PSMILES**: Polymer SMILES notation (lightweight text format)
- **wPSMILES**: Weighted PSMILES with connectivity information
- **RDKit descriptors**: 200+ molecular descriptors for traditional ML models
- **polyBERT**: Pre-computed polyBERT embeddings (600-dimensional vectors)

## Prerequisites

- **Python 3.10+** (required)
- **For polyBERT generation**: Local copy of polyBERT model at `../../Models/polyBERT/`

## Quick Start

### 1. Setup Environments

```bash
bash setup_environments.sh
```

This creates two virtual environments:
- `.venv`: For PSMILES/wPSMILES generation (pandas, numpy, rdkit)
- `polyBERT_env`: For polyBERT generation (+ sentence-transformers, torch)

### 2. Generate Basic Datasets (Recommended)

```bash
bash generate_basic_datasets.sh
```

Generates:
- `../PSMILES/` (~90MB)
- `../wPSMILES/` (~206MB)

Time: ~5-15 minutes depending on system

### 3. Generate RDKit Descriptors (Optional)

**Prerequisites:**
- PSMILES datasets must be generated first
- Takes 30+ minutes for full dataset

```bash
bash generate_rdkit_datasets.sh
```

Generates:
- `../RDKit_descriptors/` - 200+ molecular descriptors per polymer
- Same directory structure as PSMILES
- Ready for use with Random Forest, SVM, etc.

### 4. Generate polyBERT Datasets (Optional)

**Prerequisites:**
- PSMILES datasets must be generated first
- polyBERT model must be at `../../Models/polyBERT/`
- Requires ~16GB RAM
- Takes 1-3 hours depending on system

```bash
bash generate_polybert_datasets.sh
```

Generates:
- `../polyBERT/` (~7.3GB)

## Source Data Files

### Included in Repository (`./files/`)
- `Cleaned_OMersBench.jsonl` (63MB) - OMersBench dataset with **additional molecular dynamics parameters**
  - Includes: Degree of polymerization (DP), molecular weight (Mn, MW), number of chains, atom counts, and more
  - Users needing these parameters should parse this source file directly
- `polymer-chemprop-data/` (17MB) - Coley 2022 dataset files

### Included in PSMILES Directory (Small web-sourced datasets, ~5MB total)
- `../PSMILES/Kuenneth_2021/` - DFT computed properties from Kuenneth 2021
- `../PSMILES/OpenPoly_2025/` - Experimental/computed properties from OpenPoly
- `../PSMILES/PolyMetriX/` - Glass transition temperature data

### Generated (Excluded from git)
- `pSMILES_pBERT_dict.pkl` (224MB) - Pre-computed polyBERT embeddings dictionary
  - **NOT included in repo** (too large)
  - **Generated automatically** when running `create_pSMILES_pBERT_dictionary.py`
  - First run will create the dictionary from scratch (takes longer)
  - Subsequent runs incrementally update it with new SMILES

## Output Structure

```
Datasets/
├── PSMILES/
│   ├── Coley_2022/
│   │   ├── EA/
│   │   └── IP/
│   ├── MD_300/        # ~300 atom simulations (n_chains < 10)
│   └── MD_5000/       # ~5000 atom simulations (n_chains == 10)
├── wPSMILES/          # Same structure as PSMILES
└── polyBERT/          # Same structure, but with 600-dim embeddings
```

**Note**: MD datasets (MD_300, MD_5000) contain SMILES and properties only. For additional molecular dynamics parameters like degree of polymerization (DP), molecular weight (Mn, MW), and simulation details, see the source file `files/Cleaned_OMersBench.jsonl`.

## Manual Script Execution

If you need fine-grained control, you can run scripts individually:

### Step 1: Activate Environment

```bash
source .venv/bin/activate
```

### Step 2: Run Individual Scripts

#### Process Coley 2022 Data
```bash
python process_Coley_data.py
```
Generates: `Coley_2022/EA` and `Coley_2022/IP` datasets

#### Convert OMersBench Data
```bash
python OMers_convert_jsonl.py
```
Generates: `MD_300`, `MD_5000`, `MD_DP` datasets

**Note**: Creates intermediate CSV files (`OMersBench_*.csv`) that can be deleted after completion.

#### Convert Web-Sourced Datasets
```bash
python convert_web_datasets.py
```
Converts small web-sourced PSMILES datasets to wPSMILES format:
- Kuenneth_2021
- OpenPoly_2025
- PolyMetriX

Uses the `PSMILES_to_wPSMILES.py` script internally.

#### Combine Alternating + Homopolymer Datasets (Optional)
```bash
python combine_alternating_homopolymer.py <input_directory>
```

Example usage:
```bash
# For each property in MD_300
python combine_alternating_homopolymer.py ../../PSMILES/MD_300/Cp
python combine_alternating_homopolymer.py ../../PSMILES/MD_300/density
python combine_alternating_homopolymer.py ../../PSMILES/MD_300/refractive_index
python combine_alternating_homopolymer.py ../../PSMILES/MD_300/Rg

# Repeat for MD_5000
python combine_alternating_homopolymer.py ../../PSMILES/MD_5000/Cp
# ... etc
```

This combines `alternating_*.csv` and `homopolymer_*.csv` into `homopolymer_alternating_*.csv`.

### Step 3: Generate polyBERT (Optional)

```bash
# Switch to polyBERT environment
deactivate
source polyBERT_env/bin/activate

# Update dictionary (uses existing dict as starting point)
python create_pSMILES_pBERT_dictionary.py

# Convert PSMILES to polyBERT format
python pSMILES_to_pBERT.py
```

**Important**: `create_pSMILES_pBERT_dictionary.py` requires polyBERT model at:
```
../../Models/polyBERT/
```

Download from: https://huggingface.co/kuelumbus/polyBERT

Edit line 24 of the script if your model is in a different location.

## Script Descriptions

### `process_Coley_data.py`
- **Input**: `files/polymer-chemprop-data/*.csv`
- **Output**: Coley 2022 EA/IP datasets in PSMILES and wPSMILES formats
- **Dependencies**: pandas, rdkit
- **Converts**: wPSMILES → PSMILES for alternating copolymers

### `OMers_convert_jsonl.py`
- **Input**: `files/Cleaned_OMersBench.jsonl`
- **Output**: MD simulations (300 atom, 5000 atom, various DP)
- **Dependencies**: pandas, numpy, rdkit
- **Handles**: Homopolymers, alternating, and random copolymers
- **Converts**: Both PSMILES and wPSMILES formats

### `combine_alternating_homopolymer.py`
- **Input**: Directory containing `alternating_*.csv` and `homopolymer_*.csv`
- **Output**: Combined `homopolymer_alternating_*.csv` files
- **Usage**: Manual, run per directory as needed

### `convert_web_datasets.py`
- **Input**: Web-sourced PSMILES datasets (Kuenneth_2021, OpenPoly_2025, PolyMetriX)
- **Output**: Corresponding wPSMILES datasets
- **Dependencies**: pandas (via PSMILES_to_wPSMILES.py)
- **Note**: Wrapper script that calls PSMILES_to_wPSMILES.py for each dataset

### `PSMILES_to_wPSMILES.py`
- **Input**: Directory of PSMILES CSV files
- **Output**: Directory of wPSMILES CSV files
- **Dependencies**: pandas
- **Usage**: `python PSMILES_to_wPSMILES.py <input_dir> <output_dir>`
- **Note**: Recursively processes all CSV files, preserving folder structure

### `wPSMILES_to_PSMILES.py`
- **Input**: wPSMILES strings
- **Output**: PSMILES strings
- **Dependencies**: rdkit
- **Note**: Contains conversion functions for homopolymers and alternating copolymers

### `polymer_conversions.py`
- **Purpose**: Centralized module containing all PSMILES/wPSMILES conversion functions
- **Functions**: 
  - PSMILES ↔ wPSMILES conversion for homopolymers, alternating, and random copolymers
  - Ring elevation algorithms, fragment joining utilities
- **Note**: All other scripts import from this module to avoid code duplication

### `PSMILES_to_RDKit_descriptors.py`
- **Input**: All CSV files in `../PSMILES/`
- **Output**: RDKit molecular descriptors in `../RDKit_descriptors/`
- **Dependencies**: rdkit
- **Descriptors**: 200+ molecular descriptors (MW, LogP, TPSA, etc.)
- **Note**: Recursively processes all PSMILES datasets, skips already-processed files

### `create_pSMILES_pBERT_dictionary.py`
- **Input**: All CSV files in `../PSMILES/`, existing `pSMILES_pBERT_dict.pkl`
- **Output**: Updated `files/pSMILES_pBERT_dict.pkl`
- **Dependencies**: sentence-transformers, torch
- **Requires**: polyBERT model at `../../Models/polyBERT/`
- **Note**: Incremental - only processes new SMILES not in dictionary

### `pSMILES_to_pBERT.py`
- **Input**: PSMILES datasets, `pSMILES_pBERT_dict.pkl`
- **Output**: polyBERT datasets (600-dim embeddings)
- **Dependencies**: pandas
- **Note**: Fast - just looks up embeddings in dictionary

## Troubleshooting

### "ModuleNotFoundError: No module named 'rdkit'"
Activate the virtual environment:
```bash
source .venv/bin/activate
```

### "ERROR: polyBERT model not found"
Download the polyBERT model and place it at `../../Models/polyBERT/`:
```bash
# From the repository root
mkdir -p Models
cd Models
git clone https://huggingface.co/kuelumbus/polyBERT
```

The model should be a valid SentenceTransformer model directory.

### "FileNotFoundError: Cleaned_OMersBench.jsonl"
Ensure source data files are in the `./files/` directory. These should be included in the repository.

### Intermediate CSV files cluttering directory
The `OMers_convert_jsonl.py` script creates `OMersBench_alternating.csv`, `OMersBench_random.csv`, `OMersBench_homopolymer.csv` in the script directory. These can be safely deleted after generation completes.

### Out of memory during polyBERT generation
The `create_pSMILES_pBERT_dictionary.py` script requires significant RAM. Try:
- Closing other applications
- Processing datasets in batches (edit script to target specific subdirectories)
- Using a machine with more RAM

### Paths not working
All scripts use paths relative to their location. Ensure you run them from within the `Dataset_construction_scripts/` directory or use the provided bash scripts.

## Dependencies Reference

### Basic Environment (.venv)
```
pandas>=2.0.0
numpy>=1.24.0
rdkit>=2023.9.1
```

### polyBERT Environment (polyBERT_env)
```
pandas>=2.0.0
numpy>=1.24.0
sentence-transformers>=2.2.0
torch>=2.0.0
```

## Dataset Sources & Citations

- **Coley 2022**: Computational prediction of copolymer properties
- **OMersBench**: Molecular dynamics simulation benchmark for polymers
- **polyBERT Model**: https://huggingface.co/kuelumbus/polyBERT

## Notes

- All generated datasets are excluded from git (see `.gitignore`)
- Source files in `./files/` are included in the repository
- The polyBERT dictionary is incrementally updated - existing entries are preserved
- Scripts create output directories automatically if they don't exist
