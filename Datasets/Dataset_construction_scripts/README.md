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
- **For polyBERT generation**: External polyBERT checkout at `../../../polyBERT/` (a sibling of this repository)

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
- `../wPSMILES/` (~200MB)

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
- External polyBERT model must be at `../../../polyBERT/`
- Requires ~16GB RAM
- Takes 1-3 hours depending on system

```bash
bash generate_polybert_datasets.sh
```

Generates:
- `../polyBERT/` (~7.3GB)

### 5. Generate Scaling Experiment Datasets (Optional)

**Prerequisites:**
- PSMILES and wPSMILES datasets from `generate_basic_datasets.sh`
- polyBERT datasets from `generate_polybert_datasets.sh`
- RDKit in the basic dataset environment

```bash
bash generate_scaling_datasets.sh
```

This creates the exact five-fold, nested training subsets used by the
PolyBench26 scaling experiments. Use `--dry-run` to inspect the complete
experiment matrix and `--overwrite` to replace existing outputs.

Generated files are written to `../scaling_splits/` and excluded from Git. The
generator validates target alignment across PSMILES, wPSMILES, and polyBERT and
regenerates RDKit descriptors from canonical PSMILES rows. See
[`../../SCALING_EXPERIMENTS_METHODS.md`](../../SCALING_EXPERIMENTS_METHODS.md).

## Source Data Files

### Included in Repository (`./files/`)
- `Cleaned_OMersBench_v3_final.jsonl` (64MB) - Current OMersBench source, including Cv and **additional molecular dynamics parameters**
  - Includes: Degree of polymerization (DP), molecular weight (Mn, MW), number of chains, atom counts, and more
  - Users needing these parameters should parse this source file directly
- `polymer-chemprop-data/` (17MB) - Coley 2022 dataset files

### Included in PSMILES Directory (Small web-sourced datasets, ~5MB total)
- `../PSMILES/polyVERSE/` - DFT-computed electron affinity and ionization energy from polyVERSE
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
│   ├── Vipea/
│   │   ├── EA/
│   │   └── IP/
│   ├── MD_300/        # ~300 atom simulations (n_chains < 10)
│   └── MD_5000/       # ~5000 atom simulations (n_chains == 10)
├── wPSMILES/          # Same structure as PSMILES
└── polyBERT/          # Same structure, but with 600-dim embeddings
```

**Note**: MD datasets (MD_300, MD_5000) contain SMILES and properties only. For additional molecular dynamics parameters like degree of polymerization (DP), molecular weight (Mn, MW), and simulation details, see the source file `files/Cleaned_OMersBench_v3_final.jsonl`.

## Manual Script Execution

If you need fine-grained control, you can run scripts individually:

### Step 1: Activate Environment

```bash
source .venv/bin/activate
```

### Step 2: Run Individual Scripts

#### Process Coley 2022 Data
```bash
python process_Vipea_data.py
```
Generates: `Vipea/EA` and `Vipea/IP` datasets

#### Convert OMersBench Data
```bash
python OMers_convert_jsonl.py
```
Generates: `MD_300`, `MD_5000` datasets

#### Convert Web-Sourced Datasets
```bash
python convert_web_datasets.py
```
Converts small web-sourced PSMILES datasets to wPSMILES format:
- polyVERSE
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
python create_PSMILES_pBERT_dictionary.py

# Convert PSMILES to polyBERT format
python PSMILES_to_pBERT.py
```

**Important**: `create_PSMILES_pBERT_dictionary.py` requires the external polyBERT model at:
```
../../../polyBERT/
```

Download from: https://huggingface.co/HAYDERphd/polyBERT

Edit line 24 of the script if your model is in a different location.

## Script Descriptions

### `process_Vipea_data.py`
- **Input**: `files/polymer-chemprop-data/*.csv`
- **Output**: Coley 2022 EA/IP datasets in PSMILES and wPSMILES formats
- **Dependencies**: pandas, rdkit
- **Converts**: wPSMILES → PSMILES for alternating copolymers

### `OMers_convert_jsonl.py`
- **Input**: `files/Cleaned_OMersBench_v3_final.jsonl`
- **Output**: MD simulations (300 atom, 5000 atom, various DP)
- **Dependencies**: pandas, numpy, rdkit
- **Handles**: Homopolymers, alternating, and random copolymers
- **Converts**: Both PSMILES and wPSMILES formats

### `add_Cv_to_OMersBench.py`
- **Input**: An OMersBench JSONL file and an external directory of Cv calculation files
- **Output**: JSONL records augmented with matched `Cv` values
- **Note**: The large raw heat-capacity calculation directory is intentionally excluded from Git; the repository includes the compact augmented JSONL used for generation

### `combine_alternating_homopolymer.py`
- **Input**: Directory containing `alternating_*.csv` and `homopolymer_*.csv`
- **Output**: Combined `homopolymer_alternating_*.csv` files
- **Usage**: Manual, run per directory as needed

### `convert_web_datasets.py`
- **Input**: Web-sourced PSMILES datasets (polyVERSE, OpenPoly_2025, PolyMetriX)
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

### `create_PSMILES_pBERT_dictionary.py`
- **Input**: All CSV files in `../PSMILES/`, existing `pSMILES_pBERT_dict.pkl`
- **Output**: Updated `files/PSMILES_pBERT_dict.pkl`
- **Dependencies**: sentence-transformers, torch
- **Requires**: External polyBERT model at `../../../polyBERT/`
- **Note**: Incremental - only processes new SMILES not in dictionary

### `PSMILES_to_pBERT.py`
- **Input**: PSMILES datasets, `PSMILES_pBERT_dict.pkl`
- **Output**: polyBERT datasets (600-dim embeddings)
- **Dependencies**: pandas
- **Note**: Fast - just looks up embeddings in dictionary

### `create_scaling_splits.py`
- **Input**: One aligned PSMILES, wPSMILES, and polyBERT dataset
- **Output**: Nested train subsets plus fixed validation/test data and row indices
- **Dependencies**: pandas, numpy, rdkit
- **Note**: RDKit descriptors are calculated from canonical PSMILES rows

### `generate_scaling_datasets.sh`
- **Purpose**: Reproduce the complete PolyBench26 scaling experiment matrix
- **Output**: `../scaling_splits/` (generated and excluded from Git)
- **Note**: Uses five folds, base seed 42, and the documented train sizes

## Troubleshooting

### "ModuleNotFoundError: No module named 'rdkit'"
Activate the virtual environment:
```bash
source .venv/bin/activate
```

### "ERROR: polyBERT model not found"
Download the polyBERT model beside the PolyBench26 repository:
```bash
# From the repository root
git clone https://huggingface.co/HAYDERphd/polyBERT ../polyBERT
```

The sibling `../polyBERT/` directory should be a valid SentenceTransformer
model. It is separate from the repository's `Models/polyBERT/` trainer.

### "FileNotFoundError: Cleaned_OMersBench_v3_final.jsonl"
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
- **polyBERT Model**: Kuenneth, C., & Ramprasad, R. (2023). *polyBERT: a
  chemical language model to enable fully machine-driven ultrafast polymer
  informatics*. *Nature Communications*, 14, 4099.
  https://doi.org/10.1038/s41467-023-39868-6. External checkout:
  https://huggingface.co/HAYDERphd/polyBERT

## Notes

- All generated datasets are excluded from git (see `.gitignore`)
- Source files in `./files/` are included in the repository
- The polyBERT dictionary is incrementally updated - existing entries are preserved
- Scripts create output directories automatically if they don't exist
