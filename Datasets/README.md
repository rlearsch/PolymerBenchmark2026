# Datasets

Polymer property datasets in multiple formats for machine learning benchmarking.

## Dataset Formats

Four representations of the same data:

### 1. PSMILES (~90MB, generated)
Polymer SMILES notation - lightweight text format representing polymer structures.

**Example:**
```csv
smiles,EA
*c1cc(F)c(c2c(O)cc(O)c(*)c2O)cc1F,1.878
```

### 2. wPSMILES (~200MB, generated)
Weighted PSMILES with connectivity information for copolymers.

**Example:**
```csv
smiles,EA
[*:1]C[*:2].[*:3]C(C)[*:4]|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5...,2.134
```

### 3. RDKit Descriptors (generated, optional)
200+ molecular descriptors calculated with RDKit - ready for traditional ML models.

**Example:**
```csv
MolWt,LogP,TPSA,...,EA
324.25,2.14,86.99,...,1.878
```

### 4. polyBERT (~7.3GB, generated, optional)
Pre-computed polyBERT embeddings (600-dimensional vectors) from PSMILES.

**Example:**
```csv
0,1,2,3,...,599,EA
0.244,0.875,0.278,-1.034,...,0.331,1.878
```

## Repository Contents

### Included in Git
- **Dataset_construction_scripts/files/**: Compact source data files (~81MB)
  - `Cleaned_OMersBench_v3_final.jsonl` (64MB; current MD source with Cv)
  - `polymer-chemprop-data/` (17MB)
- **PSMILES/**: Small web-sourced datasets
  - `polyVERSE/` - DFT-computed electron affinity and ionization energy
  - `PolyMetriX/` - Glass transition temperature data
- **scaling_indices/**: Published canonical folds for scaling experiments (~2MB)
- **task3_complexity/**: Compact alternating-copolymer refractive-index metadata
  and legacy test-fold assignments for the repeat-unit complexity task (~3MB)

`OpenPoly_2025/` may exist in a local working tree for optional experiments,
but it is not a tracked public source input and is not part of this release
inventory.

### Generated via Scripts (Excluded from Git)
- **PSMILES/**: Large generated datasets
  - `Vipea/`, `MD_300/`, `MD_5000/`, `Tg_combined/` (~86MB)
- **wPSMILES/**: All wPSMILES datasets (~206MB)
- **RDKit_descriptors/**: Molecular descriptors (optional)
- **polyBERT/**: All polyBERT datasets (~7.3GB, optional)
- **PolyInfo/**: Restricted dataset, excluded in all formats
- **PSMILES_pBERT_dict.pkl**: 224MB embedding dictionary (auto-generated)
- **scaling_splits/**: Fixed, representation-aligned folds for scaling experiments

The compact `scaling_indices/` package is tracked; the fully materialized
`scaling_splits/` tree is generated and excluded because it duplicates each
representation at every fold and training size.

## Quick Start: Generate Datasets

```bash
cd Dataset_construction_scripts

# 1. Setup environments (one time)
bash setup_environments.sh

# 2. Generate PSMILES + wPSMILES (~5-15 minutes)
bash generate_basic_datasets.sh

# 3. (Optional) Generate RDKit descriptors (~30 minutes)
bash generate_rdkit_datasets.sh

# 4. (Optional) Generate polyBERT (~1-3 hours, requires polyBERT model)
bash generate_polybert_datasets.sh

# 5. (Optional) Generate all dataset-size scaling splits
bash generate_scaling_datasets.sh
```

See [Dataset_construction_scripts/README.md](Dataset_construction_scripts/README.md) for detailed instructions.
The exact scaling protocol and experiment matrix are documented in
[../SCALING_EXPERIMENTS_METHODS.md](../SCALING_EXPERIMENTS_METHODS.md).
Tracked source requirements and the scaling condition matrix are also recorded
in [`dataset_manifest.json`](dataset_manifest.json) and validated on every commit.

## Dataset Structure

```
Datasets/
├── PSMILES/
│   ├── Vipea/           # Electron affinity & ionization potential
│   │   ├── EA/
│   │   └── IP/
│   ├── polyVERSE/            # DFT computed properties
│   ├── MD_300/               # ~300 atom MD simulations
│   ├── MD_5000/              # ~5000 atom MD simulations
│   ├── PolyMetriX/           # Additional Tg data
├── wPSMILES/                 # Same structure as PSMILES
├── polyBERT/                 # Same structure with embeddings
├── scaling_indices/          # Published canonical row indices
├── task3_complexity/         # Tracked Task 3 complexity metadata
└── scaling_splits/           # Generated fixed folds (excluded from Git)
```

## Properties Covered

| Property | Source | Format | Description |
|----------|--------|--------|-------------|
| Electron Affinity (EA) | VIPEA | All formats | DFT computed |
| Ionization Potential (IP) | VIPEA | All formats | DFT computed |
| Glass Transition (Tg) | PolyMetriX | All formats | Experimental |
| Density | MD, PolyInfo | All formats | MD simulations & experimental |
| Refractive Index | MD | All formats | MD simulations |
| Radius of Gyration (Rg) | MD | All formats | MD simulations |
| Heat Capacity (Cp) | MD | All formats | MD simulations |

## Dataset Sizes

| Directory | Format | Size | Entries (approx) |
|-----------|--------|------|------------------|
| Vipea | All | Large | ~90,000 |
| MD_300 | All | Medium | ~20,000 |
| MD_5000 | All | Large | ~100,000 |
| polyVERSE | All | Small | ~700 |
| PolyMetriX | All | Medium | ~8,000 |
| PoLyInfo | None | Medium | ~15,000 |

## Data Sources & Citations

### VIPEA
- **Source**: Computational prediction of copolymer properties
- **Properties**: Electron affinity, ionization potential
- **Method**: DFT calculations
- **Architectures**: Alternating, random, block copolymers
- **Citation**: Aldeghi, M., & Coley, C. W. (2022). *Chemical Science*,
  13(35), 10486–10498. https://doi.org/10.1039/D2SC02839E

### OMersBench / OPoly26
- **Source**: Molecular dynamics simulation benchmark for polymers
- **Properties**: Density, Rg, Cp, refractive index
- **Method**: Molecular dynamics simulations
- **Chain lengths**: 300 atoms (~30 monomers), 5000 atoms (~500 monomers)
- **Citation**: Levine et al. (2025). *The Open Polymers 2026 (OPoly26)
  Dataset and Evaluations*. arXiv:2512.23117.

### polyVERSE
- **Source**: High-throughput DFT calculations for polymer properties
- **Properties included here**: Electron affinity and ionization energy
- **Method**: DFT calculations
- **Citation**: Ramprasad et al. (2024). *polyVERSE: Informatics-Ready Polymer
  Datasets* (Version 1.0). https://doi.org/10.5281/zenodo.13352644

### PoLyInfo
- **Source**: Experimental polymer database
- **Properties**: Tg, density
- **Method**: Experimental measurements

### PolyMetriX
- **Source**: Polymer property database
- **Properties**: Tg
- **Method**: Experimental measurements
- **Citation**: Kunchapu, S., & Jablonka, K. M. (2025). *Curated Glass
  Transition Temperature for Polymers* (Version v1). Zenodo.
  https://doi.org/10.5281/zenodo.14980914

### Source terms

The root [data-provenance record](../DATA_PROVENANCE.md) is authoritative for
source URLs, pinned revisions, citations, and redistribution terms. The
polyVERSE CSVs and covered derivatives are governed by the GTRC license in
[`THIRD_PARTY_LICENSES/`](../THIRD_PARTY_LICENSES/), rather than the
repository's MIT software license.

## Usage Notes

### Which Format Should I Use?

- **PSMILES**: Best for models that use SMILES directly (e.g., graph networks, SMILES-based transformers)
- **wPSMILES**: Best for models that need copolymer composition/connectivity (e.g., specialized polymer models)
- **RDKit descriptors**: Best for traditional ML models (e.g., Random Forest, SVM, XGBoost) - faster than polyBERT
- **polyBERT**: Best for deep learning models that benefit from pre-trained embeddings


### polyBERT Model Requirements

To generate polyBERT datasets, you need:
1. an external polyBERT model checkout beside this repository at `../polyBERT/`
2. Python environment with sentence-transformers

Download polyBERT from: https://huggingface.co/kuelumbus/polyBERT

`Models/polyBERT/` is the PolyBench26 feed-forward trainer, not the external
SentenceTransformer model used for embedding generation.

See [Dataset_construction_scripts/README.md](Dataset_construction_scripts/README.md) for detailed setup.



## Questions?

For dataset generation issues, see [Dataset_construction_scripts/README.md](Dataset_construction_scripts/README.md).

For dataset content questions, please refer to the original data sources or open an issue.
