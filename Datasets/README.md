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

### 2. wPSMILES (~206MB, generated)
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
- **Dataset_construction_scripts/files/**: Source data files (~80MB)
  - `Cleaned_OMersBench.jsonl` (63MB)
  - `polymer-chemprop-data/` (17MB)
- **PSMILES/**: Small web-sourced datasets (~5MB)
  - `Polymer_Genome/` - DFT computed properties
  - `OpenPoly_2025/` - Experimental/computed properties
  - `PolyMetriX/` - Glass transition temperature data

### Generated via Scripts (Excluded from Git)
- **PSMILES/**: Large generated datasets
  - `Vipea/`, `MD_300/`, `MD_5000/`, `Tg_combined/` (~86MB)
- **wPSMILES/**: All wPSMILES datasets (~206MB)
- **RDKit_descriptors/**: Molecular descriptors (optional)
- **polyBERT/**: All polyBERT datasets (~7.3GB, optional)
- **PolyInfo/**: Restricted dataset, excluded in all formats
- **pSMILES_pBERT_dict.pkl**: 224MB embedding dictionary (auto-generated)

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
```

See [Dataset_construction_scripts/README.md](Dataset_construction_scripts/README.md) for detailed instructions.

## Dataset Structure

```
Datasets/
├── PSMILES/
│   ├── Vipea/           # Electron affinity & ionization potential
│   │   ├── EA/
│   │   └── IP/
│   ├── Polymer_Genome/       # DFT computed properties
│   ├── MD_300/               # ~300 atom MD simulations
│   ├── MD_5000/              # ~5000 atom MD simulations
│   ├── PolyMetriX/           # Additional Tg data
├── wPSMILES/                 # Same structure as PSMILES
└── polyBERT/                 # Same structure with embeddings
```

## Properties Covered

| Property | Source | Format | Description |
|----------|--------|--------|-------------|
| Electron Affinity (EA) | Coley 2022 | PSMILES, wPSMILES | DFT computed |
| Ionization Potential (IP) | Coley 2022 | PSMILES, wPSMILES | DFT computed |
| Glass Transition (Tg) | Multiple | All formats | Experimental & computed |
| Density | MD, PolyInfo | All formats | MD simulations & experimental |
| Refractive Index | MD | All formats | MD simulations |
| Radius of Gyration (Rg) | MD | All formats | MD simulations |
| Heat Capacity (Cp) | MD | All formats | MD simulations |
| Band Gap | Polymer Genome | All formats | DFT computed |
| Dielectric Constant | Polymer Genome | All formats | DFT computed |

## Dataset Sizes

| Directory | Format | Size | Entries (approx) |
|-----------|--------|------|------------------|
| Vipea | All | Large | ~90,000 |
| MD_300 | All | Medium | ~20,000 |
| MD_5000 | All | Large | ~100,000 |
| Polymer_Genome | All | Small | ~700 |
| PolyMetriX | All | Medium | ~8,000 |
| PoLyInfo | None | Medium | ~15,000 |

## Data Sources & Citations

### Coley 2022
- **Source**: Computational prediction of copolymer properties
- **Properties**: Electron affinity, ionization potential
- **Method**: DFT calculations
- **Architectures**: Alternating, random, block copolymers

### OMersBench
- **Source**: Molecular dynamics simulation benchmark for polymers
- **Properties**: Density, Rg, Cp, refractive index
- **Method**: Molecular dynamics simulations
- **Chain lengths**: 300 atoms (~30 monomers), 5000 atoms (~500 monomers)

### Polymer Genome
- **Source**: High-throughput DFT calculations for polymer properties
- **Properties**: Band gap, dielectric constant, electron affinity, etc.
- **Method**: DFT calculations

### PolyInfo
- **Source**: Experimental polymer database
- **Properties**: Tg, density
- **Method**: Experimental measurements

### PolyMetriX
- **Source**: Polymer property database
- **Properties**: Tg
- **Method**: Experimental measurements

## Usage Notes

### Which Format Should I Use?

- **PSMILES**: Best for models that use SMILES directly (e.g., graph networks, SMILES-based transformers)
- **wPSMILES**: Best for models that need copolymer composition/connectivity (e.g., specialized polymer models)
- **RDKit descriptors**: Best for traditional ML models (e.g., Random Forest, SVM, XGBoost) - faster than polyBERT
- **polyBERT**: Best for deep learning models that benefit from pre-trained embeddings

### Memory Requirements

- **PSMILES/wPSMILES**: Minimal (<1GB RAM to load)
- **RDKit descriptors**: Low (~2GB RAM for largest datasets)
- **polyBERT**: Moderate (~8GB RAM for largest datasets)

### polyBERT Model Requirements

To generate polyBERT datasets, you need:
1. polyBERT model at `../Models/polyBERT/`
2. Python environment with sentence-transformers
3. ~16GB RAM for embedding generation

Download polyBERT from: https://huggingface.co/kuelumbus/polyBERT

See [Dataset_construction_scripts/README.md](Dataset_construction_scripts/README.md) for detailed setup.

## License

[License information to be added]

## Questions?

For dataset generation issues, see [Dataset_construction_scripts/README.md](Dataset_construction_scripts/README.md).

For dataset content questions, please refer to the original data sources or open an issue.
