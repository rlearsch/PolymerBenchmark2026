# OPoly26Benchmark

A comprehensive benchmark for machine learning models on polymer property prediction, accompanying our peer-reviewed journal article.

## Overview

This repository provides datasets and code to reproduce ML benchmarking results for polymer property prediction. We evaluate multiple model architectures across various polymer representations (PSMILES, wPSMILES, RDKit descriptors, polyBERT embeddings) on diverse property prediction tasks.

## Repository Structure

```
OPoly26Benchmark/
├── Datasets/                          # Polymer property datasets
│   ├── Dataset_construction_scripts/  # Scripts to generate datasets
│   ├── PSMILES/                       # Polymer SMILES notation (source + generated)
│   ├── wPSMILES/                      # Weighted PSMILES (generated)
│   ├── RDKit_descriptors/             # Molecular descriptors (generated)
│   └── polyBERT/                      # Pre-computed embeddings (generated)
├── Models/                            # Model implementations (to be added)
└── README.md                          # This file
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Git
- ~15 GB free disk space (for generated datasets)
- Optional: ~16GB RAM for polyBERT generation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/OPoly26Benchmark.git
cd OPoly26Benchmark
```

### 2. Generate Datasets

The repository includes source data (~85MB) but requires generation of full datasets:

```bash
cd Datasets/Dataset_construction_scripts

# Setup virtual environments (one-time)
bash setup_environments.sh

# Generate PSMILES and wPSMILES datasets (~5-15 minutes)
bash generate_basic_datasets.sh

# Optional: Generate RDKit descriptors (~30 minutes)
bash generate_rdkit_datasets.sh

# Optional: Generate polyBERT embeddings (~1-3 hours)
# Requires polyBERT model - see instructions below
bash generate_polybert_datasets.sh
```

**Output:**
- `Datasets/PSMILES/` - Polymer SMILES notation (~90MB)
- `Datasets/wPSMILES/` - Weighted PSMILES (~206MB)
- `Datasets/RDKit_descriptors/` - Molecular descriptors (~50MB, optional)
- `Datasets/polyBERT/` - Pre-computed embeddings (~7.3GB, optional)

### 3. Download polyBERT Model (Optional)

For polyBERT embedding generation:

```bash
# From repository root
mkdir -p Models
cd Models
git clone https://huggingface.co/kuelumbus/polyBERT
cd ..
```

## Dataset Formats

We provide polymer datasets in four complementary formats:

### PSMILES (Polymer SMILES)
- **Size**: ~90MB
- **Format**: Text-based polymer notation with attachment points (`*`)
- **Best for**: Graph neural networks, SMILES-based transformers
- **Example**: `*c1cc(F)c(c2c(O)cc(O)c(*)c2O)cc1F`

### wPSMILES (Weighted Polymer SMILES)
- **Size**: ~206MB
- **Format**: PSMILES with numbered attachment points and connectivity information
- **Best for**: Specialized polymer models requiring copolymer architecture
- **Example**: `[*:1]C[*:2].[*:3]C(C)[*:4]|0.5|0.5|<1-3:0.5:0.5...`

### RDKit Descriptors
- **Size**: ~50MB
- **Format**: 200+ molecular descriptors (MW, LogP, TPSA, etc.)
- **Best for**: Traditional ML (Random Forest, SVM, XGBoost)
- **Generation time**: ~30 minutes

### polyBERT Embeddings
- **Size**: ~7.3GB
- **Format**: 600-dimensional pre-trained embeddings
- **Best for**: Deep learning with transfer learning
- **Generation time**: ~1-3 hours

## Datasets Included

### From Published Sources
- **Kuenneth 2021**: DFT-computed properties (band gap, dielectric constant, etc.)
- **OpenPoly 2025**: Experimental/computational polymer properties
- **PolyMetriX**: Glass transition temperature data
- **Coley 2022**: Electron affinity and ionization potential (DFT)
- **OMersBench**: MD simulation properties (density, Rg, Cp, refractive index)

### Properties Covered
- Glass transition temperature (Tg)
- Density
- Electron affinity / Ionization potential
- Band gap (bulk, chain)
- Dielectric constant
- Refractive index
- Radius of gyration (Rg)
- Heat capacity (Cp)
- Crystallization tendency
- And more...

## Documentation

- **[Datasets/README.md](Datasets/README.md)**: Overview of dataset formats and sources
- **[Datasets/Dataset_construction_scripts/README.md](Datasets/Dataset_construction_scripts/README.md)**: Detailed dataset generation instructions

## Data Restrictions

**PolyInfo Dataset**: The PolyInfo database is NOT included in this repository and NOT approved for public distribution. Scripts exclude PolyInfo data by default. Users must obtain their own access to PolyInfo if needed.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{yourname2026opolybench,
  title={OPoly26Benchmark: A Comprehensive Benchmark for Polymer Property Prediction},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2026}
}
```

## License

[License to be specified]

## Contributing

This repository is provided to reproduce research results. For questions or issues, please open a GitHub issue.

## Acknowledgments

- polyBERT model: https://huggingface.co/kuelumbus/polyBERT
- Data sources: Kuenneth et al., Coley et al., OMersBench, OpenPoly, PolyMetriX
- RDKit: Open-source cheminformatics toolkit

## System Requirements

### Minimum
- Python 3.10+
- 8GB RAM
- 15GB disk space

### Recommended
- Python 3.10+
- 16GB RAM (for polyBERT generation)
- 20GB disk space
- SSD for faster dataset generation

## Troubleshooting

### "ModuleNotFoundError" when running scripts
Make sure you've activated the virtual environment:
```bash
cd Datasets/Dataset_construction_scripts
source .venv/bin/activate
```

### "PSMILES directory not found"
Run `generate_basic_datasets.sh` before optional generation steps.

### polyBERT generation fails
Ensure you've downloaded the polyBERT model to `Models/polyBERT/`:
```bash
mkdir -p Models
cd Models
git clone https://huggingface.co/kuelumbus/polyBERT
```

### Out of memory during generation
- Close other applications
- For polyBERT: Requires ~16GB RAM
- Consider generating smaller datasets first to test

## Contact

For questions about the benchmark or datasets, please open an issue on GitHub.

---

**Note**: This is a research repository accompanying a peer-reviewed publication. Generated datasets are excluded from git to keep the repository size manageable (~85MB). All datasets can be regenerated from source data using the provided scripts.
