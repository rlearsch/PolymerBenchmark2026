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
├── Models/                            # Model implementations
│   ├── polymer_chemprop/              # ✅ GNN for weighted polymer graphs
│   ├── RDKit_RF/                      # ✅ Random Forest baseline
│   ├── polyBERT/                      # ✅ Transfer learning with polyBERT
│   └── polymer_periodic_graph/        # ✅ Periodic graph neural network
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

### 4. Train Models

Each model has its own environment and training workflow:

```bash
# Example: Train Random Forest on RDKit descriptors
cd Models/RDKit_RF
bash setup_environment.sh  # one-time setup
bash train_rf.sh ../../Datasets/RDKit_descriptors/MD_300/density/homopolymer_density.csv

# Example: Train polymer-chemprop on wPSMILES
cd ../polymer_chemprop
bash setup_environment.sh  # one-time setup
bash train_pcp.sh ../../Datasets/wPSMILES/MD_300/density/homopolymer_density.csv

# Example: Train polyBERT on embeddings
cd ../polyBERT
bash setup_environment.sh  # one-time setup
bash train_pBERT.sh ../../Datasets/polyBERT/MD_300/density/homopolymer_density.csv

# Example: Train polymer_periodic_graph on PSMILES
cd ../polymer_periodic_graph
bash setup_environment.sh  # one-time setup
bash train_ppg.sh ../../Datasets/PSMILES/MD_300/density/homopolymer_density.csv
```

See [Models/README.md](Models/README.md) for detailed model documentation.

## Dataset Formats

We provide polymer datasets in four complementary formats:

### PSMILES (Polymer SMILES)
- **Size**: ~90MB
- **Format**: Text-based polymer notation with attachment points (`*`)
- **Best for**: Graph neural networks, SMILES-based transformers
- **Example**: `*c1cc(F)c(c2c(O)cc(O)c(*)c2O)cc1F`
- **Used by**: polymer_periodic_graph

### wPSMILES (Weighted Polymer SMILES)
- **Size**: ~206MB
- **Format**: PSMILES with numbered attachment points and connectivity information
- **Best for**: Specialized polymer models requiring copolymer architecture
- **Example**: `[*:1]C[*:2].[*:3]C(C)[*:4]|0.5|0.5|<1-3:0.5:0.5...`
- **Used by**: polymer_chemprop

### RDKit Descriptors
- **Size**: ~50MB
- **Format**: 200+ molecular descriptors (MW, LogP, TPSA, etc.)
- **Best for**: Traditional ML (Random Forest, SVM, XGBoost)
- **Generation time**: ~30 minutes
- **Used by**: RDKit_RF

### polyBERT Embeddings
- **Size**: ~7.3GB
- **Format**: 600-dimensional pre-trained embeddings
- **Best for**: Deep learning with transfer learning
- **Generation time**: ~1-3 hours
- **Used by**: polyBERT

## Models Included

### ✅ polymer_chemprop (Graph Neural Network)
- **Type**: Weighted, directed message passing neural network
- **Input**: wPSMILES
- **Training time**: ~10-30 minutes per property
- **Paper**: Aldeghi & Coley, Chem. Sci. 2022
- **Best for**: Copolymers with complex architectures

### ✅ RDKit_RF (Random Forest Baseline)
- **Type**: Traditional machine learning
- **Input**: RDKit descriptors
- **Training time**: ~1-5 minutes per property
- **Best for**: Fast baseline, interpretable predictions

### ✅ polyBERT (Transfer Learning)
- **Type**: Feed-forward neural network on pre-trained embeddings
- **Input**: polyBERT embeddings
- **Training time**: ~5-15 minutes per property
- **Paper**: Kuenneth et al., 2023
- **Best for**: Transfer learning, limited data scenarios

See [Models/README.md](Models/README.md) for detailed documentation on each model.

## Datasets Included

### From Published Sources
- **Polymer Genome**: DFT-computed properties (band gap, dielectric constant, etc.)
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
- **[Models/README.md](Models/README.md)**: Overview of all models and usage instructions
- **[Models/polymer_chemprop/README.md](Models/polymer_chemprop/README.md)**: polymer_chemprop model details
- **[Models/RDKit_RF/README.md](Models/RDKit_RF/README.md)**: RDKit_RF model details
- **[Models/polyBERT/README.md](Models/polyBERT/README.md)**: polyBERT model details

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
- Data sources: Polymer Genome, Coley et al., OMersBench, OpenPoly, PolyMetriX
- RDKit: Open-source cheminformatics toolkit

## System Requirements

### Minimum
- Python 3.10+
- 8GB RAM
- 15GB disk space (datasets only)
- 20GB disk space (datasets + all models)

### Recommended
- Python 3.10+
- 16GB RAM (for polyBERT generation and training)
- 25GB disk space
- SSD for faster dataset generation and model training
- GPU with CUDA support (optional, 2-5x speedup for polyBERT and polymer_chemprop)

## Troubleshooting

### Dataset Generation Issues

**"ModuleNotFoundError" when running scripts**

Make sure you've activated the virtual environment:
```bash
cd Datasets/Dataset_construction_scripts
source .venv/bin/activate
```

**"PSMILES directory not found"**

Run `generate_basic_datasets.sh` before optional generation steps.

**polyBERT generation fails**

Ensure you've downloaded the polyBERT model to `Models/polyBERT/`:
```bash
mkdir -p Models
cd Models
git clone https://huggingface.co/kuelumbus/polyBERT
```

**Out of memory during generation**

- Close other applications
- For polyBERT: Requires ~16GB RAM
- Consider generating smaller datasets first to test

### Model Training Issues

**"Virtual environment not found"**

Run the setup script for the specific model:
```bash
cd Models/<model_name>
bash setup_environment.sh
```

**Training is slow**

- Use smaller datasets (MD_300 instead of MD_5000) for testing
- For polyBERT/polymer_chemprop: Ensure GPU is available (if you have one)
- RDKit_RF is naturally the fastest model

**Out of memory during training**

- Reduce batch sizes in training scripts
- Use smaller datasets
- Close other applications
- For polyBERT: Edit `train_pBERT.sh` and change `--batch_size 50` to `--batch_size 32`
- For polymer_chemprop: Add `--batch_size 32` to train_pcp.sh

### General Issues

**Different results across models**

This is expected! While all models use consistent seeding (base seed 42 for RDKit_RF and polyBERT), the actual train/test splits differ due to different RNG libraries (sklearn vs PyTorch vs chemprop). This is normal and acceptable for benchmarking. See [Models/README.md](Models/README.md) for details on random seed strategy.

## Contact

For questions about the benchmark or datasets, please open an issue on GitHub.

---

**Note**: This is a research repository accompanying a peer-reviewed publication. Generated datasets are excluded from git to keep the repository size manageable (~85MB). All datasets can be regenerated from source data using the provided scripts.
