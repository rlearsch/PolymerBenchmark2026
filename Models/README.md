# Models

This directory contains implementations of various machine learning models for polymer property prediction, evaluated in the OPoly26Benchmark.

## Available Models

| Model | Input Format | Type | Status | Description |
|-------|--------------|------|--------|-------------|
| **polymer_chemprop** | wPSMILES | GNN | ✅ Ready | Weighted directed MPNN for polymers |
| **polyBERT** | polyBERT embeddings | Feed-forward NN | ✅ Ready | Transfer learning with polyBERT |
| **RDKit_RF** | RDKit descriptors | Random Forest | ✅ Ready | Traditional ML baseline |
| **polymer_periodic_graph** | wPSMILES | Periodic GNN | 🔒 Private | Periodic graph neural network |

## Quick Start

Each model has its own directory with:
- `README.md` - Model-specific documentation
- `requirements.txt` - Python dependencies
- `setup_environment.sh` - One-command setup
- Training and prediction scripts

### General Workflow

```bash
# 1. Navigate to model directory
cd polymer_chemprop

# 2. Setup environment (one-time)
bash setup_environment.sh

# 3. Train model
bash train_pcp.sh ../../Datasets/wPSMILES/MD_300/density/homopolymer_density.csv

# 4. Make predictions
bash predict_pcp.sh ./MD_300/density/homopolymer_density/random_split test_data.csv predictions.csv
```

## Model Descriptions

### polymer_chemprop
- **Paper**: Aldeghi & Coley, Chem. Sci. 2022
- **Input**: wPSMILES (weighted polymer SMILES)
- **Architecture**: Weighted, directed message passing neural network
- **Best for**: Copolymers with complex architectures
- **Training time**: ~10-30 minutes per property

### polyBERT
- **Paper**: Kuenneth et al., 2023
- **Input**: 600-dimensional polyBERT embeddings
- **Architecture**: Feed-forward neural network on pre-trained embeddings
- **Best for**: Transfer learning, limited data scenarios
- **Training time**: ~5-15 minutes per property

### RDKit_RF
- **Input**: 200+ RDKit molecular descriptors
- **Architecture**: Random Forest ensemble
- **Best for**: Baseline comparisons, interpretability, fast training
- **Training time**: ~1-5 minutes per property

## Dataset Compatibility

| Model | PSMILES | wPSMILES | RDKit Descriptors | polyBERT Embeddings |
|-------|---------|----------|-------------------|---------------------|
| polymer_chemprop | ❌ | ✅ | ❌ | ❌ |
| polyBERT | ❌ | ❌ | ❌ | ✅ |
| RDKit_RF | ❌ | ❌ | ✅ | ❌ |

## Performance Comparison

Results will be added after running all models. See paper for full benchmark results.

## Common Setup Pattern

All models follow this structure:

```
Models/<model_name>/
├── README.md              # Model documentation
├── requirements.txt       # Python dependencies
├── setup_environment.sh   # Environment setup
├── train_*.sh             # Training script
├── predict_*.sh           # Prediction script
├── .venv/                 # Virtual environment (excluded from git)
└── <dataset>/             # Training outputs (excluded from git)
    └── <property>/
        └── random_split/
            ├── fold_0/
            ├── fold_1/
            └── ...
```

## Prerequisites

- Python 3.10 or higher
- Generated datasets (see `Datasets/README.md`)
- ~5-10GB disk space for all models

## Generating Datasets

If you haven't generated datasets yet:

```bash
cd ../Datasets/Dataset_construction_scripts
bash setup_environments.sh
bash generate_basic_datasets.sh      # PSMILES + wPSMILES
bash generate_rdkit_datasets.sh      # RDKit descriptors (optional)
bash generate_polybert_datasets.sh   # polyBERT embeddings (optional)
```

## Citation

If you use these models, please cite the OPoly26Benchmark paper:

```bibtex
@article{yourname2026opolybench,
  title={OPoly26Benchmark: A Comprehensive Benchmark for Polymer Property Prediction},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2026}
}
```

And cite the individual model papers as appropriate (see each model's README).

## Troubleshooting

### Virtual Environment Issues
Each model has its own isolated environment to prevent dependency conflicts. Always activate the correct environment:

```bash
cd Models/<model_name>
source .venv/bin/activate
```

### Dataset Not Found
Ensure datasets are generated first:
```bash
cd ../../Datasets/Dataset_construction_scripts
bash generate_basic_datasets.sh
```

### Out of Memory
- Start with smaller datasets (MD_300 instead of MD_5000)
- Reduce batch sizes in training scripts
- Close other applications

## Contributing

This benchmark is provided to reproduce research results. For issues or questions, please open a GitHub issue.
