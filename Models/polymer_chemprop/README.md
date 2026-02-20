# Polymer-Chemprop Model

Weighted, directed Message Passing Neural Network (wD-MPNN) for polymer property prediction, based on the work by Aldeghi & Coley.

## Overview

This model uses a modified version of Chemprop that handles polymer ensemble representations (wPSMILES format) for property prediction. The wD-MPNN architecture accounts for:
- Multiple monomer types in copolymers
- Weighted connectivity patterns
- Degree of polymerization effects

## Reference

**Paper**: ["A graph representation of molecular ensembles for polymer property prediction"](https://pubs.rsc.org/en/content/articlelanding/2022/SC/D2SC02839E)  
**Authors**: Matteo Aldeghi and Connor W. Coley  
**Journal**: Chem. Sci., 2022, 13, 10486-10498  
**Repository**: https://github.com/coleygroup/polymer-chemprop

## Prerequisites

- Python 3.10 or higher
- ~2GB disk space for dependencies
- Generated wPSMILES datasets (see `Datasets/README.md`)

## Setup

### One-Time Setup

```bash
cd Models/polymer_chemprop
bash setup_environment.sh
```

This will:
1. Create a virtual environment (`.venv/`)
2. Install polymer-chemprop from GitHub
3. Install all dependencies

### Manual Setup (Alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

Train a model on any wPSMILES dataset:

```bash
bash train_pcp.sh ../../Datasets/wPSMILES/MD_300/density/homopolymer_density.csv
```

**Training parameters** (in script):
- 5-fold cross-validation
- 50 epochs
- PyTorch seed: 0
- Saves SMILES splits for reproducibility

**Output structure**:
```
Models/polymer_chemprop/
└── MD_300/density/homopolymer_density/random_split/
    ├── fold_0/
    │   ├── model.pt
    │   ├── test_scores.csv
    │   └── ...
    ├── fold_1/
    └── ...
```

### Prediction

Make predictions using trained checkpoints (ensemble across all folds):

```bash
bash predict_pcp.sh \
  ./MD_300/density/homopolymer_density/random_split \
  ../../Datasets/wPSMILES/test_data.csv \
  predictions.csv
```

**Note**: The checkpoint directory should contain all `fold_*` subdirectories. The model will automatically perform ensemble prediction across all folds.

## Input Format

Polymer-chemprop expects **wPSMILES** format:

```csv
smiles,property
[*:1]C[*:2].[*:3]C(C)[*:4]|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5...,1.234
```

**Components**:
1. **Monomer SMILES**: `[*:1]C[*:2].[*:3]C(C)[*:4]` - Numbered attachment points
2. **Stoichiometry**: `|0.5|0.5|` - Relative abundance of each monomer
3. **Connectivity**: `<1-3:0.5:0.5<1-4:0.5:0.5...` - Bond probabilities between attachment points

See `Datasets/README.md` for full format details.

## Model Architecture

- **Input**: wPSMILES polymer representation
- **Model**: Message Passing Neural Network with:
  - Weighted edges (bond probabilities)
  - Directed message passing
  - Degree of polymerization weighting
- **Output**: Continuous property prediction

## Citation

If you use this model, please cite:

```bibtex
@article{aldeghi2022graph,
  title={A graph representation of molecular ensembles for polymer property prediction},
  author={Aldeghi, Matteo and Coley, Connor W},
  journal={Chemical Science},
  volume={13},
  number={35},
  pages={10486--10498},
  year={2022},
  publisher={Royal Society of Chemistry}
}
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'chemprop'"
Make sure you've activated the virtual environment:
```bash
source .venv/bin/activate
```

### "ERROR: Virtual environment not found"
Run the setup script first:
```bash
bash setup_environment.sh
```

### "Dataset not found"
Ensure you've generated the wPSMILES datasets:
```bash
cd ../../Datasets/Dataset_construction_scripts
bash generate_basic_datasets.sh
```

### Out of memory during training
- Reduce batch size by adding to `train_pcp.sh`: `--batch_size 32`
- Use fewer workers: `--num_workers 0` (already set)
- Train on smaller datasets first

## Advanced Usage

### Custom Hyperparameters

Edit `train_pcp.sh` to modify training parameters:

```bash
chemprop_train \
  --data_path "$DATA_PATH" \
  --dataset_type regression \
  --epochs 100 \              # More epochs
  --hidden_size 600 \          # Larger model
  --depth 4 \                  # Deeper network
  --dropout 0.1 \              # Add dropout
  --polymer \
  ...
```

See [Chemprop documentation](https://chemprop.readthedocs.io/) for all available parameters.

### Hyperparameter Optimization

For automatic hyperparameter tuning:

```bash
source .venv/bin/activate
chemprop_hyperopt \
  --data_path ../../Datasets/wPSMILES/dataset.csv \
  --dataset_type regression \
  --num_iters 20 \
  --config_save_path best_hyperparams.json \
  --polymer
```

## System Requirements

### Minimum
- Python 3.10+
- 4GB RAM
- 2GB disk space

### Recommended
- Python 3.10+
- 8GB RAM
- GPU with CUDA support (for faster training)
- 5GB disk space
