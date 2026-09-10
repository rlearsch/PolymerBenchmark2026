# RDKit_RF Model

Random Forest baseline model using RDKit molecular descriptors for polymer property prediction.

## Overview

This model provides a simple, interpretable baseline for polymer property prediction using:
- **200+ RDKit molecular descriptors** (MW, LogP, TPSA, aromatic rings, etc.)
- **Random Forest Regressor** with median imputation for missing values
- **5-fold cross-validation** with consistent 80/10/10 train/val/test splits

## Prerequisites

- Python 3.10 or higher
- ~500MB disk space for dependencies
- Generated RDKit descriptor datasets (see `Datasets/README.md`)

## Setup

### One-Time Setup

```bash
cd Models/RDKit_RF
bash setup_environment.sh
```

This will:
1. Create a virtual environment (`.venv/`)
2. Install scikit-learn, RDKit, pandas, numpy
3. Install all dependencies

### Manual Setup (Alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

Train a Random Forest model on any RDKit descriptor dataset:

```bash
bash train_rf.sh ../../Datasets/RDKit_descriptors/MD_300/density/homopolymer_density.csv
```

**Training parameters** (in train_rf.py):
- 5-fold cross-validation
- 100 trees per forest
- Median imputation for missing values
- Random seed: 42 + fold number

**Output structure**:
```
Models/RDKit_RF/
└── MD_300/density/homopolymer_density/random_split/
    ├── feature_names.json         # List of features used
    ├── overall_metrics.json        # Aggregate results
    ├── fold_0/
    │   ├── rf.joblib               # Trained model
    │   ├── metrics.json            # Train/val/test scores
    │   └── split_info.json         # Split sizes and seed
    ├── fold_1/
    └── ...
```

### Prediction

Make predictions using trained models (ensemble across all folds):

```bash
bash predict_rf.sh \
  ./MD_300/density/homopolymer_density/random_split \
  ../../Datasets/RDKit_descriptors/test_data.csv \
  predictions.csv
```

**Note**: The checkpoint directory should contain all `fold_*` subdirectories. The model will automatically perform ensemble prediction across all folds.

**Output**: CSV with original data plus prediction columns:
- `pred_fold_0`, `pred_fold_1`, ... - Individual fold predictions
- `pred_mean` - Ensemble mean
- `pred_std` - Ensemble standard deviation

## Input Format

RDKit_RF expects **RDKit descriptor CSVs** from `Datasets/RDKit_descriptors/`:

```csv
MolWt,LogP,TPSA,NumHDonors,NumHAcceptors,...,property
150.13,2.45,29.46,1,2,...,1.234
```

These datasets contain:
- **200+ descriptor columns**: Molecular properties calculated by RDKit
- **1 target column**: The property to predict (e.g., density, Tg)

## Model Details

### Architecture

```
Input: RDKit Descriptors (200+ features)
  ↓
Preprocessing:
  - Drop problematic columns (partial charges, Ipc)
  - Replace inf/nan values
  ↓
Imputation: Median strategy
  ↓
Random Forest Regressor:
  - 100 estimators
  - Default max_depth (unlimited)
  - All CPU cores (-1)
  ↓
Output: Continuous property prediction
```

### Advantages

- **Fast training**: ~1-5 minutes per property
- **Interpretable**: Feature importances available
- **Robust**: Handles missing values and outliers well
- **No GPU required**: Runs on any machine
- **Proven baseline**: Standard approach in molecular ML

### Limitations

- **Feature engineering**: Requires pre-computed descriptors
- **Less expressive**: Cannot learn complex molecular patterns like GNNs
- **Fixed representations**: Cannot adapt features to task

## Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn'"
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
Ensure you've generated the RDKit descriptor datasets:
```bash
cd ../../Datasets/Dataset_construction_scripts
bash generate_rdkit_datasets.sh
```

### Out of memory during training
- Use smaller datasets (MD_300 instead of MD_5000)
- Reduce number of trees: edit `train_rf.py` and change `n_estimators=100` to `n_estimators=50`

### Poor performance
Random Forest is a baseline model. For better performance, consider:
- **polymer_chemprop**: Graph neural network on wPSMILES
- **polyBERT**: Transfer learning with pre-trained embeddings

## Advanced Usage

### Custom Hyperparameters

Edit `train_rf.py` to modify Random Forest parameters:

```python
RandomForestRegressor(
    n_estimators=200,        # More trees
    max_depth=20,            # Limit depth
    min_samples_split=10,    # Regularization
    max_features='sqrt',     # Feature sampling
    random_state=random_state,
    n_jobs=-1
)
```

### Feature Importance Analysis

After training, inspect feature importances:

```python
from joblib import load
import numpy as np

model = load('./MD_300/density/homopolymer_density/random_split/fold_0/rf.joblib')
rf = model.named_steps['rf']
importances = rf.feature_importances_

# Get top 10 features
indices = np.argsort(importances)[::-1][:10]
print("Top 10 features:")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")
```

## System Requirements

### Minimum
- Python 3.10+
- 2GB RAM
- 500MB disk space

### Recommended
- Python 3.10+
- 4GB RAM (for large datasets)
- SSD for faster data loading

## Citation

This is a baseline Random Forest model using standard RDKit descriptors. If you use RDKit, please cite:

```bibtex
@misc{rdkit,
  author = {Greg Landrum and others},
  title = {RDKit: Open-Source Cheminformatics Software},
  howpublished = {\url{https://www.rdkit.org}},
  note = {Version 2020.03.1}
}
```

For the benchmark, cite the Polymer Bench 2026 (PolyBench26) paper (see the main README).
