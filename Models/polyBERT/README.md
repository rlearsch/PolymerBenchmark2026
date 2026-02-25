# polyBERT Model

Feed-forward neural network for polymer property prediction using pre-trained polyBERT embeddings.

## Overview

This model trains a deep feed-forward neural network on 600-dimensional polyBERT embeddings for polymer property prediction. polyBERT provides transfer learning capabilities by leveraging pre-trained representations of polymer structures.

**Key Features:**
- **Transfer learning**: Leverages pre-trained polyBERT embeddings
- **Fast training**: ~5-15 minutes per property (no embedding computation needed)
- **5-fold cross-validation** with 80/10/10 train/val/test splits
- **Reproducible**: Uses seed-based splitting (default seed: 42)

## Reference

**polyBERT Model**: [Kuenneth et al., 2023](https://huggingface.co/kuelumbus/polyBERT)  
**Note**: Embeddings must be pre-computed using the dataset generation scripts in `Datasets/Dataset_construction_scripts/`

## Prerequisites

- Python 3.10 or higher
- ~1GB disk space for dependencies
- Generated polyBERT embedding datasets (see `Datasets/README.md`)
- ~16GB RAM recommended for large datasets

## Setup

### One-Time Setup

```bash
cd Models/polyBERT
bash setup_environment.sh
```

This will:
1. Create a virtual environment (`.venv/`)
2. Install PyTorch, numpy, pandas
3. Install all dependencies

### Manual Setup (Alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

Train a model on any polyBERT embedding dataset:

```bash
bash train_pBERT.sh ../../Datasets/polyBERT/MD_300/density/homopolymer_density.csv
```

**Training parameters** (can be customized in train_pBERT.sh):
- 5-fold cross-validation
- 50 epochs per fold
- Batch size: 50
- Base random seed: 42 (fold i uses seed 42+i)
- Architecture: 3-layer feed-forward network with ReLU activations

**Output structure**:
```
Models/polyBERT/
└── MD_300/density/homopolymer_density/random_split/
    ├── fold_00_seed_42/
    │   ├── checkpoint_best.pt      # Best model checkpoint
    │   ├── history.json            # Training history
    │   └── args.json               # Training configuration
    ├── fold_01_seed_43/
    └── ...
```

### Prediction

Make ensemble predictions using trained checkpoints (averages across all folds):

```bash
bash predict_pBERT.sh \
  ./MD_300/density/homopolymer_density/random_split \
  ../../Datasets/polyBERT/test_data.csv \
  predictions.csv
```

**Note**: The checkpoint directory should contain all `fold_*` subdirectories with `checkpoint_best.pt` files. The script automatically finds and uses all available checkpoints for ensemble prediction.

**Output**: CSV with columns:
- `orig_index` - Original row index
- `y_true` - True value (if present in test data)
- `y_pred_ens` - Ensemble prediction (averaged across folds)

## Input Format

polyBERT model expects **polyBERT embedding CSVs** from `Datasets/polyBERT/`:

```csv
emb_0,emb_1,emb_2,...,emb_599,property
0.123,-0.456,0.789,...,0.321,1.234
```

These datasets contain:
- **600 embedding columns**: Pre-computed polyBERT embeddings (columns 0-599)
- **1 target column**: The property to predict (column 600)

## Model Architecture

```
Input: polyBERT Embeddings (600 features)
  ↓
Layer 1: Linear(600 → 300) + ReLU + Dropout(0.0)
  ↓
Layer 2: Linear(300 → 300) + ReLU + Dropout(0.0)
  ↓
Layer 3: Linear(300 → 300) + ReLU + Dropout(0.0)
  ↓
FFN Layer 1: Linear(300 → 300) + ReLU + Dropout(0.0)
  ↓
FFN Layer 2: Linear(300 → 300) + ReLU + Dropout(0.0)
  ↓
Output: Linear(300 → 1)
```

**Training details:**
- **Optimizer**: Adam with cosine annealing learning rate schedule
- **Initial LR**: 1e-4
- **Max LR**: 1e-3 (reached after warmup)
- **Final LR**: 1e-4
- **Warmup**: 2 epochs
- **Loss**: MSE for regression
- **Normalization**: Per-fold feature standardization (mean=0, std=1)

## Random Seeds

polyBERT uses **base seed 42** for reproducibility:
- Fold 0: seed = 42
- Fold 1: seed = 43
- Fold 2: seed = 44
- Fold 3: seed = 45
- Fold 4: seed = 46

This matches the seeding pattern used by RDKit_RF for consistency across models.

**Note**: While seeds are consistent across models, the actual train/val/test splits will differ between models due to different RNG libraries (PyTorch vs sklearn). This is normal and acceptable for benchmarking.

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
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
Ensure you've generated the polyBERT embedding datasets:
```bash
cd ../../Datasets/Dataset_construction_scripts
bash generate_polybert_datasets.sh
```

This requires the polyBERT model to be downloaded (see main README).

### Out of memory during training
- Reduce batch size: Edit `train_pBERT.sh` and change `--batch_size 50` to `--batch_size 32`
- Use smaller datasets (MD_300 instead of MD_5000)
- Close other applications

### Training is slow
- Ensure PyTorch can access GPU if available
- Reduce number of epochs for testing: `--epochs 10`
- polyBERT is faster than polymer_chemprop but slower than RDKit_RF

## Advanced Usage

### Custom Hyperparameters

Modify training parameters by editing the arguments in `train_pBERT.sh`:

```bash
python train_pBERT.py \
  --data_path "$DATA_PATH" \
  --save_dir "$SAVE_DIR" \
  --num_folds 10 \          # More folds
  --epochs 100 \            # More epochs
  --batch_size 32 \         # Smaller batches
  --seed 42                 # Custom seed
```

Or call Python directly with custom config:

```bash
source .venv/bin/activate
python train_pBERT.py \
  --data_path ../../Datasets/polyBERT/MD_300/density/homopolymer_density.csv \
  --save_dir ./custom_run \
  --num_folds 5 \
  --epochs 50 \
  --batch_size 50 \
  --seed 42
```

### Modifying Model Architecture

Edit the `config` dictionary in `train_pBERT.py`:

```python
config = {
    'hidden_size': 512,        # Larger hidden layers
    'ffn_hidden_size': 512,    # Larger FFN
    'ffn_num_layers': 3,       # More FFN layers
    'depth': 4,                # Deeper network
    'dropout': 0.1,            # Add dropout
    'bias': True,              # Use bias terms
    # ... other parameters
}
```

### Learning Rate Schedule

Modify the learning rate schedule in `train_pBERT.py`:

```python
config = {
    'init_lr': 1e-5,      # Lower initial LR
    'max_lr': 5e-4,       # Lower max LR
    'final_lr': 1e-5,     # Lower final LR
    'warmup_epochs': 5.0, # Longer warmup
    # ...
}
```

## Performance Expectations

**Training time** (approximate, CPU-only):
- MD_300 datasets (~300 samples): ~2 minutes
- MD_5000 datasets (~5000 samples): ~5-10 minutes
- Large datasets (>10k samples): ~15 minutes

**With GPU**: 2-5x faster

**Memory usage**:
- Small datasets: ~2GB RAM
- Large datasets: ~8-16GB RAM


## Citation

If you use polyBERT embeddings, please cite:

```bibtex
@article{kuenneth2023polybert,
  title={polyBERT: A chemical language model for polymer property prediction},
  author={Kuenneth, Christopher and Ramprasad, Rampi},
  journal={...},
  year={2023}
}
```

For the benchmark, cite the OPoly26Benchmark paper (see main README).

## System Requirements

### Minimum
- Python 3.10+
- 4GB RAM
- 1GB disk space

### Recommended
- Python 3.10+
- 16GB RAM
- GPU with CUDA support (for faster training)
- 2GB disk space
