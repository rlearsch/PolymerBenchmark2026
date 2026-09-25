# Models

This directory contains implementations of various machine learning models for polymer property prediction, evaluated in Polymer Benchmark 2026 (PolyBench26).

## Available Models

| Model | Input Format | Type | Status | Description |
|-------|--------------|------|--------|-------------|
| **polymer_chemprop** | wPSMILES | GNN | ✅ Ready | Weighted directed MPNN for polymers |
| **polyBERT** | polyBERT embeddings | Feed-forward NN | ✅ Ready | Transfer learning with polyBERT |
| **RDKit_RF** | RDKit descriptors | Random Forest | ✅ Ready | Traditional ML baseline |
| **polymer_periodic_graph** | PSMILES | Periodic GNN | ✅ Ready | Periodic graph neural network |
| **OpenAI** | PSMILES | LLM | ✅ Ready | Zero-shot LLM property prediction |

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

# 4. Make predictions (ensemble across all folds)
bash predict_pcp.sh ./MD_300/density/homopolymer_density/random_split test_data.csv predictions.csv
```

## Model Descriptions

### polymer_chemprop
- **Paper**: Aldeghi & Coley, Chem. Sci. 2022
- **Input**: wPSMILES (weighted polymer SMILES)
- **Architecture**: Weighted, directed message passing neural network
- **Best for**: Copolymers with complex architectures
- **Training time**: ~10-30 minutes per property
- **Dependency**: Installed from GitHub (no vendored code)

### polyBERT
- **Paper**: Kuenneth et al., 2023
- **Input**: 600-dimensional polyBERT embeddings
- **Architecture**: Feed-forward neural network on pre-trained embeddings
- **Best for**: Transfer learning, limited data scenarios
- **Training time**: ~5-15 minutes per property
- **Note**: Requires pre-computed embeddings from Datasets/

### RDKit_RF
- **Input**: 200+ RDKit molecular descriptors
- **Architecture**: Random Forest ensemble (100 trees)
- **Best for**: Baseline comparisons, interpretability, fast training
- **Training time**: ~1-5 minutes per property
- **Note**: Uses pre-computed descriptors from Datasets/

### polymer_periodic_graph
- **Input**: PSMILES (Polymer SMILES)
- **Architecture**: Message passing neural network with periodic graph representation
- **Best for**: Capturing repeating structure of homopolymers and alternating copolymers
- **Training time**: ~10-30 minutes per property
- **Dependency**: Uses a customized local version of chemprop

### OpenAI
- **Input**: PSMILES (Polymer SMILES)
- **Architecture**: Large language model with specialized prompts
- **Best for**: Zero-shot prediction with explanation and reasoning
- **Prediction time**: ~30-60 seconds per polymer
- **Advantage**: No training required, provides detailed chemical reasoning
- **Note**: Requires OpenAI API key

## Dataset Compatibility

| Model | PSMILES | wPSMILES | RDKit Descriptors | polyBERT Embeddings |
|-------|---------|----------|-------------------|---------------------|
| polymer_chemprop | ❌ | ✅ | ❌ | ❌ |
| polyBERT | ❌ | ❌ | ❌ | ✅ |
| RDKit_RF | ❌ | ❌ | ✅ | ❌ |
| polymer_periodic_graph | ✅ | ❌ | ❌ | ❌ |
| OpenAI | ✅ | ❌ | ❌ | ❌ |

## Cross-Validation and Random Seeds

All models use **5-fold cross-validation** with **80/10/10 train/val/test splits** for consistency.

### Random Seed Strategy

For reproducibility, models use a base seed of **42**:
- **RDKit_RF**: Uses seeds 42, 43, 44, 45, 46 for folds 0-4
- **polyBERT**: Uses seeds 42, 43, 44, 45, 46 for folds 0-4
- **polymer_chemprop**: Uses pytorch_seed=0 (chemprop internal)
- **polymer_periodic_graph**: Uses pytorch_seed=0 (chemprop internal)

**Important Notes:**
- ✅ Each model's results are fully reproducible (same seed → same splits within that model)
- ⚠️ Splits differ between models due to different RNG libraries (sklearn vs torch vs chemprop)
- ✅ Multiple folds average out random variations in splitting

### Customizing Seeds

Both RDKit_RF and polyBERT support custom seeds:

```bash
# RDKit_RF
python train_rf.py --data_path data.csv --save_dir output/ --seed 123

# polyBERT  
python train_pBERT.py --data_path data.csv --save_dir output/ --seed 123
```

## Performance Comparison

| Model | Training Speed | Memory | GPU Benefit | Interpretability |
|-------|----------------|--------|-------------|------------------|
| **RDKit_RF** | Fastest (1-5 min) | Low (2-4GB) | No | High (feature importance) |
| **polyBERT** | Fast (5-15 min) | Medium (4-16GB) | Yes (2-5x) | Low |
| **polymer_chemprop** | Moderate (10-30 min) | Medium (4-16GB) | Yes (2-5x) | Low |
| **polymer_periodic_graph** | Moderate (10-30 min) | Medium (4-16GB) | Yes (2-5x) | Low |
| **OpenAI** | N/A (zero-shot) | Low (API-based) | N/A | High (text explanations) |


## Common Setup Pattern

All models follow this structure:

```
Models/<model_name>/
├── README.md              # Model documentation
├── requirements.txt       # Python dependencies
├── setup_environment.sh   # Environment setup
├── train_*.py             # Training script (with argparse CLI)
├── train_*.sh             # Training wrapper
├── predict_*.py           # Prediction script (if applicable)
├── predict_*.sh           # Prediction wrapper
├── .venv/                 # Virtual environment (excluded from git)
└── <dataset>/             # Training outputs (excluded from git)
    └── <property>/
        └── random_split/
            ├── fold_0/    # or fold_00_seed_42 (polyBERT)
            ├── fold_1/
            └── ...
```

## Prerequisites

- Python 3.10 or higher
- Generated datasets (see `Datasets/README.md`)
- ~5-10GB disk space for all models
- ~2-4GB disk space per model for dependencies

## Generating Datasets

If you haven't generated datasets yet:

```bash
cd ../Datasets/Dataset_construction_scripts

# Setup dataset generation environments (one-time)
bash setup_environments.sh

# Generate datasets (choose what you need)
bash generate_basic_datasets.sh      # PSMILES + wPSMILES (~15 min)
bash generate_rdkit_datasets.sh      # RDKit descriptors (initial cache fill ~30 min)
bash generate_polybert_datasets.sh   # polyBERT embeddings (~1-3 hours)
```

**Note**: You only need to generate datasets for the models you plan to use:
- **polymer_chemprop**: Requires `generate_basic_datasets.sh` (wPSMILES)
- **RDKit_RF**: Requires `generate_rdkit_datasets.sh` (RDKit descriptors)
- **polyBERT**: Requires `generate_polybert_datasets.sh` (polyBERT embeddings)

## Architecture Overview

### polymer_chemprop (Graph Neural Network)
```
Input: wPSMILES → Graph Representation
  ↓
Message Passing Layers (weighted, directed edges)
  ↓
Graph Aggregation
  ↓
Feed-forward Network
  ↓
Output: Property Prediction
```

### polyBERT (Transfer Learning)
```
Input: Pre-computed polyBERT Embeddings (600-dim)
  ↓
3 × [Linear → ReLU → Dropout]
  ↓
2 × [Feed-forward → ReLU → Dropout]
  ↓
Output Layer
  ↓
Output: Property Prediction
```

### RDKit_RF (Traditional ML)
```
Input: RDKit Descriptors (200+ features)
  ↓
Median Imputation (missing values)
  ↓
Random Forest (100 trees)
  ↓
Output: Property Prediction
```

### OpenAI (Large Language Model)
```
Input: PSMILES + Prompt Template
  ↓
API Request to OpenAI LLM
  ↓
Model Response with Reasoning
  ↓
Regex Extraction of Predicted Value
  ↓
Output: Property Prediction + Explanation
```

## Citation

For PolyBench26 citation guidance, use the repository-level instructions in
[`../README.md`](../README.md). The first public release will include final
machine-readable citation metadata.

And cite the individual model papers as appropriate:

**polymer_chemprop:**
```bibtex
@article{aldeghi2022graph,
  title={A graph representation of molecular ensembles for polymer property prediction},
  author={Aldeghi, Matteo and Coley, Connor W},
  journal={Chemical Science},
  volume={13},
  number={35},
  pages={10486--10498},
  year={2022}
}
```

**polyBERT:**
```bibtex
@article{kuenneth2023polybert,
  title={polyBERT: A chemical language model for polymer property prediction},
  author={Kuenneth, Christopher and Ramprasad, Rampi},
  journal={...},
  year={2023}
}
```

**RDKit:**
```bibtex
@misc{rdkit,
  author = {Greg Landrum and others},
  title = {RDKit: Open-Source Cheminformatics Software},
  howpublished = {\url{https://www.rdkit.org}}
}
```

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
bash generate_basic_datasets.sh  # or the appropriate generation script
```

### Out of Memory
- Start with smaller datasets (MD_300 instead of MD_5000)
- Reduce batch sizes in training scripts:
  - polymer_chemprop: Add `--batch_size 32` to train_pcp.sh
  - polyBERT: Change `--batch_size 50` to `--batch_size 32` in train_pBERT.sh
  - RDKit_RF: Reduce `n_estimators` in train_rf.py
- Close other applications

### Training is Too Slow
- **RDKit_RF**: Already fast, no optimization needed
- **polyBERT/polymer_chemprop**: 
  - Use GPU if available (PyTorch will auto-detect)
  - Reduce epochs for testing: `--epochs 10`
  - Use smaller datasets for prototyping (MD_300)

### "ModuleNotFoundError" Errors
Make sure you've run the setup script and activated the environment:
```bash
cd Models/<model_name>
bash setup_environment.sh
source .venv/bin/activate
```

### Inconsistent Results Across Models
This is expected! While all models use the same seed philosophy (base seed 42), the actual train/test splits differ due to different RNG libraries. This is normal and acceptable for benchmarking. The key is that:
- ✅ Each model is internally consistent (reproducible splits)
- ✅ All use 5-fold cross-validation
- ✅ All use 80/10/10 train/val/test splits
- ✅ Multiple folds average out splitting variance

## Development Notes

### Dependency Management
- **polymer_chemprop**: Installed from GitHub (`git+https://...`) with pinned commit hash
- **RDKit_RF**: Standard PyPI packages (sklearn, rdkit, pandas, numpy)
- **polyBERT**: Standard PyPI packages (torch, numpy, pandas)

### Code Organization
All models follow clean CLI patterns:
- Argparse for all Python scripts
- Shell wrappers for easy usage
- No hardcoded paths
- Proper error handling and validation
- Timing information

### Adding New Models
To add a new model to the benchmark:

1. Create directory: `Models/new_model/`
2. Add required files:
   - `README.md` - Documentation
   - `requirements.txt` - Dependencies
   - `setup_environment.sh` - Environment setup
   - `train_*.py` - Training script with argparse
   - `train_*.sh` - Training wrapper
   - `predict_*.py` - Prediction script (optional)
   - `predict_*.sh` - Prediction wrapper
3. Follow seeding convention: base_seed=42, fold_i uses seed=42+i
4. Use 5-fold CV with 80/10/10 splits
5. Update this README

## Contributing

This benchmark is provided to reproduce research results. For issues or questions, please open a GitHub issue.
