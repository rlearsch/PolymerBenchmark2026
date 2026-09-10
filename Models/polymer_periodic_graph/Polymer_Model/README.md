# polymer_periodic_graph

A message-passing neural network model for polymer property prediction that treats polymers as periodic graphs.

## Overview

The polymer_periodic_graph model is a customized version of chemprop that represents polymers as periodic graphs to better capture their repeating structure. By treating polymer chains as periodic graphs, this approach can more accurately model the long-range interactions and structural patterns that affect polymer properties.

## Features

- Treats polymers as periodic graphs to model repeating units
- Uses message-passing neural networks (MPNN) for molecular representation
- Supports regression tasks for various polymer properties
- Includes 5-fold cross-validation for robust evaluation
- Leverages a customized version of chemprop designed specifically for polymers

## Reference Paper

[Paper information to be filled in]

## Prerequisites

- Python 3.10 or higher
- PyTorch 1.5.1 or higher
- pandas, numpy, scikit-learn
- typed-argument-parser (TAP)

## Setup

### One-Time Setup

```bash
# Clone the repository (if not already done)
git clone https://github.com/[username]/PolyBench26.git
cd PolyBench26/Models/polymer_periodic_graph

# Set up the virtual environment
bash setup_environment.sh
```

### Manual Setup

If you prefer to set up the environment manually:

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Make scripts executable
chmod +x train_ppg.sh predict_ppg.sh
```

## Usage

### Training a Model

```bash
# Activate the virtual environment
source .venv/bin/activate

# Basic usage
bash train_ppg.sh <path_to_dataset.csv>

# Example
bash train_ppg.sh ../../Datasets/PSMILES/MD_300/density/homopolymer_density.csv
```

The training script will:
1. Extract the dataset name and property from the provided path
2. Create an appropriate output directory
3. Train the model with 5-fold cross-validation
4. Save model checkpoints for each fold

### Making Predictions

```bash
# Activate the virtual environment
source .venv/bin/activate

# Basic usage
bash predict_ppg.sh <checkpoint_dir> <test_data.csv> <output.csv>

# Example
bash predict_ppg.sh ./MD_300/density/homopolymer_density/random_split/ test.csv predictions.csv
```

## Input Format

The model expects CSV files with PSMILES representations of polymers. The training data should have the following format:

```
smiles,target_property
[*]CC[*],0.954
[*]CCCC[*],0.912
...
```

Where:
- `smiles` contains the PSMILES representation of the polymer
- `target_property` is the numeric property value to predict

## Model Architecture

polymer_periodic_graph uses a message-passing neural network (MPNN) architecture with the following key components:

1. **Periodic Graph Representation**: Polymers are represented as graphs with periodic connections to model their repeating structure.
2. **Message Passing Phase**: Information is passed between atoms and bonds to build molecular representations.
3. **Readout Phase**: The final molecular representation is used to predict properties through feed-forward layers.

## Random Seed

This model uses a fixed random seed (pytorch_seed=0) to ensure reproducibility. Seeds are aligned with the overall benchmark suite's strategy:

- Base seed: 0 (used in polymer_chemprop)
- Each fold uses the same seed

## Troubleshooting

### Common Issues and Solutions

1. **Error: Dataset not found**
   - Ensure the path to the dataset is correct
   - Verify the dataset exists at the specified location

2. **Error: Virtual environment not found**
   - Run `bash setup_environment.sh` to create the virtual environment

3. **ModuleNotFoundError: No module named 'chemprop'**
   - Ensure you've activated the virtual environment with `source .venv/bin/activate`
   - If the error persists, try reinstalling the dependencies with `pip install -r requirements.txt`

4. **Memory issues during training**
   - Reduce batch size by modifying the `--batch_size` parameter in `train_ppg.sh`
   - Use fewer folds with `--num_folds 3` instead of the default 5

5. **CUDA-related errors**
   - Verify CUDA installation with `python -c "import torch; print(torch.cuda.is_available())"`
   - If GPU is available but not detected, try setting `CUDA_VISIBLE_DEVICES=0` before running

## Advanced Usage

### Customizing Training Parameters

You can modify the `train_ppg.sh` script to customize training parameters:

```bash
python Polymer_Model/train.py \
  --data_path "$DATA_PATH" \
  --dataset_type regression \
  --pytorch_seed 0 \
  --save_dir "$SAVE_DIR" \
  --quiet \
  --epochs 100 \  # Increase number of epochs
  --batch_size 32 \  # Add batch size parameter
  --num_folds 5 \
  --num_workers 4  # Increase worker threads
```

### Using Different Fingerprint Types

The model supports various fingerprinting methods for molecular representation:

```bash
python Polymer_Model/train.py \
  --data_path "$DATA_PATH" \
  --dataset_type regression \
  --features_generator morgan \  # Use Morgan fingerprints
  --features_size 2048 \  # Specify fingerprint size
  --save_dir "$SAVE_DIR"
```

## Performance Expectations

- **Training time**: Approximately 30-60 minutes per dataset on a modern CPU
- **GPU acceleration**: Training is 5-10x faster with a GPU
- **Prediction time**: Less than 1 minute for most test sets
- **Memory usage**: 2-4 GB RAM during training
- **Disk usage**: 10-50 MB per model checkpoint

## Comparison with Other Models

| Model | Unique Features | Input Format | Strengths | Limitations |
|-------|----------------|--------------|-----------|-------------|
| polymer_periodic_graph | Periodic graph representation | PSMILES | Better captures polymer repetition | More complex model |
| polymer_chemprop | Standard graph with polymer adaptation | wPSMILES | Simpler implementation | Less accurate for some properties |
| RDKit_RF | Molecular descriptors | RDKit descriptors | Faster training | Less structural insight |
| polyBERT | Natural language model for polymers | polyBERT | Better for diverse structures | Needs more training data |

## Citation

If you use this model in your research, please cite:

```
[Citation information to be filled in]
```

## System Requirements

- **Minimum**: 8GB RAM, 2 CPU cores, 500MB disk space
- **Recommended**: 16GB RAM, 4+ CPU cores, GPU with 4GB+ VRAM
- **Operating System**: Linux, macOS, or Windows with WSL

## Contact

For questions about this model, please open an issue on the GitHub repository.

## License

This project is licensed under the [License information].
