# OpenAI Model for Polymer Property Prediction

This directory contains scripts and example outputs for predicting homopolymer properties using OpenAI large language models (LLMs). The implementation focuses on predicting electron affinity (EA) and ionization potential (IP) for polymers represented as PSMILES strings.

## Setup

Before using the scripts, you need to set up the Python environment:

```bash
./setup_environment.sh
```

This script will:
1. Check for Python 3.10+ requirement
2. Create a virtual environment in `.venv/`
3. Install all required dependencies from `requirements.txt`

## Environment Variables

The scripts require an OpenAI API key, which should be stored in a file named `OPENAI_KEY.env` in the root directory with the following format:

```
MY_SECRET_KEY=your_openai_api_key
```

## Directory Structure

- `EA_predictions/`: Scripts and outputs for predicting electron affinity
  - Python scripts for different prediction approaches (`homopolymer_EA_*.py`)
  - JSON output files containing model responses and predictions
- `IP_predictions/`: Scripts and outputs for predicting ionization potential
  - Python scripts for different prediction approaches (`homopolymer_IP_*.py`) 
  - JSON output files containing model responses and predictions

## Available Prediction Scripts

### Electron Affinity (EA) Prediction

- `EA_predictions/homopolymer_EA_4o_neg.py`: Prediction using GPT-4o without referencing common names
- `EA_predictions/homopolymer_EA_4o_pos.py`: Prediction using GPT-4o with common names
- `EA_predictions/homopolymer_EA_o3_neg.py`: Prediction using older model without common names
- `EA_predictions/homopolymer_EA_o3_pos.py`: Prediction using older model with common names

### Ionization Potential (IP) Prediction

- `IP_predictions/homopolymer_IP_4o_neg.py`: Prediction using GPT-4o without referencing common names
- `IP_predictions/homopolymer_IP_4o_pos.py`: Prediction using GPT-4o with common names
- `IP_predictions/homopolymer_IP_o3_neg.py`: Prediction using older model without common names
- `IP_predictions/homopolymer_IP_o3_pos.py`: Prediction using older model with common names

## Usage

To run a prediction script, first activate the virtual environment:

```bash
source .venv/bin/activate
```

Then execute one of the Python scripts:

```bash
python EA_predictions/homopolymer_EA_4o_neg.py
```

Each script will:
1. Load PSMILES strings from corresponding datasets
2. Send prompts to the OpenAI API asking for property predictions
3. Parse and extract numerical values from the model's responses
4. Save full model responses in JSON format

## Output Files

The scripts generate JSON output files with the naming convention:
- `full_reply_homopolymer_EA_4o_neg_0.json`: Full model responses for EA predictions
- `full_reply_homopolymer_IP_4o_neg_0.json`: Full model responses for IP predictions

Each JSON file maps PSMILES strings to model responses, which include the reasoning process and final predicted value.

## Model Variations

The scripts demonstrate different approaches to property prediction:
- Using GPT-4o (`4o`) vs older models (`o3`) 
- Including polymer common names in the prompts (`pos`) vs excluding them (`neg`)

This allows for comparison of different prompt engineering strategies and model capabilities for polymer property prediction.

## Dataset Sources

The scripts use datasets from:
- `./../../Datasets/PSMILES/Polymer_Genome/electron_affinity/electron_affinity_data_polymers_v4.csv`
- `./../../Datasets/PSMILES/Polymer_Genome/ionization_energy/ionization_energy_data_polymers_v4.csv`

## Requirements

The main dependencies for running these scripts are:
- Python 3.10+
- pandas
- numpy
- litellm (for API interaction)
- dotenv (for loading API keys)
- regex (for extracting values from responses)

For a full list of dependencies, see `requirements.txt`.