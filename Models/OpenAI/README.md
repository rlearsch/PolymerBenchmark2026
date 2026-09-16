# Archived OpenAI Polymer-Prediction Examples

This directory preserves the prompt scripts and full responses from historical
LLM experiments predicting homopolymer electron affinity (EA) and ionization
potential (IP) from PSMILES strings. It is included so readers can inspect what
was asked, how prompt variants differed, and the responses used in the
original analysis.

## Archival status

These files are **archived examples, not a reproducible benchmark workflow**.
The original requests used an LLNL-specific OpenAI-compatible endpoint that is
not generally available to public users. In addition, hosted-model outputs can
change over time. Do not treat a rerun as a reconstruction of the recorded
responses, and do not run these scripts as part of the repository's routine
validation.

The tracked `full_reply_*.json` files are the authoritative historical record.
They contain the complete responses keyed by PSMILES; they are provided for
transparency and qualitative inspection, rather than as a claim that the API
calls can be exactly repeated.

## Historical environment

The scripts are retained in their original form. They require an environment
with the dependencies listed in `requirements.txt`:

```bash
./setup_environment.sh
```

This script will:
1. Check for Python 3.10+ requirement
2. Create a virtual environment in `.venv/`
3. Install all required dependencies from `requirements.txt`

## Endpoint and credentials

The historical scripts load a key from a local file named `OPENAI_KEY.env`:

```
MY_SECRET_KEY=your_openai_api_key
```

`OPENAI_KEY.env` is intentionally ignored by Git. The scripts are configured
for the LLNL endpoint recorded in their source code. Access to that endpoint is
not assumed or provided by this repository.

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

## Inspecting the archived record

No API access is required to read the tracked JSON files. Each maps a PSMILES
string to the complete model response. The `*_pos` and `*_neg` names indicate
whether the prompt requested a common/English polymer name; `4o` and `o3`
identify the historical model families used in the scripts.

## Historical script behavior

If you have authorized access to the original endpoint, the scripts can be
examined or adapted at your own discretion:

```bash
source .venv/bin/activate
```

For example:

```bash
python EA_predictions/homopolymer_EA_4o_neg.py
```

Each script was designed to:
1. Load PSMILES strings from corresponding datasets
2. Send prompts to the OpenAI API asking for property predictions
3. Parse and extract numerical values from the model's responses
4. Save full model responses in JSON format

## Archived output files

The scripts generate JSON output files with the naming convention:
- `full_reply_homopolymer_EA_4o_neg_0.json`: Full model responses for EA predictions
- `full_reply_homopolymer_IP_4o_neg_0.json`: Full model responses for IP predictions

Each JSON file maps PSMILES strings to model responses, including the model's
reasoning and usually a final predicted value. The wording and numeric format
of a response are model-generated, so consumers should validate any parsed
values for their own analysis.

## Model Variations

The scripts demonstrate different approaches to property prediction:
- Using GPT-4o (`4o`) vs older models (`o3`) 
- Including polymer common names in the prompts (`pos`) vs excluding them (`neg`)

This allows for comparison of different prompt engineering strategies and model capabilities for polymer property prediction.

## Dataset Sources

The scripts use datasets from:
- `./../../Datasets/PSMILES/polyVERSE/electron_affinity/electron_affinity_data_polymers_v4.csv`
- `./../../Datasets/PSMILES/polyVERSE/ionization_energy/ionization_energy_data_polymers_v4.csv`

## Requirements

The main dependencies for running these scripts are:
- Python 3.10+
- pandas
- numpy
- litellm (for API interaction)
- dotenv (for loading API keys)
- regex (for extracting values from responses)

For a full list of dependencies, see `requirements.txt`.
