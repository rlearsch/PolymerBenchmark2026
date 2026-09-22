# Reproducing PolyBench26 Results

This runbook takes a fresh clone from source data to the shared-split scaling
results used for direct model comparisons. Run commands from the repository root
unless a step explicitly changes directories.

Before generating or redistributing data, consult
[`DATA_PROVENANCE.md`](../DATA_PROVENANCE.md). Source datasets retain their
own terms; notably, polyVERSE files and covered derivatives are governed by
the accompanying GTRC license rather than this repository's MIT software
license.

## 1. Record Provenance

For a scientifically traceable run, record the exact revision and machine
configuration before creating environments:

```bash
git rev-parse HEAD
git status --short
python3 --version
uname -a
```

Keep the commit hash, package lock output (`pip freeze` for each environment),
hardware details, start time, and any deviations with the final results.

## 2. Understand the Two Benchmark Modes

The repository contains two distinct workflows:

- The model-specific `Models/*/train_*.sh` wrappers reproduce legacy five-fold
  random-split runs. Each implementation performs its own split.
- The scaling workflow materializes shared train, validation, and test CSVs.
  Use it for chemically aligned comparisons and for benchmarking a new model.

The remainder of this document prioritizes the shared scaling workflow.

The scaling matrix is the repository's fully automated paper-results workflow.
There is not currently one machine-readable manifest covering every non-scaling
paper table or figure, and the manuscript is not included in the repository.
For those results, use the paper's method section to select the exact dataset
and invoke the legacy wrapper documented by the corresponding model README.
Record that mapping rather than guessing when the paper is unavailable.

## 3. Requirements

- **Supported baseline:** Python 3.10 or newer, Git, and a Bash-compatible
  macOS or Linux environment. The setup scripts create one virtual environment
  per dataset or model workflow.
- **CPU execution:** supported for dataset construction and model training;
  CUDA is optional. A CUDA GPU can accelerate the neural models, but no CUDA
  version is required for the baseline workflow.
- **Platform dependency:** use a platform on which `rdkit==2025.9.6` can be
  installed. This version is required to reproduce the published canonical
  PSMILES checksums; newer RDKit releases change alternating-copolymer output.
- **Resources:** approximately 16 GB RAM for polyBERT generation and
  approximately 15 GB free disk for generated datasets; allow more space for
  model checkpoints and concurrent runs.

The dataset-construction requirements pin RDKit to the checksum-verified
version. Other dependency requirements remain supported floors; for an exact
model-training reproduction, record `pip freeze` separately for every environment, together
with the Python version, operating-system details, CUDA details when used, and
the repository commit as shown in Section 1.

The external polyBERT SentenceTransformer model is not stored in this
repository. Clone it as a sibling of the repository:

```bash
git clone https://huggingface.co/kuelumbus/polyBERT ../polyBERT
```

The expected layout is:

```text
parent/
  PolyBench26/
  polyBERT/
```

Before running the public release workflow, check out the immutable polyBERT
revision used for the release, then verify it:

```bash
# Public-release polyBERT revision: deaa98fb65a7bdfb537457d42f43bd468963f695
git -C ../polyBERT checkout deaa98fb65a7bdfb537457d42f43bd468963f695
git -C ../polyBERT rev-parse HEAD
```

Do not use an unrecorded moving branch for a published-result reproduction.
The external model is described by Kuenneth and Ramprasad, *Nature
Communications* 14, 4099 (2023), https://doi.org/10.1038/s41467-023-39868-6.

## 4. Build the Basic Dataset Environment

```bash
cd Datasets/Dataset_construction_scripts
bash setup_environments.sh
bash generate_basic_datasets.sh
cd ../..
```

At this point a custom model can use the tracked `Datasets/scaling_indices/`
packages directly. Verify that the generated canonical files match the
published checksums:

```bash
Datasets/Dataset_construction_scripts/.venv/bin/python \
  Datasets/Dataset_construction_scripts/manage_scaling_indices.py verify
```

To reproduce all built-in representations, also generate polyBERT:

```bash
cd Datasets/Dataset_construction_scripts
bash generate_polybert_datasets.sh
cd ../..
```

`generate_polybert_datasets.sh` asks for confirmation and may take hours. The
full standalone RDKit representation is optional for scaling because the split
generator calculates descriptors from canonical PSMILES. Generate it for legacy
random-split RDKit runs with:

```bash
cd Datasets/Dataset_construction_scripts
bash generate_rdkit_datasets.sh
cd ../..
```

Expected generated roots are `Datasets/wPSMILES/`, `Datasets/polyBERT/`, and,
if requested, `Datasets/RDKit_descriptors/`. These are ignored by Git.

## 5. Generate the Shared Scaling Splits

Inspect the exact ten-condition matrix first, then generate it:

```bash
bash Datasets/Dataset_construction_scripts/generate_scaling_datasets.sh --dry-run
bash Datasets/Dataset_construction_scripts/generate_scaling_datasets.sh
```

Use `--overwrite` only when intentionally replacing existing split files. The
generator automatically consumes matching published indices from
`Datasets/scaling_indices/`; it does not choose new folds. The output root is
`Datasets/scaling_splits/`. Each condition contains a manifest,
five folds, nested training sizes, fixed validation/test data, four molecular
representations, and canonical row-index files.

The authoritative design and condition list are in
[`../SCALING_EXPERIMENTS_METHODS.md`](../SCALING_EXPERIMENTS_METHODS.md).

## 6. Build Model Environments

Each built-in model has an isolated environment:

```bash
for model in RDKit_RF polyBERT polymer_chemprop polymer_periodic_graph; do
  (cd "Models/$model" && bash setup_environment.sh)
done
```

The setup scripts ask whether to replace an existing `.venv`. For unattended
runs, create environments one at a time and respond explicitly rather than
assuming an existing environment can be discarded.

## 7. Train RDKit_RF Across Every Condition

```bash
repo_root="$PWD"
while IFS= read -r -d '' manifest; do
  split_root="$(dirname "$manifest")"
  relative="${split_root#"$repo_root/Datasets/scaling_splits/"}"
  Models/RDKit_RF/.venv/bin/python Models/RDKit_RF/train_rf_scaling.py \
    --split-root "$split_root" \
    --save-root "Models/RDKit_RF/scaling_results/$relative"
done < <(find "$repo_root/Datasets/scaling_splits" -name manifest.json -print0 | sort -z)
```

## 8. Train polyBERT Across Every Condition

```bash
repo_root="$PWD"
while IFS= read -r -d '' manifest; do
  split_root="$(dirname "$manifest")"
  relative="${split_root#"$repo_root/Datasets/scaling_splits/"}"
  Models/polyBERT/.venv/bin/python Models/polyBERT/train_pBERT_scaling.py \
    --split-root "$split_root" \
    --save-root "Models/polyBERT/scaling_results/$relative"
done < <(find "$repo_root/Datasets/scaling_splits" -name manifest.json -print0 | sort -z)
```

RDKit_RF and polyBERT can run concurrently if memory permits. Both trainers
write fold-level metrics, predictions, and summaries beneath ignored model
result roots.

## 9. Train the Graph Models

```bash
bash scripts/train_graph_scaling.sh polymer_chemprop
bash scripts/train_graph_scaling.sh polymer_periodic_graph
```

Run these in separate terminals to train concurrently, or use:

```bash
bash scripts/train_graph_scaling.sh both
```

Use `--only DATASET/PROPERTY/FILE_STEM` for a single condition and `--dry-run`
to inspect commands. The wrapper passes explicit validation and test paths; do
not replace them with internal random splitting.

## 10. Collect Compact Results

```bash
bash scripts/collect_scaling_results.sh
```

This reads available outputs from the four model directories and writes:

```text
scaling_results/<dataset>/<property>/<file_stem>/seed_42/
  scaling_results_detailed.csv
  scaling_results_aggregate.csv
  scaling_results_summary.json

scaling_results/all_seed_42/
  scaling_results_detailed.csv
  scaling_results_aggregate.csv
  scaling_results_summary.json
```

These compact summaries are eligible for Git. Raw splits, checkpoints,
predictions, and logs remain ignored.

## 11. Validate Completeness

```bash
Datasets/Dataset_construction_scripts/.venv/bin/python - <<'PY'
import pandas as pd

path = "scaling_results/all_seed_42/scaling_results_aggregate.csv"
data = pd.read_csv(path)
test = data[data["split"] == "test"]
print(test[["experiment", "model", "train_size", "rmse_mean", "rmse_std", "n_folds"]])

incomplete = test[test["n_folds"] != 5]
if len(incomplete):
    print("\nWARNING: incomplete five-fold results")
    print(incomplete[["experiment", "model", "train_size", "n_folds"]])
PY
```

Before treating a reconstruction as successful, verify:

- all intended experiment/model/training-size rows exist;
- every completed aggregate has five distinct folds;
- test sample counts are constant across models for a fold and condition;
- no generated data or checkpoint was accidentally made Git-eligible;
- deviations in package versions, hardware, seeds, or epochs are recorded.

Small floating-point differences may occur across hardware and library versions.
Large differences require checking split identity, target selection, units,
preprocessing leakage, and whether the best validation checkpoint was used.

## 12. Legacy Random-Split Runs

For a model-specific benchmark run, generate the required full representation
and call the model wrapper with one CSV:

```bash
(cd Models/RDKit_RF && bash train_rf.sh ../../Datasets/RDKit_descriptors/MD_300/density/homopolymer_density.csv)
(cd Models/polyBERT && bash train_pBERT.sh ../../Datasets/polyBERT/MD_300/density/homopolymer_density.csv)
(cd Models/polymer_chemprop && bash train_pcp.sh ../../Datasets/wPSMILES/MD_300/density/homopolymer_density.csv)
(cd Models/polymer_periodic_graph && bash train_ppg.sh ../../Datasets/PSMILES/MD_300/density/homopolymer_density.csv)
```

Consult each model README for its output layout and prediction command. Because
these wrappers split independently, do not use them for claims that require
identical held-out chemicals across representations.
