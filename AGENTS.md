# PolyBench26 Agent Context

This file is the first stop for an automated coding or research agent working
in Polymer Benchmark 2026 (PolyBench26). Read it before changing code, creating
datasets, or launching experiments.

## Mission

PolyBench26 compares polymer-property prediction models across several molecular
representations. A user should be able to clone the repository and either:

1. reconstruct the repository's benchmark and dataset-size scaling results, or
2. evaluate a new model on exactly the same chemicals and folds.

For operational commands, continue with:

- [`docs/REPRODUCING_RESULTS.md`](docs/REPRODUCING_RESULTS.md)
- [`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md)
- [`SCALING_EXPERIMENTS_METHODS.md`](SCALING_EXPERIMENTS_METHODS.md)
- [`SCALING_EXPERIMENTS_AGENT_GUIDE.md`](SCALING_EXPERIMENTS_AGENT_GUIDE.md)

## Repository Map

```text
Datasets/
  Dataset_construction_scripts/   Source-to-representation pipelines
  PSMILES/                        Canonical polymer strings and small sources
  wPSMILES/                       Generated weighted polymer strings
  RDKit_descriptors/              Generated descriptor matrices
  polyBERT/                       Generated 600-dimensional embeddings
  scaling_splits/                 Generated shared train/val/test datasets
Models/
  RDKit_RF/                       Random forest on RDKit descriptors
  polyBERT/                       Feed-forward network on polyBERT embeddings
  polymer_chemprop/               Polymer-aware Chemprop on wPSMILES
  polymer_periodic_graph/         Periodic graph model on PSMILES
  OpenAI/                         Prompt-based EA/IP experiments
scripts/                          Scaling training and result collection
scaling_results/                  Compact, Git-eligible scaling summaries
```

## Authoritative Data Flow

```text
included compact sources
  -> generate_basic_datasets.sh
  -> PSMILES + wPSMILES
  -> generate_polybert_datasets.sh
  -> polyBERT
  -> generate_scaling_datasets.sh
  -> fixed, aligned scaling_splits
  -> model trainers
  -> ignored raw model artifacts
  -> collect_scaling_results.sh
  -> compact scaling_results summaries
```

For scaling experiments, PSMILES is the canonical row order. The split builder
validates wPSMILES and polyBERT targets row by row and regenerates RDKit
descriptors from canonical PSMILES. Never create model-specific random splits
when comparing against the scaling results.

## Representation-to-Model Contract

| Representation | Location | Built-in model |
|---|---|---|
| PSMILES | `Datasets/PSMILES/` | `polymer_periodic_graph` |
| wPSMILES | `Datasets/wPSMILES/` | `polymer_chemprop` |
| RDKit descriptors | `Datasets/RDKit_descriptors/` or scaling splits | `RDKit_RF` |
| polyBERT | `Datasets/polyBERT/` | `polyBERT` |

PSMILES and wPSMILES CSVs use `smiles` as the feature column and one property
column as the target. Descriptor and embedding CSVs place the target last.
Scaling split directories also contain `indices/*.csv`; `row_index` maps every
row back to the canonical full PSMILES dataset.

## Reproducibility Invariants

- Scaling split seed: 42.
- Scaling folds: 5.
- Validation fraction: 10% of the full dataset per fold.
- Test fraction: 10% of the full dataset per fold.
- Training subsets are nested within a fold.
- Validation and test rows are fixed across training sizes within a fold.
- Every representation uses the same chemicals for a given fold and size.
- Fit preprocessing, normalization, imputation, and feature selection on the
  training split only.
- Report held-out test RMSE at minimum; preserve fold-level results.
- Do not compare a partial-fold aggregate with a completed five-fold result
  without reporting `n_folds`.

## Data and Git Policy

The repository intentionally tracks only compact source datasets, generation
code, documentation, and compact result summaries. `.gitignore` excludes:

- generated wPSMILES, descriptor, polyBERT, and scaling-split datasets;
- the large polyBERT dictionary;
- raw Cv calculation files and their archive;
- virtual environments, checkpoints, predictions, and raw model outputs.

Before adding data or artifacts, run `git check-ignore <path>` and inspect file
sizes. Do not force-add ignored generated data. New large artifacts should be
reconstructible from a tracked script and compact source or documented external
dependency.

## Safe Agent Workflow

1. Run `git status --short` and preserve unrelated user changes.
2. Read the root, dataset, and relevant model README files.
3. Confirm whether the task targets legacy random-split benchmarking or the
   shared scaling protocol.
4. Use the existing environment and wrapper scripts where possible.
5. Start with `--dry-run`, `--help`, one fold, or one small condition.
6. Keep output beneath an already ignored generated-data or model-results path.
7. Validate row identity, target alignment, fold count, and metrics before
   reporting completion.
8. Do not commit, upload, call paid APIs, or overwrite completed experiments
   unless the user explicitly requests it.

## Known Caveats

- The repository currently provides a complete automated manifest for the
  dataset-size scaling experiments. It does not encode every paper table and
  figure as one master manifest, and the manuscript itself is not included.
  When reconstructing results beyond scaling, map the paper's stated dataset,
  representation, split, and model settings to the legacy wrappers explicitly;
  do not infer or invent missing experimental details.
- Standard `train_*.sh` wrappers create model-specific random folds. They are
  useful for legacy runs but are not the cross-representation comparison path.
- `generate_polybert_datasets.sh` is interactive, needs substantial memory and
  disk, and expects the external polyBERT model in a sibling directory named
  `polyBERT` next to the repository.
- Setup scripts are interactive when an environment already exists.
- The scaling result collector knows the four built-in result roots. A new
  model can use `scripts/summarize_scaling_results.py` directly as described in
  `docs/ADDING_A_MODEL.md`.
- The OpenAI experiments require credentials, network access, and paid API
  calls. Never launch them merely as a validation step.
- A compact result file may represent an incomplete run. Inspect `n_folds` and
  the detailed CSV rather than inferring completeness from file presence.

## Minimum Validation Before Handoff

```bash
bash -n Datasets/Dataset_construction_scripts/*.sh scripts/*.sh Models/*/*.sh
python3 -m compileall -q Datasets/Dataset_construction_scripts/*.py Models/RDKit_RF/*.py Models/polyBERT/*.py scripts/*.py
bash Datasets/Dataset_construction_scripts/generate_scaling_datasets.sh --dry-run
git diff --check
```

Use the project environments instead of system `python3` when dependencies are
required. Full training is expensive and is not a routine smoke test.
