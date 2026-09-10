# Benchmarking a New Model with PolyBench26

This guide defines the minimum contract for comparing a new regression model
with the built-in PolyBench26 scaling results.

## 1. Use the Shared Splits

Generate splits as described in
[`REPRODUCING_RESULTS.md`](REPRODUCING_RESULTS.md). Select an existing
representation beneath each training-size directory:

```text
Datasets/scaling_splits/<dataset>/<property>/<file_stem>/seed_42/
  fold_00/
    train_300/
      indices/{train,val,test}.csv
      PSMILES/{train,val,test}.csv
      wPSMILES/{train,val,test}.csv
      RDKit_descriptors/{train,val,test}.csv
      polyBERT/{train,val,test}.csv
```

Do not resplit these files. For a custom representation, transform each row
from the PSMILES files or canonical `row_index` values while preserving order.
Keep the index CSVs beside predictions so chemical identity remains auditable.

## 2. Data Contracts

### PSMILES and wPSMILES

- Feature column: `smiles`
- Target: the single remaining column
- One row per polymer

### RDKit descriptors and polyBERT

- Numeric feature columns precede the target
- Target is the final column
- polyBERT uses the first 600 columns as features

### Canonical indices

Each `indices/*.csv` contains:

- `order`: position in the materialized split
- `row_index`: position in the full canonical PSMILES CSV
- `split`: train, validation, or test label
- `smiles`: canonical PSMILES identity
- the property target

Join on `row_index` when attaching a separately computed representation. Never
infer identity from target values alone.

## 3. Training Rules

- Train one independent model for every fold and training size.
- Use only `train.csv` to fit the model and all preprocessing.
- Use `val.csv` for hyperparameter selection, early stopping, or checkpoint
  selection.
- Evaluate `test.csv` once with the selected model.
- Do not fit normalization, imputation, vocabulary, feature selection, or
  dimensionality reduction on validation/test rows.
- Use the same hyperparameter policy across folds and sizes unless a documented
  validation-only selection procedure is part of the model.
- Record software versions, hardware, model seed, training duration, and the
  exact PolyBench26 Git revision.

The split seed is 42. A model may use a separate initialization seed, but it
must be recorded. For strict comparison with the current built-ins, use one
model run per fold; additional initialization repeats should be reported as a
separate experimental axis.

## 4. Required Metrics

At minimum, report fold-level test RMSE:

```text
RMSE = sqrt(mean((prediction - target)^2))
```

MAE and MSE are recommended. Preserve predictions with canonical row indices:

```csv
row_index,y_true,y_pred
123,1.25,1.31
...
```

Do not average predictions or metrics across folds before saving fold-level
values. Aggregate with population standard deviation (`ddof=0`) to match the
repository summaries.

## 5. Compatible Output Layout

Use this structure so the existing summarizer can parse training size and fold:

```text
Models/<your_model>/scaling_results/
  <dataset>/<property>/<file_stem>/seed_42/
    train_300/fold_00/metrics.json
    train_300/fold_01/metrics.json
    ...
```

A compatible `metrics.json` is:

```json
{
  "model": "your_model",
  "seed": 42,
  "metrics": {
    "train": {
      "rmse": 0.12,
      "mae": 0.08,
      "mse": 0.0144,
      "n_samples": 300
    },
    "val": {
      "rmse": 0.18,
      "mae": 0.13,
      "mse": 0.0324,
      "n_samples": 180
    },
    "test": {
      "rmse": 0.19,
      "mae": 0.14,
      "mse": 0.0361,
      "n_samples": 180
    }
  }
}
```

The summarizer derives `train_size` and `fold` from the directory names. A
trainer may also include them at the JSON top level.

## 6. Summarize a New Model

The convenience collector has a fixed list of built-in models. Summarize a new
model directly for each experiment:

```bash
python3 scripts/summarize_scaling_results.py \
  --result-roots Models/your_model/scaling_results/MD_5000/Cp/alternating_Cp/seed_42 \
  --out-dir scaling_results/your_model/MD_5000/Cp/alternating_Cp/seed_42
```

To compare it with built-in outputs in one table, pass multiple roots from the
same condition:

```bash
python3 scripts/summarize_scaling_results.py \
  --result-roots \
    Models/your_model/scaling_results/MD_5000/Cp/alternating_Cp/seed_42 \
    Models/RDKit_RF/scaling_results/MD_5000/Cp/alternating_Cp/seed_42 \
    Models/polyBERT/scaling_results/MD_5000/Cp/alternating_Cp/seed_42 \
    Models/polymer_chemprop/scaling_results/MD_5000/Cp/alternating_Cp/seed_42 \
    Models/polymer_periodic_graph/scaling_results/MD_5000/Cp/alternating_Cp/seed_42 \
  --out-dir scaling_results/comparisons/your_model/MD_5000/Cp/alternating_Cp/seed_42
```

Use an environment containing pandas and NumPy if system `python3` does not
have them.

## 7. Recommended Model Directory

```text
Models/your_model/
  README.md                 Installation, method, citation, and limitations
  requirements.txt         Pinned or bounded runtime dependencies
  setup_environment.sh     Reproducible environment setup
  train_scaling.py         Shared-split training entry point
  predict.py               Optional inference entry point
  scaling_results/         Generated and ignored
```

Do not modify shared split files or another model's outputs. Add a focused
ignore rule if the new model writes checkpoints somewhere not already covered.

## 8. Pre-Comparison Checklist

- [ ] Same dataset, property, file stem, split seed, fold, and training size
- [ ] Same canonical `row_index` values in train, validation, and test
- [ ] No train/validation/test preprocessing leakage
- [ ] Target units and column verified
- [ ] Best checkpoint selected without test-set feedback
- [ ] Fold-level predictions and metrics retained
- [ ] Exactly five completed folds or an explicit partial-run label
- [ ] Mean and population standard deviation reported
- [ ] Code, dependency versions, model seed, and hardware documented
- [ ] Generated datasets and raw model artifacts remain ignored by Git

If any item differs, label the result as a protocol variant rather than a direct
PolyBench26 comparison.
