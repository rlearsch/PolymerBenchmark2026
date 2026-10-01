# Task 4: Architecture Transfer

Task 4 measures transfer from homopolymers plus alternating copolymers to
unseen random and block copolymer architectures for electron affinity (EA).
It is a protocol variant, not a dataset-scaling result.

## Canonical data

The shared training source is constructed from:

- `PSMILES/polyVERSE/electron_affinity/electron_affinity_data_polymers_v4.csv`
- `wPSMILES/polyVERSE/electron_affinity/electron_affinity_data_polymers_v4.csv`
- `wPSMILES/Vipea/EA/alternating_EA.csv`

The VIPEA files are generated, ignored outputs. Create them from their tracked
source files once per checkout:

```bash
python Datasets/Dataset_construction_scripts/process_Vipea_data.py
```

The protocol converts alternating VIPEA wPSMILES to PSMILES, then deduplicates
the combined set by PSMILES. The 28 duplicated VIPEA PSMILES pairs use their
mean EA target and the earliest source wPSMILES. The resulting canonical set
has 6,478 polymers: 6,110 alternating and 368 homopolymers.

Random and block VIPEA copolymers form a fixed 36,828-row OOD holdout (18,414
of each) and never enter model fitting or hyperparameter selection.

## Shared folds

Materialize deterministic PSMILES-grouped five-fold 80/10/10 splits:

```bash
python Datasets/Dataset_construction_scripts/create_task4_shared_splits.py \
  --polybert-dictionary Datasets/Dataset_construction_scripts/files/PSMILES_pBERT_dict.pkl
```

Omit `--polybert-dictionary` to materialize PSMILES, wPSMILES, and RDKit
splits first. Before training polyBERT, update the ignored dictionary with
`create_pSMILES_pBERT_dictionary.py` and rerun with the option above. Outputs
are ignored beneath `generated/shared_splits/`; the manifest records source
hashes, row counts, the seed (42), and fold membership.

Each fold contains the same trainer-compatible representation layout:

```text
fold_00/train_<n>/{PSMILES,wPSMILES,RDKit_descriptors,polyBERT,indices}/
```

The per-fold `train_<n>` name reflects group-preserving partition sizes. The
validation and in-distribution test sets are fixed within a fold.

## Training and OOD scoring

Use the Task 4 training entry point to keep this protocol's outputs separate
from scaling experiments. Its default model roots are
`Models/<model>/task4-architecture-transfer/`:

```bash
bash Datasets/task4_architecture_transfer/train_models.sh rdkit_rf
bash Datasets/task4_architecture_transfer/train_models.sh polybert
bash Datasets/task4_architecture_transfer/train_models.sh chemprop
bash Datasets/task4_architecture_transfer/train_models.sh periodic_graph
```

`both_graph` runs Chemprop and periodic graph concurrently. The generic
`scripts/train_graph_scaling.sh` retains its `scaling_results` default for
published scaling experiments; use its `--result-dir` option only when a
non-Task-4 workflow needs a custom result directory.

### In-distribution evaluation

Every trainer writes held-out `preds_test.csv` or `test_scores.json` per fold.
Join the prediction `row_index` to `canonical_train.csv` to report alternating
and homopolymer test RMSE separately. Fold-level ID RMSEs are summarized using
their mean and population standard deviation.

### OOD evaluation

Chemprop predicts `ood_wpsmiles.csv` directly. RDKit RF, polyBERT, and the
periodic graph model predict the unique PSMILES in `ood_components.csv`; join
those predictions as `psmiles,prediction` and combine them with the supplied
molar fractions using the Task 4-local helper:

```bash
python Datasets/task4_architecture_transfer/summarize_weighted_predictions.py \
  --holdout Datasets/task4_architecture_transfer/generated/shared_splits/architecture_holdout.csv \
  --component-predictions component_predictions.csv \
  --output task4_ood_predictions.csv --metrics task4_ood_metrics.json
```

For the RF pipeline, `evaluate_rdkit_ood.py` generates component predictions
and per-fold OOD prediction files directly from saved RF models:

```bash
Models/RDKit_RF/.venv/bin/python Datasets/task4_architecture_transfer/evaluate_rdkit_ood.py \
  --split-root Datasets/task4_architecture_transfer/generated/shared_splits \
  --result-root Models/RDKit_RF/task4-architecture-transfer \
  --output-root Models/RDKit_RF/task4-architecture-transfer/ood \
  --datasets-root Datasets
```

The OOD holdout is fixed across folds, so fold metrics are not independent
replicates. Retain all five prediction vectors, average predictions per
polymer, and calculate one random, block, and combined OOD RMSE/MAE/MSE from
that five-fold ensemble. Do not apply a t-test to the five fold-level OOD
RMSEs.

## Distributed results

`task4_all_model_scores.csv` contains the delivered fold-level ID scores and
five-fold-ensemble OOD scores for all four models. The full OOD ensemble
predictions remain ignored with their model outputs.
