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

Run the existing four trainers against `generated/shared_splits`. Save Task 4
results under a Task-4-specific model directory, not `scaling_results`.

Chemprop predicts `ood_wpsmiles.csv` directly. RDKit RF, polyBERT, and the
periodic graph model predict the unique PSMILES in `ood_components.csv`; join
those predictions as `psmiles,prediction` and combine them with the supplied
molar fractions:

```bash
python Datasets/task4_architecture_transfer/summarize_weighted_predictions.py \
  --holdout Datasets/task4_architecture_transfer/generated/shared_splits/architecture_holdout.csv \
  --component-predictions component_predictions.csv \
  --output task4_ood_predictions.csv --metrics task4_ood_metrics.json
```

Report fold-level RMSE, MAE, and MSE for `random`, `block`, and combined OOD
holdouts, separately from in-distribution test metrics. Aggregate fold metrics
with population standard deviation.
