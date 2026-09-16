# Polymer Benchmark 2026 (PolyBench26): Dataset-Size Scaling Methods

## Experimental Objective

Dataset size scaling experiments were designed to quantify the dependence of polymer property prediction performance on the number of training examples. The central design constraint was that comparisons across molecular representations and model classes should use chemically identical validation and test sets. This avoids confounding model comparisons with differences in data partitioning.

## Datasets And Properties

Experiments used polymer property datasets from the `MD_5000` and `PolyMetriX` collections. For `MD_5000`, alternating copolymer datasets were evaluated for heat capacity at constant pressure (`Cp`), density, refractive index, and radius of gyration (`Rg`). Homopolymer datasets were evaluated for `Cp`, heat capacity at constant volume (`Cv`), density, refractive index, and `Rg`. For `PolyMetriX`, the homopolymer glass-transition-temperature dataset `Tg/Tg.csv` was evaluated.

Alternating `MD_5000` datasets were evaluated at training sizes of 300, 1000, 3000, and 10000 examples. Homopolymer `MD_5000` datasets were evaluated at training sizes of 300 and 1000 examples; although 3000 examples were initially requested, these datasets contained only approximately 1715-1800 total rows, making a 3000-example training subset infeasible after holding out validation and test sets. The `PolyMetriX` `Tg` dataset was evaluated at training sizes of 300, 1000, and 3000 examples.

## Molecular Representations

Four representations were considered:

- PSMILES, used by the polymer periodic graph model.
- wPSMILES, used by the polymer chemprop model.
- 600-dimensional polyBERT embeddings, used by a feed-forward neural network.
- RDKit molecular descriptors, used by a random forest baseline.

The PSMILES files were treated as the canonical chemical ordering for split construction. wPSMILES and polyBERT files were required to match the PSMILES row count, target column, and target values. RDKit descriptors were regenerated directly from the canonical PSMILES rows for each split to ensure chemical identity across representations.

## Data Splitting Protocol

All experiments used five repeated folds generated with base seed 42. Within each fold, row indices were shuffled deterministically and partitioned into a test set, a validation set, and a training pool. The validation and test fractions were each 10% of the full dataset. Training subsets of increasing size were then sampled as nested prefixes of the fold-specific training pool.

This procedure ensured that, within a fold, all training sizes shared the same validation and test chemicals. It also ensured that smaller training subsets were strict subsets of larger training subsets. Across folds, validation and test chemicals differed, providing an estimate of split sensitivity. The same index partitions were applied to all molecular representations.

## Models

The RDKit baseline used a random forest regressor trained on precomputed RDKit molecular descriptors. Missing descriptor values were imputed with the median value estimated from the training split. The default scaling implementation used 100 trees per forest.

The polyBERT model used a feed-forward neural network trained on 600-dimensional polyBERT embeddings. Input normalization was computed from the training split only and then applied to validation and test splits. Models were trained for 50 epochs with batch size 50, and the checkpoint with the lowest validation RMSE was retained for evaluation.

The polymer chemprop model was trained on wPSMILES inputs using the repository's Chemprop-based polymer model implementation. The polymer periodic graph model was trained on PSMILES inputs using the repository's periodic graph model implementation. For both graph models, external split files were passed using separate validation and test paths so that the models did not perform internal random splitting.

## Evaluation And Aggregation

The primary evaluation metric was root mean squared error (RMSE) on the held-out test set. For RDKit and polyBERT models, train, validation, and test metrics were recorded. For the graph models, the central result collector records test RMSE from Chemprop-style `test_scores.json` outputs.

For each experiment, metrics were aggregated across completed folds by training size and model. Aggregate files report the mean and population standard deviation of RMSE across folds. For partially completed graph-model runs, the number of completed folds is reported and may be less than five.

Per-experiment summaries are stored under:

```text
scaling_results/<dataset>/<property>/<file_stem>/seed_42/
```

Combined summaries across available experiments are stored under:

```text
scaling_results/all_seed_42/
```

The detailed summary files contain one row per experiment, model, train size, split, and fold. The aggregate summary files contain one row per experiment, model, train size, and split.

## Reproducibility

The split-generation seed was fixed at 42. All split assignments were materialized as CSV files in `Datasets/scaling_splits`, including per-fold training-pool, validation, test, and train-size-specific indices. Model training scripts consume these explicit files, making the data partitions independent of model-specific random number generators.

The canonical assignments are published in compact form under
`Datasets/scaling_indices`. Each fold stores integer `train_pool`, `val`, and
`test` arrays plus a canonical PSMILES SHA-256. A training set of size `N` is
the prefix `train_pool[:N]`. This lets external models use the exact benchmark
folds without materializing built-in representations.

Generate the complete experiment matrix from the repository root with:

```bash
bash Datasets/Dataset_construction_scripts/generate_scaling_datasets.sh
```

The generated split files are intentionally excluded from Git; the source data,
generation scripts, experiment definitions, and compact aggregate results remain
eligible for version control.
