# Published legacy full-dataset results

`full_dataset_5fold_cv_rmse.csv` is the simplified public-facing
summary of legacy full-dataset RMSE results. It records 64 mean/std RMSE
entries from five-fold cross-validation over the reported full dataset sizes.
Only conditions represented by the release-facing data inventory are included.

The table preserves reported labels alongside canonical PolyBench26 names:

| Reported model | Canonical model |
|---|---|
| `PCP` | `polymer_chemprop` |
| `PPG` | `polymer_periodic_graph` |
| `RDKit` | `RDKit_RF` |
| `polyBERT` | `polyBERT` |

This is an aggregate-only historical result table. Individual fold metrics,
prediction row indices, split assignments, seeds, runtime environment, and the
definition of the reported standard deviation are not available. The raw
reported labels remain in the `*_reported` columns.

These results are **not shared-split scaling results**. Do not combine them
with `scaling_results` or present them as directly comparable to the aligned
scaling protocol: their split identity and preprocessing provenance are
unavailable, and their reported dataset sizes are full source datasets rather
than fold-specific `train_pool` sizes.

PolyMetriX source provenance and redistribution status are recorded in
[`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md) and must be resolved before a
public release.
