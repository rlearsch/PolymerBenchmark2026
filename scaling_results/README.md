# PolyBench26 Scaling Results

This directory contains compact CSV and JSON summaries from the Polymer
Benchmark 2026 dataset-size scaling experiments. Raw split CSVs, trained models,
checkpoints, predictions, and logs are generated locally and excluded from Git.

- `all_seed_42/` combines every available experiment.
- Dataset-specific directories contain the same summaries for one property and
  polymer architecture.
- `scaling_results_detailed.csv` contains one row per model, training size,
  split, and fold.
- `scaling_results_aggregate.csv` reports means and population standard
  deviations across completed folds.

See [`../SCALING_EXPERIMENTS_METHODS.md`](../SCALING_EXPERIMENTS_METHODS.md) for
the methodology and [`../SCALING_EXPERIMENTS_AGENT_GUIDE.md`](../SCALING_EXPERIMENTS_AGENT_GUIDE.md)
for reproduction commands.
