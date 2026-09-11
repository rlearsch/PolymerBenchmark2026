# Published Scaling Indices

This directory contains the compact, canonical train/validation/test assignments
used by the PolyBench26 dataset-size scaling experiments. It is intentionally
tracked in Git so a new model can use the published folds without generating
polyBERT embeddings, RDKit descriptors, or the 7.6 GB materialized split tree.

Each experiment contains:

- `manifest.json`: canonical dataset path and SHA-256, target, fold settings,
  training sizes, and row counts;
- `fold_NN.npz`: integer arrays named `train_pool`, `val`, and `test`.

For a training size `N`, use `train_pool[:N]`. Validation and test indices stay
fixed across all training sizes in that fold.

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("Datasets/scaling_indices/MD_5000/Cp/alternating_Cp/seed_42")
manifest = json.loads((root / "manifest.json").read_text())
full_data = pd.read_csv(Path("Datasets") / manifest["canonical_source"])

with np.load(root / "fold_00.npz") as indices:
    train = full_data.iloc[indices["train_pool"][:300]]
    validation = full_data.iloc[indices["val"]]
    test = full_data.iloc[indices["test"]]
```

After generating the canonical PSMILES datasets, verify their checksums and all
index partitions with:

```bash
Datasets/Dataset_construction_scripts/.venv/bin/python \
  Datasets/Dataset_construction_scripts/manage_scaling_indices.py verify
```

Maintainers can regenerate this directory from materialized scaling splits with
the `export` subcommand. Export is deterministic and rejects non-nested or
incomplete partitions.
