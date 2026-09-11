# Every-Commit Dataset Tests

Run the fast dataset-construction suite from the repository root with:

```bash
python3 -m unittest discover -s tests -v
```

Use `Datasets/Dataset_construction_scripts/.venv/bin/python` when the system
interpreter does not provide pandas, NumPy, and RDKit.

The suite checks fresh-clone source availability and schemas, Git ignore policy,
machine-independent paths, the scaling experiment matrix, conversion behavior,
a tiny end-to-end four-representation scaling pipeline, row alignment, nested
splits, deterministic outputs, and CLI behavior from an unrelated directory.

The tests write only to temporary directories and do not modify production
datasets.
