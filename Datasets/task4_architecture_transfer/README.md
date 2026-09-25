# Task 4: Architecture Transfer

Task 4 trains on the legacy combined alternating-copolymer and polyVERSE
homopolymer electron-affinity dataset, then evaluates random and block
copolymers as architectures held out from training.

`legacy_train.csv` is the exact 5,855-row legacy training source: 5,525
alternating rows followed by 330 polyVERSE-derived homopolymer rows. Its
SHA-256 is `a28f790cbf0336bbd1144d946edcdefa56350c189452431e7cfc298d29749208`.

Materialize model inputs with:

```bash
python Datasets/Dataset_construction_scripts/task4_architecture_transfer.py \
  --output-dir Datasets/task4_architecture_transfer/generated
```

For architecture-unaware models, predict the two component PSMILES strings and
use `summarize_task4_weighted_predictions.py` to calculate
`x_A * prediction_A + x_B * prediction_B` for each random/block holdout row.
The native wPSMILES model should use `holdout.csv` directly.
