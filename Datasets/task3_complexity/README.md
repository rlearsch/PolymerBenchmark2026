# Task 3: Repeat-Unit Complexity

`alternating_refractive_index.csv` is the compact, tracked metadata source for
PolyBench26 Task 3. It contains alternating-copolymer refractive-index targets
and the repeat-unit complexity labels used for Figure 4 of the paper.

Task 3 is a post-hoc stratification of the legacy alternating refractive-index
test predictions, not a complexity-held-out training task. Train a model using
the legacy alternating refractive-index protocol, evaluate its held-out test
predictions, join predictions to this table by `original_index`, and report
RMSE separately for each `n_distinct_monomers` value from 5 through 10.

## Schema

| Column | Meaning |
|---|---|
| `psmiles` | Standard PSMILES representation of the complete alternating repeat sequence. |
| `wpsmiles` | Weighted PSMILES representation of the same sequence. |
| `original_index` | Stable row identifier from the legacy full refractive-index source. Use this as the join key. |
| `n_distinct_monomers` | Number of chemically distinct monomers in the alternating repeat sequence. |
| `refractive_index` | DFT-calculated refractive-index target. |
| `legacy_test_fold` | Legacy five-fold test assignment (1 through 5). |

The paper's Task 3 analysis uses rows with `5 <= n_distinct_monomers <= 10`.
The table also retains legacy rows with two or four units for provenance; do not
include those rows in the published Task 3 aggregate.

The source values were migrated from the legacy OpenPolymerBench Task 3
preparation artifact on 2026-09-25. The pre-migration file SHA-256 was
`6d099ee5dd158c52448741859843a7eaa5c8c9ff7d4dc43cdc483bbb7d7d469a`.
