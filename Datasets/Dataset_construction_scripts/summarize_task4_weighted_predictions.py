#!/usr/bin/env python3
"""Combine component predictions into Task 4 random/block copolymer predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from task4_architecture_transfer import weighted_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--component-predictions", type=Path, required=True, help="CSV with psmiles,prediction columns")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    holdout = pd.read_csv(args.holdout)
    predictions = pd.read_csv(args.component_predictions)
    if not {"psmiles", "prediction"} <= set(predictions):
        raise ValueError("Component predictions require psmiles and prediction columns")
    if predictions["psmiles"].duplicated().any():
        raise ValueError("Component predictions must have one prediction per PSMILES")
    lookup = predictions.set_index("psmiles")["prediction"]
    holdout["prediction_a"] = holdout["psmiles_a"].map(lookup)
    holdout["prediction_b"] = holdout["psmiles_b"].map(lookup)
    if holdout[["prediction_a", "prediction_b"]].isna().any().any():
        raise ValueError("Missing component predictions for one or more holdout polymers")
    holdout["prediction"] = holdout.apply(lambda row: weighted_prediction(row.prediction_a, row.prediction_b, row.molar_fraction_a, row.molar_fraction_b), axis=1)
    holdout["squared_error"] = (holdout["electron_affinity"] - holdout["prediction"]) ** 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(args.output, index=False)
    summary = holdout.groupby("architecture")["squared_error"].mean().pow(0.5).rename("rmse")
    print(summary.to_csv(header=True))


if __name__ == "__main__":
    main()
