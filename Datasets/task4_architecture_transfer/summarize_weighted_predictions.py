#!/usr/bin/env python3
"""Combine Task 4 component predictions into architecture-OOD predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def weighted_prediction(
    prediction_a: float, prediction_b: float, fraction_a: float, fraction_b: float
) -> float:
    """Return the composition-weighted prediction for a two-component polymer."""
    if fraction_a < 0 or fraction_b < 0 or not np.isclose(fraction_a + fraction_b, 1.0):
        raise ValueError("Molar fractions must be non-negative and sum to one")
    return prediction_a * fraction_a + prediction_b * fraction_b


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    error = frame["electron_affinity"] - frame["prediction"]
    mse = float(np.mean(np.square(error)))
    return {
        "n_samples": int(len(frame)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument(
        "--component-predictions", type=Path, required=True,
        help="CSV with one psmiles,prediction row for every unique holdout component",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, help="Optional JSON destination for OOD metrics")
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
    holdout["prediction"] = holdout.apply(
        lambda row: weighted_prediction(
            row.prediction_a, row.prediction_b,
            row.molar_fraction_a, row.molar_fraction_b,
        ),
        axis=1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(args.output, index=False)

    metric_rows = []
    for architecture, frame in list(holdout.groupby("architecture", sort=True)) + [("combined", holdout)]:
        metric_rows.append({"architecture": architecture, **metrics(frame)})
    print(pd.DataFrame(metric_rows).to_csv(index=False))
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps({"metrics": metric_rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
