#!/usr/bin/env python3
"""Evaluate Task 4 RF fold models on component-weighted architecture OOD data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Datasets" / "Dataset_construction_scripts"))
from rdkit_descriptor_cache import DescriptorCache  # noqa: E402


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
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=Path("Datasets"))
    parser.add_argument("--descriptor-cache", type=Path)
    args = parser.parse_args()

    holdout = pd.read_csv(args.split_root / "architecture_holdout.csv")
    components = pd.unique(pd.concat([holdout["psmiles_a"], holdout["psmiles_b"]], ignore_index=True))
    cache_path = args.descriptor_cache or args.datasets_root / "RDKit_descriptors" / ".task4-rf-descriptor_cache.sqlite3"
    with DescriptorCache(cache_path) as cache:
        features, invalid = cache.descriptor_frame(components)
    if invalid:
        raise ValueError(f"Cannot calculate descriptors for OOD components: {invalid[:10]}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fold_dir in sorted(args.result_root.glob("train_*/fold_*")):
        model = joblib.load(fold_dir / "rf.joblib")
        feature_names = json.loads((fold_dir / "feature_names.json").read_text())
        predictions = pd.DataFrame(
            {"psmiles": components, "prediction": model.predict(features.reindex(columns=feature_names))}
        )
        lookup = predictions.set_index("psmiles")["prediction"]
        result = holdout.copy()
        result["prediction"] = (
            result["molar_fraction_a"] * result["psmiles_a"].map(lookup)
            + result["molar_fraction_b"] * result["psmiles_b"].map(lookup)
        )
        output = args.output_root / fold_dir.name
        output.mkdir(exist_ok=True)
        predictions.to_csv(output / "component_predictions.csv", index=False)
        result.to_csv(output / "ood_predictions.csv", index=False)
        record = {"fold": fold_dir.name, "metrics": {}}
        for architecture, subset in list(result.groupby("architecture", sort=True)) + [("combined", result)]:
            record["metrics"][architecture] = metrics(subset)
        (output / "ood_metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        summaries.append(record)
    (args.output_root / "ood_metrics_all_folds.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
