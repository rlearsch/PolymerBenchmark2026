#!/usr/bin/env python3
"""Normalize the aggregate legacy full-dataset RMSE table for publication."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "Dataset size",
    "Mean RMSE",
    "Std RMSE",
    "Quantity",
    "Model",
    "Polymer Type",
    "Source",
]

MODEL_NAMES = {
    "polyBERT": "polyBERT",
    "PCP": "polymer_chemprop",
    "PPG": "polymer_periodic_graph",
    "RDKit": "RDKit_RF",
}

PROPERTY_NAMES = {
    "$C_p$": "Cp",
    "$C_v$": "Cv",
    "Density": "density",
    "Refractive Index": "refractive_index",
    "Refractive index": "refractive_index",
    "Radius of gyration": "Rg",
    "$T_g$ (PoLyInfo)": "Tg",
    "$T_g$ (PolyMetriX)": "Tg",
    "Tg (combined)": "Tg",
    "EA": "EA",
    "IP": "IP",
}

ARCHITECTURES = {
    "Homopolymer": "homopolymer",
    "Alternating Copolymer": "alternating",
    "Random copolymer": "random",
    "Block copolymer": "block",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Legacy tab-separated source table.")
    parser.add_argument("--output", type=Path, required=True, help="Normalized comma-separated output CSV.")
    return parser.parse_args()


def normalized_table(source: pd.DataFrame) -> pd.DataFrame:
    if list(source.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected input columns. Expected exactly: " + ", ".join(EXPECTED_COLUMNS)
        )

    unknown_models = set(source["Model"]) - MODEL_NAMES.keys()
    unknown_properties = set(source["Quantity"]) - PROPERTY_NAMES.keys()
    unknown_architectures = set(source["Polymer Type"]) - ARCHITECTURES.keys()
    if unknown_models or unknown_properties or unknown_architectures:
        raise ValueError(
            "Unmapped labels: "
            f"models={sorted(unknown_models)}, "
            f"properties={sorted(unknown_properties)}, "
            f"architectures={sorted(unknown_architectures)}"
        )

    result = pd.DataFrame(
        {
            "source_row": source.index + 2,
            "protocol": "legacy_full_dataset_5fold_cv",
            "shared_splits": False,
            "n_folds": 5,
            "metric": "rmse",
            "dataset_size": pd.to_numeric(source["Dataset size"], errors="raise"),
            "rmse_mean": pd.to_numeric(source["Mean RMSE"], errors="raise"),
            "rmse_std": pd.to_numeric(source["Std RMSE"], errors="raise"),
            "property": source["Quantity"].map(PROPERTY_NAMES),
            "model": source["Model"].map(MODEL_NAMES),
            "polymer_architecture": source["Polymer Type"].map(ARCHITECTURES),
            "source_label": source["Source"],
            "quantity_reported": source["Quantity"],
            "model_reported": source["Model"],
            "polymer_type_reported": source["Polymer Type"],
            "source_reported": source["Source"],
            "std_definition": "not_recorded_in_source_table",
            "fold_metrics_available": False,
        }
    )
    if (result[["dataset_size", "rmse_mean", "rmse_std"]] < 0).any().any():
        raise ValueError("Dataset sizes and RMSE values must be non-negative.")
    identifiers = ["property", "model", "polymer_architecture", "source_label", "dataset_size"]
    if result.duplicated(identifiers).any():
        raise ValueError("Normalized table contains duplicate condition/model rows.")
    return result.sort_values(identifiers).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input, sep="\t")
    result = normalized_table(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} normalized rows to {args.output}")


if __name__ == "__main__":
    main()
