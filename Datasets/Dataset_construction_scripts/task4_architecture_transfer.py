#!/usr/bin/env python3
"""Materialize the PolyBench26 Task 4 architecture-transfer inputs.

The native wPSMILES model receives the holdout ``wpsmiles`` directly. Models
without an architecture-aware representation predict each component PSMILES and
combine predictions with the supplied molar fractions.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "Datasets/task4_architecture_transfer/legacy_train.csv"
DEFAULT_INFO = ROOT / "Datasets/Dataset_construction_scripts/files/polymer-chemprop-data/dataset.csv"
DEFAULT_WPSMILES = ROOT / "Datasets/Dataset_construction_scripts/files/polymer-chemprop-data/dataset-poly_chemprop.csv"

LEGACY_ALTERNATING_ROWS = 5525
LEGACY_TRAIN_ROWS = 5855
MAP_PATTERN = re.compile(r"\[\*:\d+\]")


def parse_wpsmiles(wpsmiles: str) -> tuple[str, str, float, float]:
    """Return the two component PSMILES strings and their molar fractions."""
    parts = wpsmiles.split("|")
    units = parts[0].split(".")
    fractions = [float(value) for value in parts[1:-1]]
    if len(units) != 2 or len(fractions) != 2:
        raise ValueError("Task 4 requires exactly two wPSMILES components and fractions")
    if any(value < 0 for value in fractions) or not abs(sum(fractions) - 1.0) < 1e-9:
        raise ValueError(f"Invalid molar fractions in {wpsmiles!r}")
    return (
        MAP_PATTERN.sub("*", units[0]),
        MAP_PATTERN.sub("*", units[1]),
        fractions[0],
        fractions[1],
    )


def weighted_prediction(prediction_a: float, prediction_b: float, fraction_a: float, fraction_b: float) -> float:
    """Compute the Task 4 composition-weighted baseline prediction."""
    if fraction_a < 0 or fraction_b < 0 or not abs(fraction_a + fraction_b - 1.0) < 1e-9:
        raise ValueError("Molar fractions must be non-negative and sum to one")
    return prediction_a * fraction_a + prediction_b * fraction_b


def build_task4(train_path: Path, info_path: Path, wpsmiles_path: Path, output_dir: Path) -> None:
    train = pd.read_csv(train_path)
    if list(train.columns) != ["SMILES", "EA (eV)"] or len(train) != LEGACY_TRAIN_ROWS:
        raise ValueError("Unexpected legacy Task 4 training source schema or row count")
    train_output = pd.DataFrame(
        {
            "psmiles": train["SMILES"],
            "electron_affinity": train["EA (eV)"],
            "architecture": ["alternating"] * LEGACY_ALTERNATING_ROWS
            + ["homopolymer"] * (LEGACY_TRAIN_ROWS - LEGACY_ALTERNATING_ROWS),
        }
    )

    info = pd.read_csv(info_path)
    wpsmiles = pd.read_csv(wpsmiles_path)
    if len(info) != len(wpsmiles):
        raise ValueError("VIPEA information and wPSMILES rows are not aligned")
    merged = info.join(wpsmiles[["poly_chemprop_input"]])
    holdout = merged[merged["poly_type"].isin(["random", "block"])].copy()
    parsed = holdout["poly_chemprop_input"].map(parse_wpsmiles)
    holdout[["psmiles_a", "psmiles_b", "molar_fraction_a", "molar_fraction_b"]] = pd.DataFrame(parsed.tolist(), index=holdout.index)
    if not (holdout["fracA"].astype(float).round(12) == holdout["molar_fraction_a"].round(12)).all():
        raise ValueError("VIPEA molar fractions disagree with wPSMILES")
    if not (holdout["fracB"].astype(float).round(12) == holdout["molar_fraction_b"].round(12)).all():
        raise ValueError("VIPEA molar fractions disagree with wPSMILES")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_output.to_csv(output_dir / "train.csv", index=False)
    holdout.rename(columns={"poly_id": "polymer_id", "poly_type": "architecture", "poly_chemprop_input": "wpsmiles", "EA (eV)": "electron_affinity"})[
        ["polymer_id", "architecture", "wpsmiles", "psmiles_a", "psmiles_b", "molar_fraction_a", "molar_fraction_b", "electron_affinity"]
    ].to_csv(output_dir / "holdout.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--vipea-info", type=Path, default=DEFAULT_INFO)
    parser.add_argument("--vipea-wpsmiles", type=Path, default=DEFAULT_WPSMILES)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_task4(args.legacy_train, args.vipea_info, args.vipea_wpsmiles, args.output_dir)


if __name__ == "__main__":
    main()
