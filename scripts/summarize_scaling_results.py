#!/usr/bin/env python3
"""Combine scaling experiment metrics across model output directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize explicit-split scaling results.")
    parser.add_argument(
        "--result-roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more scaling result roots, e.g. Models/polyBERT/scaling_results/...",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for combined summaries.")
    return parser.parse_args()


def parse_train_and_fold(metrics_path: Path) -> tuple[int | None, int | None]:
    train_size = None
    fold = None
    parts = metrics_path.parts
    for index, part in enumerate(parts):
        if part.startswith("train_"):
            try:
                train_size = int(part.split("_", 1)[1])
            except ValueError:
                pass
            if index + 1 < len(parts) and parts[index + 1].startswith("fold_"):
                try:
                    fold = int(parts[index + 1].split("_", 1)[1])
                except ValueError:
                    pass
        elif fold is None and part.startswith("fold_"):
            try:
                fold = int(part.split("_", 1)[1])
            except ValueError:
                pass
    return train_size, fold


def experiment_name(result_root: Path) -> str:
    parts = result_root.resolve().parts
    if "scaling_results" in parts:
        index = parts.index("scaling_results")
        return str(Path(*parts[index + 1 :]))
    return result_root.name


def experiment_metadata(experiment: str) -> dict[str, str | None]:
    """Extract dataset metadata from a scaling-results experiment path.

    Expected experiment format is generally:
      <dataset>/<quantity>/<file_stem>/<seed>

    Examples:
      MD_5000/Cp/alternating_Cp/seed_42
      MD_5000/density/homopolymer_density/seed_42
      PolyMetriX/Tg/Tg/seed_42
    """
    parts = Path(experiment).parts
    dataset = parts[0] if len(parts) > 0 else None
    quantity = parts[1] if len(parts) > 1 else None
    file_stem = parts[2] if len(parts) > 2 else None

    polymer_type = None
    if dataset == "PolyMetriX":
        polymer_type = "homopolymer"
    elif file_stem is not None:
        if file_stem.startswith("alternating_"):
            polymer_type = "alternating"
        elif file_stem.startswith("homopolymer_"):
            polymer_type = "homopolymer"

    return {
        "dataset": dataset,
        "quantity": quantity,
        "polymer_type": polymer_type,
    }


def model_name(result_root: Path) -> str:
    parts = result_root.resolve().parts
    for candidate in ["polyBERT", "RDKit_RF", "polymer_chemprop", "polymer_periodic_graph"]:
        if candidate in parts:
            return candidate
    return "unknown"


def portable_path(path: Path) -> str:
    """Return a repository-relative path when the result lives in this checkout."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def rows_from_metrics(metrics_path: Path, result_root: Path) -> list[dict]:
    with open(metrics_path) as handle:
        data = json.load(handle)

    if "metrics" not in data:
        return []

    train_size, fold = parse_train_and_fold(metrics_path)
    experiment = experiment_name(result_root)
    metadata = experiment_metadata(experiment)
    rows = []
    for split, metrics in data["metrics"].items():
        rows.append(
            {
                "model": data.get("model", "unknown"),
                "experiment": experiment,
                **metadata,
                "train_size": data.get("train_size", train_size),
                "fold": data.get("fold", fold),
                "split": split,
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mse": metrics.get("mse"),
                "n_samples": metrics.get("n_samples"),
                "seed": data.get("seed"),
                "metrics_path": portable_path(metrics_path),
            }
        )
    return rows


def test_size_for_chemprop(test_scores_path: Path) -> int | None:
    test_full = test_scores_path.parent / "test_full.csv"
    if not test_full.exists():
        return None
    return max(0, sum(1 for _ in open(test_full)) - 1)


def rows_from_chemprop_scores(test_scores_path: Path, result_root: Path) -> list[dict]:
    with open(test_scores_path) as handle:
        data = json.load(handle)

    train_size, fold = parse_train_and_fold(test_scores_path)
    experiment = experiment_name(result_root)
    metadata = experiment_metadata(experiment)
    rows = []
    n_samples = test_size_for_chemprop(test_scores_path)
    for metric_name, values in data.items():
        if not isinstance(values, list) or not values:
            continue
        if metric_name.lower() != "rmse":
            continue
        rows.append(
            {
                "model": model_name(result_root),
                "experiment": experiment,
                **metadata,
                "train_size": train_size,
                "fold": fold,
                "split": "test",
                "rmse": float(values[0]),
                "mae": None,
                "mse": None,
                "n_samples": n_samples,
                "seed": None,
                "metrics_path": portable_path(test_scores_path),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    for root in args.result_roots:
        root = root.resolve()
        for metrics_path in sorted(root.rglob("metrics.json")):
            rows.extend(rows_from_metrics(metrics_path, root))
        for test_scores_path in sorted(root.rglob("test_scores.json")):
            rows.extend(rows_from_chemprop_scores(test_scores_path, root))

    if not rows:
        raise ValueError("No metrics.json files with a 'metrics' object were found.")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed = pd.DataFrame(rows)
    detailed = detailed.sort_values(["experiment", "model", "train_size", "fold", "split"])
    detailed.to_csv(out_dir / "scaling_results_detailed.csv", index=False)

    aggregate = (
        detailed.groupby(
            ["experiment", "dataset", "quantity", "polymer_type", "model", "train_size", "split"],
            as_index=False,
        )
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", lambda x: float(np.std(x, ddof=0))),
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda x: float(np.std(x, ddof=0))),
            mse_mean=("mse", "mean"),
            n_folds=("fold", "nunique"),
            n_samples_mean=("n_samples", "mean"),
        )
        .sort_values(["experiment", "model", "train_size", "split"])
    )
    aggregate.to_csv(out_dir / "scaling_results_aggregate.csv", index=False)

    with open(out_dir / "scaling_results_summary.json", "w") as handle:
        json.dump(
            {
                "detailed": detailed.to_dict(orient="records"),
                "aggregate": aggregate.to_dict(orient="records"),
            },
            handle,
            indent=2,
            default=json_default,
        )

    print(f"Wrote detailed summary to: {out_dir / 'scaling_results_detailed.csv'}")
    print(f"Wrote aggregate summary to: {out_dir / 'scaling_results_aggregate.csv'}")


if __name__ == "__main__":
    main()
