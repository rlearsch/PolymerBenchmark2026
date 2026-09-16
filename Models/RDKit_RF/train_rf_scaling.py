#!/usr/bin/env python3
"""Train Random Forest models on explicit dataset-size scaling splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from train_rf import load_and_prepare_data


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RDKit RF on explicit scaling split CSVs.")
    parser.add_argument("--split-root", type=Path, required=True, help="Root created by create_scaling_splits.py")
    parser.add_argument("--save-root", type=Path, required=True, help="Directory for model outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_row_indices(path: Path, expected_rows: int) -> np.ndarray:
    df = pd.read_csv(path)
    if "row_index" not in df.columns:
        raise ValueError(f"Missing row_index column: {path}")
    if len(df) != expected_rows:
        raise ValueError(f"Index file {path} has {len(df)} rows, expected {expected_rows}.")
    return df["row_index"].to_numpy(dtype=np.int64)


def train_random_forest(x_train: pd.DataFrame, y_train: np.ndarray, random_state: int, n_estimators: int) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=0,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train.ravel())
    return model


def evaluate(model: Pipeline, x: pd.DataFrame, y: np.ndarray, split: str) -> tuple[dict, np.ndarray]:
    pred = np.asarray(model.predict(x)).reshape(-1)
    true = np.asarray(y).reshape(-1)
    mse = float(mean_squared_error(true, pred))
    return (
        {
            "split": split,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(true, pred)),
            "mse": mse,
            "n_samples": int(len(true)),
        },
        pred,
    )


def write_predictions(path: Path, original_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_index": original_indices.astype(np.int64),
            "y_true": np.asarray(y_true).reshape(-1),
            "y_pred": np.asarray(y_pred).reshape(-1),
        }
    ).to_csv(path, index=False)


def align_columns(x: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in x.columns]
    for col in missing:
        x[col] = np.nan
    return x[feature_cols]


def train_one_split(
    split_dir: Path,
    output_dir: Path,
    seed: int,
    n_estimators: int,
) -> dict:
    train_path = split_dir / "RDKit_descriptors" / "train.csv"
    val_path = split_dir / "RDKit_descriptors" / "val.csv"
    test_path = split_dir / "RDKit_descriptors" / "test.csv"

    x_train, y_train, feature_cols, target_cols = load_and_prepare_data(train_path)
    x_val, y_val, _, _ = load_and_prepare_data(val_path)
    x_test, y_test, _, _ = load_and_prepare_data(test_path)
    x_val = align_columns(x_val, feature_cols)
    x_test = align_columns(x_test, feature_cols)

    train_indices = load_row_indices(split_dir / "indices" / "train.csv", len(y_train))
    val_indices = load_row_indices(split_dir / "indices" / "val.csv", len(y_val))
    test_indices = load_row_indices(split_dir / "indices" / "test.csv", len(y_test))

    model = train_random_forest(x_train, y_train, random_state=seed, n_estimators=n_estimators)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_metrics = {}
    for split_name, x, y, original_indices in [
        ("train", x_train, y_train, train_indices),
        ("val", x_val, y_val, val_indices),
        ("test", x_test, y_test, test_indices),
    ]:
        metrics, pred = evaluate(model, x, y, split_name)
        split_metrics[split_name] = metrics
        write_predictions(output_dir / f"preds_{split_name}.csv", original_indices, y, pred)

    dump(model, output_dir / "rf.joblib")
    with open(output_dir / "feature_names.json", "w") as handle:
        json.dump(feature_cols, handle, indent=2)
    with open(output_dir / "train_columns.json", "w") as handle:
        json.dump(feature_cols, handle, indent=2)

    result = {
        "model": "RDKit_RF",
        "seed": seed,
        "n_estimators": n_estimators,
        "target_columns": target_cols,
        "metrics": split_metrics,
    }
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(result, handle, indent=2)
    return result


def train_size_from_dir(path: Path) -> int:
    return int(path.name.split("_", 1)[1])


def write_summary(rows: list[dict], save_root: Path) -> None:
    detailed = pd.DataFrame(rows)
    detailed.to_csv(save_root / "summary.csv", index=False)
    aggregate = (
        detailed.groupby(["model", "train_size", "split"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", lambda x: float(np.std(x, ddof=0))),
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda x: float(np.std(x, ddof=0))),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["train_size", "split"])
    )
    aggregate.to_csv(save_root / "summary_aggregate.csv", index=False)
    with open(save_root / "summary.json", "w") as handle:
        json.dump(
            {
                "detailed": detailed.to_dict(orient="records"),
                "aggregate": aggregate.to_dict(orient="records"),
            },
            handle,
            indent=2,
            default=json_default,
        )


def main() -> None:
    args = parse_args()
    split_root = args.split_root.resolve()
    save_root = args.save_root.resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for fold_dir in sorted(split_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        fold = int(fold_dir.name.split("_", 1)[1])
        for train_dir in sorted(fold_dir.glob("train_*"), key=train_size_from_dir):
            train_size = train_size_from_dir(train_dir)
            run_seed = args.seed + fold
            output_dir = save_root / f"train_{train_size}" / f"fold_{fold:02d}"
            if not args.quiet:
                print(f"Training RDKit_RF train_size={train_size} fold={fold} seed={run_seed}")
            result = train_one_split(train_dir, output_dir, run_seed, args.n_estimators)
            result.update({"fold": fold, "train_size": train_size})
            with open(output_dir / "run_metadata.json", "w") as handle:
                json.dump(result, handle, indent=2)

            for split_name, split_metrics in result["metrics"].items():
                rows.append(
                    {
                        "model": "RDKit_RF",
                        "train_size": train_size,
                        "fold": fold,
                        "split": split_name,
                        "rmse": split_metrics["rmse"],
                        "mae": split_metrics["mae"],
                        "mse": split_metrics["mse"],
                        "n_samples": split_metrics["n_samples"],
                        "seed": run_seed,
                    }
                )

    if not rows:
        raise ValueError(f"No fold/train directories found under {split_root}")
    write_summary(rows, save_root)
    print(f"Wrote RDKit_RF scaling results to: {save_root}")


if __name__ == "__main__":
    main()
