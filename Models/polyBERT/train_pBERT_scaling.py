#!/usr/bin/env python3
"""Train polyBERT models on explicit dataset-size scaling splits."""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_pBERT import ArrayDataset, FeedForwardNet, config, predict_with_indices, train_one_epoch


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train polyBERT on explicit scaling split CSVs.")
    parser.add_argument("--split-root", type=Path, required=True, help="Root created by create_scaling_splits.py")
    parser.add_argument("--save-root", type=Path, required=True, help="Directory for model outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto when omitted")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_polybert_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", dtype=float, filling_values=np.nan, skip_header=1)
    data = np.atleast_2d(data)
    if data.shape[1] < 601:
        raise ValueError(f"Expected at least 601 columns in {path}; found {data.shape[1]}.")
    x = data[:, :600]
    y = data[:, 600]
    if not np.isfinite(y).all():
        raise ValueError(f"Target column contains non-finite values: {path}")
    return x, y


def load_row_indices(path: Path, expected_rows: int) -> np.ndarray:
    df = pd.read_csv(path)
    if "row_index" not in df.columns:
        raise ValueError(f"Missing row_index column: {path}")
    if len(df) != expected_rows:
        raise ValueError(f"Index file {path} has {len(df)} rows, expected {expected_rows}.")
    return df["row_index"].to_numpy(dtype=np.int64)


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, split: str) -> dict[str, float | int | str]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = float(np.mean((y_pred - y_true) ** 2))
    return {
        "split": split,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "mse": mse,
        "n_samples": int(len(y_true)),
    }


def predict_split(model: torch.nn.Module, dataset: ArrayDataset, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred, y, local_idx = predict_with_indices(model, loader, device, is_classification=False)
    return pred.reshape(-1), y.reshape(-1), local_idx.reshape(-1)


def write_predictions(path: Path, original_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_index": original_indices.astype(np.int64),
            "y_true": y_true.reshape(-1),
            "y_pred": y_pred.reshape(-1),
        }
    ).to_csv(path, index=False)


def train_one_split(
    split_dir: Path,
    output_dir: Path,
    cfg: dict,
    seed: int,
    device: str,
    quiet: bool,
) -> dict:
    train_x, train_y = load_polybert_csv(split_dir / "polyBERT" / "train.csv")
    val_x, val_y = load_polybert_csv(split_dir / "polyBERT" / "val.csv")
    test_x, test_y = load_polybert_csv(split_dir / "polyBERT" / "test.csv")

    train_indices = load_row_indices(split_dir / "indices" / "train.csv", len(train_y))
    val_indices = load_row_indices(split_dir / "indices" / "val.csv", len(val_y))
    test_indices = load_row_indices(split_dir / "indices" / "test.csv", len(test_y))

    torch.manual_seed(seed)
    train_ds = ArrayDataset(train_x, train_y)
    val_ds = ArrayDataset(val_x, val_y)
    test_ds = ArrayDataset(test_x, test_y)

    feature_mean = train_ds.X.mean(dim=0)
    feature_std = train_ds.X.std(dim=0, unbiased=False)
    for dataset in (train_ds, val_ds, test_ds):
        dataset.set_norm(feature_mean, feature_std)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = FeedForwardNet(input_dim=train_x.shape[1], cfg=cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["init_lr"])

    steps_per_epoch = max(1, len(train_loader))
    pct_start = min(max(cfg["warmup_epochs"] / max(cfg["epochs"], 1), 0.0), 0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["max_lr"],
        epochs=cfg["epochs"],
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=max(cfg["max_lr"] / max(cfg["init_lr"], 1e-12), 1.0),
        final_div_factor=max(cfg["max_lr"] / max(cfg["final_lr"], 1e-12), 1.0),
    )

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = -1
    history = {"train_rmse": [], "train_mse": [], "val_rmse": [], "val_mse": [], "best_epoch": None}

    for epoch in range(1, cfg["epochs"] + 1):
        train_mse = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_pred, val_true, _ = predict_split(model, val_ds, cfg["batch_size"], device)
        val_metrics = metrics_from_arrays(val_true, val_pred, "val")

        train_rmse = math.sqrt(train_mse)
        val_rmse = float(val_metrics["rmse"])
        history["train_mse"].append(float(train_mse))
        history["train_rmse"].append(float(train_rmse))
        history["val_mse"].append(float(val_metrics["mse"]))
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch

        if not quiet and (epoch == 1 or epoch == cfg["epochs"] or epoch % max(1, cfg["log_frequency"]) == 0):
            print(f"[Epoch {epoch:03d}] train_rmse={train_rmse:.6f} val_rmse={val_rmse:.6f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch

    output_dir.mkdir(parents=True, exist_ok=True)
    split_metrics = {}
    for split_name, dataset, original_indices in [
        ("train", train_ds, train_indices),
        ("val", val_ds, val_indices),
        ("test", test_ds, test_indices),
    ]:
        pred, true, local_idx = predict_split(model, dataset, cfg["batch_size"], device)
        ordered_indices = original_indices[local_idx]
        split_metrics[split_name] = metrics_from_arrays(true, pred, split_name)
        write_predictions(output_dir / f"preds_{split_name}.csv", ordered_indices, true, pred)

    checkpoint = {
        "model_state_dict": best_state,
        "config": cfg,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_rmse": float(best_val_rmse),
        "feature_mean": feature_mean.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "metrics": split_metrics,
    }
    torch.save(checkpoint, output_dir / "checkpoint_best.pt")
    np.savez(output_dir / "norm.npz", mean=feature_mean.detach().cpu().numpy(), std=feature_std.detach().cpu().numpy())

    with open(output_dir / "history.json", "w") as handle:
        json.dump(history, handle, indent=2)

    result = {
        "model": "polyBERT",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_rmse": float(best_val_rmse),
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

    cfg = dict(config)
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["num_folds"] = 1
    cfg["dataset_type"] = "regression"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
                print(f"Training polyBERT train_size={train_size} fold={fold} seed={run_seed}")
            result = train_one_split(train_dir, output_dir, cfg, run_seed, device, args.quiet)
            result.update({"fold": fold, "train_size": train_size})
            with open(output_dir / "run_metadata.json", "w") as handle:
                json.dump(result, handle, indent=2)

            for split_name, split_metrics in result["metrics"].items():
                rows.append(
                    {
                        "model": "polyBERT",
                        "train_size": train_size,
                        "fold": fold,
                        "split": split_name,
                        "rmse": split_metrics["rmse"],
                        "mae": split_metrics["mae"],
                        "mse": split_metrics["mse"],
                        "n_samples": split_metrics["n_samples"],
                        "seed": run_seed,
                        "best_epoch": result["best_epoch"],
                    }
                )

    if not rows:
        raise ValueError(f"No fold/train directories found under {split_root}")
    write_summary(rows, save_root)
    print(f"Wrote polyBERT scaling results to: {save_root}")


if __name__ == "__main__":
    main()
