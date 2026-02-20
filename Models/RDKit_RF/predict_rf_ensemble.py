import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from rdkit import Chem
from rdkit.Chem.Descriptors import CalcMolDescriptors
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ---------- feature generation (match training) ----------
def calc_all_descriptors(psmi: str, missing=float("nan")):
    m = Chem.MolFromSmiles(psmi)
    if m is None:
        return {}
    return CalcMolDescriptors(m, missingVal=np.nan, silent=True)


def sanitize(X: pd.DataFrame) -> pd.DataFrame:
    try:
        X = X.drop(
            columns=[
                "MaxPartialCharge",
                "MinPartialCharge",
                "MaxAbsPartialCharge",
                "MinAbsPartialCharge",
                "Ipc",
            ]
        )
    except Exception:
        pass
    X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def convert_file(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError("Expected a 'smiles' column in the CSV.")
    y = None
    y_cols = [c for c in df.columns if c != "smiles"]
    if len(y_cols) > 0:
        y = df[y_cols].to_numpy()

    X = df["smiles"].apply(calc_all_descriptors).apply(pd.Series)
    X = sanitize(X)
    return X, y, df


def align_columns(X: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    missing = [c for c in train_columns if c not in X.columns]
    for c in missing:
        X[c] = np.nan
    # drop extras + reorder
    return X[train_columns]


# ---------- helpers to locate folds / columns ----------
def find_fold_models(root: Path):
    # root can be .../density_data or .../density_data/fold_0 etc.
    root = root.resolve()
    if root.name.startswith("fold_") and (root / "rf.joblib").exists():
        return [root / "rf.joblib"]

    models = sorted(root.glob("fold_*/rf.joblib"))
    if not models:
        # fallback: recursive search
        models = sorted(root.rglob("fold_*/rf.joblib"))
    return models


def load_train_columns(fold_dir: Path):
    # preferred: per-fold saved columns
    p = fold_dir / "train_columns.json"
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)

    # fallback: try to extract from the pipeline
    # works if sklearn has feature_names_in_
    model = load(fold_dir / "rf.joblib")
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # last resort: you NEED a saved column list from training
    raise RuntimeError(
        f"No train_columns.json and can't infer feature_names_in_ for {fold_dir}. "
        "Save train_columns.json during training (recommended)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-root", type=Path, required=True,
                    help="Path to dataset directory containing fold_*/rf.joblib (e.g. .../density_data)")
    ap.add_argument("--data", type=Path, required=True, help="New CSV to predict on")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV")
    ap.add_argument("--prefix", type=str, default="pred",
                    help="Column prefix for predictions (default: pred)")
    args = ap.parse_args()

    model_paths = find_fold_models(args.fold_root)
    if not model_paths:
        raise FileNotFoundError(f"Couldn't find fold_*/rf.joblib under {args.fold_root}")

    # featurize once
    X_new, y_new, df_raw = convert_file(args.data)

    # Use fold_0 (first found) as canonical column set unless per-fold columns exist.
    first_fold_dir = model_paths[0].parent
    canonical_cols = load_train_columns(first_fold_dir)
    X_new = align_columns(X_new, canonical_cols)

    # predict with each fold
    fold_preds = []
    fold_names = []
    for mp in model_paths:
        fold_dir = mp.parent
        fold_name = fold_dir.name  # fold_0, fold_1, ...
        model = load(mp)

        # If folds saved different columns, align to THIS model's columns.
        # If not available, we already aligned to canonical.
        try:
            cols = load_train_columns(fold_dir)
            X_aligned = align_columns(X_new.copy(), cols)
        except Exception:
            X_aligned = X_new

        p = model.predict(X_aligned)
        p = np.asarray(p)
        fold_preds.append(p)
        fold_names.append(fold_name)

    # stack: (n_folds, n_samples) or (n_folds, n_samples, n_targets)
    P = np.stack(fold_preds, axis=0)

    mean_pred = P.mean(axis=0)
    std_pred = P.std(axis=0, ddof=0)

    out_df = df_raw.copy()

    # Save per-fold preds (helpful for debugging)
    # Handle 1D vs 2D targets
    if mean_pred.ndim == 1:
        for i, fname in enumerate(fold_names):
            out_df[f"{args.prefix}_{fname}"] = P[i, :]
        out_df[f"{args.prefix}_mean"] = mean_pred
        out_df[f"{args.prefix}_std"] = std_pred
    else:
        # multi-target: create columns pred_mean_t0, pred_mean_t1, etc.
        n_targets = mean_pred.shape[1]
        for i, fname in enumerate(fold_names):
            for t in range(n_targets):
                out_df[f"{args.prefix}_{fname}_t{t}"] = P[i, :, t]
        for t in range(n_targets):
            out_df[f"{args.prefix}_mean_t{t}"] = mean_pred[:, t]
            out_df[f"{args.prefix}_std_t{t}"] = std_pred[:, t]

    out_df.to_csv(args.out, index=False)
    print(f"Wrote ensemble predictions to {args.out}")

    # optional scoring if labels exist and are 1D
    if y_new is not None and y_new.size > 0 and mean_pred.ndim == 1:
        y_flat = y_new.ravel()
        scores = {
            "RMSE": float(np.sqrt(mean_squared_error(y_flat, mean_pred))),
            "MAE": float(mean_absolute_error(y_flat, mean_pred)),
        }
        score_path = args.out.with_suffix(".scores.json")
        with open(score_path, "w") as f:
            json.dump(scores, f, indent=2)
        print("Scores:", scores)
        print(f"Wrote scores to {score_path}")


if __name__ == "__main__":
    main()
