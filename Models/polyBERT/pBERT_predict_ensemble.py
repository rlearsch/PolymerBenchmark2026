#!/usr/bin/env python3
"""
Ensemble predictions from saved `checkpoint_best.pt` files.

- Loads every checkpoint that matches a glob (default: **/checkpoint_best.pt)
- Rebuilds the model from checkpoint["config"]
- Applies that checkpoint's feature_mean/std normalization
- Predicts on the provided CSV
- Ensembles by averaging per-checkpoint predictions (regression) OR averaging probabilities then argmax (classification)

Outputs a CSV with:
  orig_index, y_true (if present), y_pred_ens

Example:
  python ensemble_predict.py \
    --data_csv ./../../Datasets/polyBERT/Tg_combined/Tg_full.csv \
    --ckpt_glob "./runs_ffn/**/checkpoint_best.pt" \
    --out_csv  ./ensemble_preds.csv
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---- Model: same as your training code
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        hs = cfg["hidden_size"]
        fhs = cfg["ffn_hidden_size"]
        depth = cfg["depth"]
        head_layers = cfg["ffn_num_layers"]
        p = cfg["dropout"]
        bias = cfg["bias"]
        num_tasks = cfg["num_tasks"]

        layers = []
        last = input_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hs, bias=bias), nn.ReLU(), nn.Dropout(p)]
            last = hs
        for _ in range(head_layers):
            layers += [nn.Linear(last, fhs, bias=bias), nn.ReLU(), nn.Dropout(p)]
            last = fhs
        layers += [nn.Linear(last, num_tasks, bias=bias)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class XOnlyDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], idx


@torch.no_grad()
def predict_one_checkpoint(
    ckpt_path: Path,
    X_raw: np.ndarray,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    is_classification = str(cfg.get("dataset_type", "regression")).lower() == "classification"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = X_raw.shape[1]
    model = FeedForwardNet(input_dim=input_dim, cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Apply per-checkpoint normalization
    mean = ckpt["feature_mean"].detach().cpu().numpy()
    std = ckpt["feature_std"].detach().cpu().numpy()
    X = (X_raw - mean) / (std + 1e-8)

    ds = XOnlyDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds = []
    for xb, _idx in loader:
        xb = xb.to(device)
        out = model(xb)

        if is_classification:
            # out: (B, C). Convert to probabilities.
            probs = torch.softmax(out, dim=1)
            preds.append(probs.detach().cpu())
        else:
            # regression: (B, 1) typically
            preds.append(out.detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()

    if (not is_classification) and preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.reshape(-1)

    return preds


def load_csv_as_array(
    csv_path: Path,
    x_cols: int,
    y_col: Optional[int],
    skip_header: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.genfromtxt(csv_path, delimiter=",", dtype=float, filling_values=np.nan, skip_header=skip_header)
    X = data[:, :x_cols]
    y = None
    if y_col is not None:
        y = data[:, y_col]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, required=True, help="Path to dataset CSV (same format you trained on).")
    ap.add_argument("--ckpt_glob", type=str, default="./runs_ffn/**/checkpoint_best.pt",
                    help="Glob for checkpoint files (recursive glob supported).")
    ap.add_argument("--out_csv", type=str, default="./ensemble_preds.csv", help="Output CSV path.")
    ap.add_argument("--x_cols", type=int, default=600, help="Number of feature columns (default 600).")
    ap.add_argument("--y_col", type=int, default=600, help="Target column index (default 600). Set to -1 to omit.")
    ap.add_argument("--batch_size", type=int, default=2048, help="Batch size for inference.")
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (default auto).")
    args = ap.parse_args()

    data_csv = Path(args.data_csv)
    ckpt_paths = sorted(Path().glob(args.ckpt_glob))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found for glob: {args.ckpt_glob}")

    y_col = None if args.y_col < 0 else args.y_col

    X_raw, y = load_csv_as_array(data_csv, x_cols=args.x_cols, y_col=y_col, skip_header=1)

    # Drop rows with NaN targets if y is present (mirrors your training mask)
    orig_idx = np.arange(X_raw.shape[0], dtype=np.int64)
    if y is not None:
        mask = np.isfinite(y)
        X_raw = X_raw[mask]
        y = y[mask]
        orig_idx = orig_idx[mask]

    # Determine classification vs regression from the first checkpoint config
    first_ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    cfg0 = first_ckpt["config"]
    is_classification = str(cfg0.get("dataset_type", "regression")).lower() == "classification"

    # Predict for each checkpoint
    preds_list: List[np.ndarray] = []
    for p in ckpt_paths:
        print(f"Predicting with: {p}")
        preds = predict_one_checkpoint(
            ckpt_path=p,
            X_raw=X_raw,
            batch_size=args.batch_size,
            device=args.device,
        )
        preds_list.append(preds)

    # Ensemble
    if is_classification:
        # preds are probabilities (N, C). Average then argmax.
        probs_avg = np.mean(np.stack(preds_list, axis=0), axis=0)
        y_pred_ens = np.argmax(probs_avg, axis=1).astype(np.int64)
    else:
        # regression: average predictions (N,)
        y_pred_ens = np.mean(np.stack(preds_list, axis=0), axis=0).astype(np.float64)

    # Write output
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        if y is None:
            f.write("orig_index,y_pred_ens\n")
            for i, yp in zip(orig_idx, y_pred_ens):
                f.write(f"{int(i)},{float(yp)}\n")
        else:
            f.write("orig_index,y_true,y_pred_ens\n")
            for i, yt, yp in zip(orig_idx, y, y_pred_ens):
                f.write(f"{int(i)},{float(yt)},{float(yp)}\n")

    print(f"\nWrote ensemble predictions to: {out_path}")
    print(f"Used {len(ckpt_paths)} checkpoints")
    if is_classification:
        print("Mode: classification (avg probs -> argmax)")
    else:
        print("Mode: regression (avg predictions)")


if __name__ == "__main__":
    main()
