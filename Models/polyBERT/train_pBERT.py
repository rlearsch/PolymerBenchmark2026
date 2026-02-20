#!/usr/bin/env python3
"""
Train polyBERT model on polymer property prediction.

This script trains a feed-forward neural network on pre-computed polyBERT embeddings.
"""

import argparse
import math
from copy import deepcopy
import statistics as stats
from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
import sys
from pathlib import Path


config = {
    'hidden_size': 300,
    'ffn_hidden_size': 300,
    'ffn_num_layers': 2,
    'depth': 3,
    'num_tasks': 1,          # 1 for regression target
    'dropout': 0.0,
    'bias': False,
    'batch_size': 50,
    'epochs': 50,
    'init_lr': 1e-4,
    'max_lr': 1e-3,
    'final_lr': 1e-4,
    'warmup_epochs': 2.0,
    'log_frequency': 10,
    'num_folds': 5,         # repeat count
    'split_sizes': (0.8, 0.1, 0.1),
    'dataset_type': 'regression',  # change to 'classification' if needed
}


def make_out_dir_from_dataset(dataset_path: str, anchor: str = "polyBERT"):
    p = Path(dataset_path).resolve()
    parts = p.parts

    if anchor not in parts:
        raise ValueError(f"Anchor '{anchor}' not found in dataset_path")

    i = parts.index(anchor)
    subpath = Path(*parts[i + 1:]).with_suffix("")  # strip .csv

    # Leading slash to match your example; remove if undesired
    return Path("./") / subpath

# ---- Dataset that can apply per-fold normalization and returns original index
class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2, "X must be (N, D)"
        self.X = torch.tensor(X, dtype=torch.float32)
        # regression target: (N, 1)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def set_norm(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        xi = self.X[idx]
        if self.mean is not None and self.std is not None:
            xi = (xi - self.mean) / (self.std + 1e-8)
        yi = self.y[idx]
        return xi, yi, idx  # include original index

# ---- Model: Feed-Forward Net for tabular regression/classification
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        hs = cfg['hidden_size']
        fhs = cfg['ffn_hidden_size']
        depth = cfg['depth']
        head_layers = cfg['ffn_num_layers']
        p = cfg['dropout']
        bias = cfg['bias']
        num_tasks = cfg['num_tasks']

        layers = []
        last = input_dim
        # main stack
        for _ in range(depth):
            layers += [nn.Linear(last, hs, bias=bias), nn.ReLU(), nn.Dropout(p)]
            last = hs
        # optional head
        for i in range(head_layers):
            out_dim = fhs
            layers += [nn.Linear(last, out_dim, bias=bias), nn.ReLU(), nn.Dropout(p)]
            last = out_dim
        # output
        layers += [nn.Linear(last, num_tasks, bias=bias)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, D)
        return self.net(x)

@torch.no_grad()
def evaluate(model, loader, criterion, device, is_classification: bool):
    model.eval()
    if is_classification:
        # unchanged path for classification
        total_loss, total_correct, total_count = 0.0, 0, 0
        for xb, yb, _ in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb.view(-1).long())
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            total_correct += (out.argmax(dim=1) == yb.view(-1).long()).sum().item()
        avg_loss = total_loss / max(total_count, 1)
        acc = total_correct / max(total_count, 1)
        return {"loss": avg_loss, "rmse": None, "acc": acc}
    else:
        # regression: compute MSE & RMSE
        sse, count = 0.0, 0
        for xb, yb, _ in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            # criterion is nn.MSELoss() but we aggregate manually for exact RMSE
            sse += torch.sum((pred - yb) ** 2).item()
            count += xb.size(0)
        mse = sse / max(count, 1)
        rmse = math.sqrt(mse)
        return {"loss": mse, "rmse": rmse, "acc": None}
    
@torch.no_grad()
def predict_with_indices(model, loader, device, is_classification: bool):
    model.eval()
    preds, ys, idxs = [], [], []
    for xb, yb, idx in loader:
        xb = xb.to(device)
        out = model(xb)
        if is_classification:
            out = out.argmax(dim=1, keepdim=True).float()
        preds.append(out.detach().cpu())
        ys.append(yb.detach().cpu())
        idxs.append(idx.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    ys    = torch.cat(ys, dim=0).numpy()
    idxs  = torch.cat(idxs, dim=0).numpy()
    return preds, ys, idxs


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running = 0.0
    for xb, yb, _idx in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)

def run_repeated_experiments_from_arrays(X: np.ndarray, y: np.ndarray, cfg: dict, out_dir: str = "./runs_ffn", base_seed: int = 42):
    all_best_states = []
    all_norms = []   # (feature_mean, feature_std) per fold

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_classification = cfg['dataset_type'].lower() == 'classification'
    assert X.shape[0] == y.shape[0], "X and y must have same length"

    base_ds = ArrayDataset(X, y)
    N = len(base_ds)
    p_train, p_val, p_test = cfg['split_sizes']
    n_train = int(N * p_train)
    n_val   = int(N * p_val)
    n_test  = N - n_train - n_val  # remainder to test

    all_hist, all_test_metrics, all_test_preds, all_test_indices = [], [], [], []

    input_dim = X.shape[1]
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for fold in range(cfg['num_folds']):
        seed = base_seed + fold
        fold_dir = out_root / f"fold_{fold:02d}_seed_{seed}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)

        train_ds, val_ds, test_ds = random_split(base_ds, [n_train, n_val, n_test], generator=gen)

        # --- compute train-only normalization on raw arrays, then set on base dataset
        tr_idx = torch.as_tensor(train_ds.indices, dtype=torch.long)
        train_X_raw = base_ds.X[tr_idx]  # (n_train, D)
        feature_mean = train_X_raw.mean(dim=0)
        feature_std  = train_X_raw.std(dim=0, unbiased=False)
        base_ds.set_norm(feature_mean, feature_std)
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

        model = FeedForwardNet(input_dim, cfg).to(device)
        criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['init_lr'])

        # OneCycleLR setup
        steps_per_epoch = max(1, len(train_loader))
        pct_start = min(max(cfg['warmup_epochs'] / max(cfg['epochs'], 1), 0.0), 0.9)
        final_div_factor = max(cfg['max_lr'] / max(cfg['final_lr'], 1e-12), 1.0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg['max_lr'],
            epochs=cfg['epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy='cos',
            div_factor=max(cfg['max_lr'] / max(cfg['init_lr'], 1e-12), 1.0),
            final_div_factor=final_div_factor,
        )

        best_val_rmse = float("inf")
        best_state, best_epoch = None, -1
        history = {
            "train_mse": [], "train_rmse": [],
            "val_mse": [], "val_rmse": [],
            "best_epoch": None
        }

        for epoch in range(1, config['epochs'] + 1):
            # train_one_epoch still returns average MSE because criterion=nn.MSELoss()
            train_mse = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler)
            train_rmse = math.sqrt(train_mse)

            val_metrics = evaluate(model, val_loader, criterion, device, is_classification=False)
            val_mse, val_rmse = val_metrics["loss"], val_metrics["rmse"]

            history["train_mse"].append(train_mse)
            history["train_rmse"].append(train_rmse)
            history["val_mse"].append(val_mse)
            history["val_rmse"].append(val_rmse)

            # pick best by **validation RMSE**
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch

            if (epoch % max(1, config['log_frequency']) == 0) or epoch in (1, config['epochs']):
                print(f"[Epoch {epoch:03d}] train_rmse={train_rmse:.6f} "
                    f"val_rmse={val_rmse:.6f} (val_mse={val_mse:.6f})")

        # restore best-by-RMSE
        model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch
        all_best_states.append(deepcopy(best_state))
        all_norms.append((feature_mean.cpu().numpy(), feature_std.cpu().numpy()))


        test_metrics = evaluate(model, test_loader, criterion, device, is_classification=False)
        test_mse, test_rmse = test_metrics["loss"], test_metrics["rmse"]
        print(f"[Fold {fold+1}/{config['num_folds']}] best_epoch={best_epoch:03d} "
            f"| val_rmse={best_val_rmse:.6f} | test_rmse={test_rmse:.6f}")
        
        # ---- Save splits (original indices)
        train_idx = np.array(train_ds.indices, dtype=np.int64)
        val_idx   = np.array(val_ds.indices, dtype=np.int64)
        test_idx  = np.array(test_ds.indices, dtype=np.int64)
        np.savez(fold_dir / "splits.npz", train=train_idx, val=val_idx, test=test_idx)

        # ---- Save normalization used for this fold
        np.savez(
            fold_dir / "norm.npz",
            mean=feature_mean.detach().cpu().numpy(),
            std=feature_std.detach().cpu().numpy(),
        )

        # ---- Predictions for test set (aligned with original dataset indices)
        test_pred, test_y, test_orig_idx = predict_with_indices(model, test_loader, device, is_classification)

        # Save CSV: index, y_true, y_pred (for regression: single column)
        # (If multi-task later, you can expand columns.)
        csv_path = fold_dir / "preds_test.csv"
        with open(csv_path, "w") as f:
            f.write("orig_index,y_true,y_pred\n")
            for i, yt, yp in zip(test_orig_idx, test_y.reshape(-1), test_pred.reshape(-1)):
                f.write(f"{int(i)},{float(yt)},{float(yp)}\n")

        # ---- Save best checkpoint (state_dict + metadata)
        ckpt = {
            "model_state_dict": best_state,
            "config": cfg,
            "seed": seed,
            "fold": fold,
            "best_epoch": best_epoch,
            "best_val_rmse": float(best_val_rmse),
            "test_mse": float(test_mse),
            "test_rmse": float(test_rmse),
            "feature_mean": feature_mean.detach().cpu(),
            "feature_std": feature_std.detach().cpu(),
            "split_indices": {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            },
        }
        torch.save(ckpt, fold_dir / "checkpoint_best.pt")

        # Optional: save history as json (nice for plotting later)
        with open(fold_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"[Fold {fold+1}/{cfg['num_folds']}] best_epoch={best_epoch:03d} "
              f"| val_RMSE={best_val_rmse:.6f} | test_RMSE={test_rmse:.6f}")

    mses = [m[0] for m in all_test_metrics]
    rmses = [m[1] for m in all_test_metrics if m[1] is not None]
    summary = {
        'test_mse_mean': float(stats.mean(mses))  if mses else None,
        'test_mse_std':  float(stats.pstdev(mses)) if len(mses) > 1 else 0.0,
    }

    # RMSE is for regression; include it when we actually computed it
    if (not is_classification) and rmses:
        summary['test_rmse_mean'] = float(stats.mean(rmses))
        summary['test_rmse_std']  = float(stats.pstdev(rmses)) if len(rmses) > 1 else 0.0
        summary = {
            'test_rmse_mean': float(stats.mean(rmses))  if mses else None,
            'test_rmse_std':  float(stats.pstdev(rmses)) if len(mses) > 1 else 0.0,
        }
    return {
        # 'history': all_hist,
        # 'test_metrics': all_test_metrics,
        # 'test_predictions': all_test_preds,
        # 'test_indices': all_test_indices,
        'summary': summary,
        'checkpoints': all_best_states,   # NEW
        'norms': all_norms              
    }

def main():
    parser = argparse.ArgumentParser(
        description="Train polyBERT model on polymer property prediction"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to polyBERT embedding CSV (e.g., ../../Datasets/polyBERT/MD_300/density/homopolymer_density.csv)"
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory to save trained models and results"
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42). Fold i uses seed+i"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data_path.exists():
        print(f"ERROR: Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Update config with command-line args
    config['num_folds'] = args.num_folds
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    
    if not args.quiet:
        print(f"Loading data from: {args.data_path}")
    
    # Load polyBERT embeddings (600-dim) + target
    dataset = np.genfromtxt(
        args.data_path,
        delimiter=',',
        dtype=float,
        filling_values=np.nan,
        skip_header=1
    )
    
    X = dataset[:, :600]  # First 600 columns are polyBERT embeddings
    y = dataset[:, 600:]  # Remaining column(s) are target(s)
    y = np.ravel(y)
    
    # Filter out samples with NaN targets
    mask_X = np.isfinite(y)
    X = X[mask_X]
    y = y[mask_X]
    
    if not args.quiet:
        print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    
    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    results = run_repeated_experiments_from_arrays(
        X, y, config, 
        out_dir=str(args.save_dir),
        base_seed=args.seed
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("Summary:", results['summary'])
    print(f"Dataset: {args.data_path}")
    print(f"Results saved to: {args.save_dir}")

if __name__ == "__main__":
    main()

