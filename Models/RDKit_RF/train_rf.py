#!/usr/bin/env python3
"""
Train Random Forest models on RDKit descriptor datasets.

This script:
1. Reads a pre-computed RDKit descriptor CSV from Datasets/RDKit_descriptors/
2. Creates 5-fold cross-validation splits (80/10/10 train/val/test)
3. Trains a Random Forest model on each fold
4. Saves models, predictions, and scores for each fold
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def sanitize_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove problematic columns and handle inf/nan values.
    
    This matches the preprocessing done in Datasets/Dataset_construction_scripts/
    """
    # Drop problematic columns that RDKit sometimes produces
    cols_to_drop = [
        'MaxPartialCharge', 'MinPartialCharge', 
        'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'Ipc'
    ]
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
    
    # Convert to numeric and replace inf with nan
    X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    
    all_nan_columns = X.columns[X.isna().all()].tolist()
    if all_nan_columns:
        print(f"Dropping all-NaN descriptor columns: {', '.join(all_nan_columns)}")
        X = X.drop(columns=all_nan_columns)

    return X, all_nan_columns


def load_and_prepare_data(data_path: Path):
    """Load RDKit descriptor dataset and separate features from target."""
    df = pd.read_csv(data_path)
    
    # Identify target column(s) - everything except descriptors
    # RDKit descriptor datasets have all numeric columns except the target
    # The target column is typically the property name (e.g., 'density', 'Tg', etc.)
    
    # Find the target column (usually the last column, or first non-descriptor column)
    # For safety, we'll assume all columns except numeric descriptors are targets
    # In practice, the datasets have descriptors + 1 target column
    
    # Get column names - descriptors are typically MolWt, LogP, etc.
    # Target is the property name
    target_cols = []
    feature_cols = []
    
    for col in df.columns:
        # Common RDKit descriptor names start with capital letters or specific patterns
        # Target columns are typically property names
        # Simple heuristic: if column has many unique values and is numeric, it's likely a descriptor
        if df[col].dtype in ['float64', 'int64']:
            unique_ratio = df[col].nunique() / len(df)
            # Descriptors usually have high variance, targets might too
            # Better: check if column name matches known descriptor names
            # For now, assume last column is target
            if col == df.columns[-1]:
                target_cols.append(col)
            else:
                feature_cols.append(col)
    
    if len(target_cols) == 0:
        print(f"ERROR: Could not identify target column in {data_path}")
        print(f"Columns: {df.columns.tolist()}")
        sys.exit(1)
    
    X = df[feature_cols]
    y = df[target_cols].values
    
    # Sanitize features
    X, all_nan_columns = sanitize_features(X)
    feature_cols = X.columns.tolist()
    
    return X, y, feature_cols, target_cols, all_nan_columns


def train_random_forest(X_train: pd.DataFrame, y_train: np.ndarray, random_state: int = 42):
    """
    Train a Random Forest model with preprocessing pipeline.
    
    Pipeline:
    1. Impute missing values with median
    2. Train Random Forest regressor
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        ))
    ])
    
    pipeline.fit(X_train, y_train.ravel())
    return pipeline


def evaluate_model(model, X: pd.DataFrame, y: np.ndarray, split_name: str):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    y_true = y.ravel()
    
    metrics = {
        'split': split_name,
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'n_samples': len(y_true)
    }
    
    return metrics, y_pred


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest models on RDKit descriptor datasets"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to RDKit descriptor CSV (e.g., ../../Datasets/RDKit_descriptors/MD_300/density/homopolymer_density.csv)"
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
    
    # Load data
    if not args.quiet:
        print(f"Loading data from: {args.data_path}")
    
    X, y, feature_cols, target_cols, all_nan_columns = load_and_prepare_data(args.data_path)
    
    if not args.quiet:
        print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
        print(f"Target column(s): {target_cols}")
    
    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature names for prediction
    with open(args.save_dir / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(args.save_dir / "preprocessing.json", "w") as f:
        json.dump({"dropped_all_nan_columns": all_nan_columns}, f, indent=2)
    
    # Train models for each fold
    all_test_metrics = []
    
    for fold in range(args.num_folds):
        fold_seed = args.seed + fold
        fold_dir = args.save_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        if not args.quiet:
            print(f"\n{'='*60}")
            print(f"Training fold {fold} (seed={fold_seed})")
            print(f"{'='*60}")
        
        # Split: 80% train, 10% val, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=fold_seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=fold_seed
        )
        
        # Save splits
        split_info = {
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'seed': fold_seed
        }
        with open(fold_dir / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)
        
        # Train model
        if not args.quiet:
            print(f"Training on {len(X_train)} samples...")
        
        model = train_random_forest(X_train, y_train, random_state=fold_seed)
        
        # Evaluate on all splits
        train_metrics, train_pred = evaluate_model(model, X_train, y_train, 'train')
        val_metrics, val_pred = evaluate_model(model, X_val, y_val, 'val')
        test_metrics, test_pred = evaluate_model(model, X_test, y_test, 'test')
        
        all_test_metrics.append(test_metrics)
        
        # Save metrics
        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        if not args.quiet:
            print(f"Train RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}")
            print(f"Val   RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}")
            print(f"Test  RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
        
        # Save model
        model_path = fold_dir / "rf.joblib"
        dump(model, model_path)
        
        if not args.quiet:
            print(f"Model saved to: {model_path}")
    
    # Compute overall statistics
    test_rmses = [m['rmse'] for m in all_test_metrics]
    test_maes = [m['mae'] for m in all_test_metrics]
    
    overall_stats = {
        'test_rmse_mean': float(np.mean(test_rmses)),
        'test_rmse_std': float(np.std(test_rmses)),
        'test_mae_mean': float(np.mean(test_maes)),
        'test_mae_std': float(np.std(test_maes)),
        'num_folds': args.num_folds
    }
    
    with open(args.save_dir / "overall_metrics.json", "w") as f:
        json.dump(overall_stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Test RMSE: {overall_stats['test_rmse_mean']:.4f} ± {overall_stats['test_rmse_std']:.4f}")
    print(f"Test MAE:  {overall_stats['test_mae_mean']:.4f} ± {overall_stats['test_mae_std']:.4f}")
    print(f"\nResults saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
