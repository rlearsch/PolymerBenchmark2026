#!/usr/bin/env python3
"""Create explicit dataset-size scaling splits across representations.

The PSMILES dataset is treated as the canonical chemical ordering. PSMILES,
wPSMILES, and polyBERT are validated for row-wise target alignment. RDKit
descriptors are regenerated from the canonical PSMILES rows so chemical identity
is preserved even if an existing full RDKit descriptor CSV is not row-aligned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from manage_scaling_indices import sha256_file, validate_partition
from rdkit_descriptor_cache import DescriptorCache, default_cache_path

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - handled at runtime when RDKit is requested
    Chem = None


DEFAULT_TRAIN_SIZES = [300, 1000, 3000, 10000]
DEFAULT_REPRESENTATIONS = ["PSMILES", "wPSMILES", "polyBERT", "RDKit_descriptors"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed fold/size scaling datasets for all representations."
    )
    parser.add_argument("--dataset", default="MD_5000", help="Dataset directory, e.g. MD_5000")
    parser.add_argument("--property", required=True, help="Property directory, e.g. Cp")
    parser.add_argument("--file", required=True, help="CSV filename, e.g. alternating_Cp.csv")
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_TRAIN_SIZES,
        help="Nested training sizes to generate.",
    )
    parser.add_argument("--num-folds", type=int, default=5, help="Number of repeated folds.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction per fold.")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Test fraction per fold.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=repo_root() / "Datasets",
        help="Root containing PSMILES, wPSMILES, polyBERT, etc.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root() / "Datasets" / "scaling_splits",
        help="Directory where scaling split datasets are written.",
    )
    parser.add_argument(
        "--descriptor-cache",
        type=Path,
        help=(
            "SQLite descriptor cache path (default: "
            "Datasets/RDKit_descriptors/.descriptor_cache.sqlite3)."
        ),
    )
    parser.add_argument(
        "--indices-root",
        type=Path,
        default=repo_root() / "Datasets" / "scaling_indices",
        help=(
            "Root containing published compact indices. A matching package is used "
            "automatically; otherwise indices are generated from --seed."
        ),
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=DEFAULT_REPRESENTATIONS,
        choices=DEFAULT_REPRESENTATIONS,
        help="Representations to write.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split CSVs. Existing files are otherwise left unchanged.",
    )
    return parser.parse_args()


def target_columns(df: pd.DataFrame, representation: str) -> list[str]:
    if representation in {"PSMILES", "wPSMILES"}:
        return [col for col in df.columns if col != "smiles"]
    return [df.columns[-1]]


def require_single_target(df: pd.DataFrame, representation: str) -> str:
    targets = target_columns(df, representation)
    if len(targets) != 1:
        raise ValueError(
            f"Expected exactly one target column for {representation}; found {targets}."
        )
    return targets[0]


def dataset_path(root: Path, representation: str, dataset: str, prop: str, filename: str) -> Path:
    return root / representation / dataset / prop / filename


def load_and_validate_inputs(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], str]:
    frames: dict[str, pd.DataFrame] = {}

    psmiles_path = dataset_path(args.datasets_root, "PSMILES", args.dataset, args.property, args.file)
    if not psmiles_path.exists():
        raise FileNotFoundError(f"Canonical PSMILES file not found: {psmiles_path}")

    psmiles = pd.read_csv(psmiles_path)
    if "smiles" not in psmiles.columns:
        raise ValueError(f"Canonical PSMILES file must contain a 'smiles' column: {psmiles_path}")
    target_col = require_single_target(psmiles, "PSMILES")
    frames["PSMILES"] = psmiles

    canonical_y = pd.to_numeric(psmiles[target_col], errors="raise").to_numpy()
    if not np.isfinite(canonical_y).all():
        raise ValueError("Canonical PSMILES target contains NaN or infinite values.")

    for representation in args.representations:
        if representation in {"PSMILES", "RDKit_descriptors"}:
            continue

        path = dataset_path(args.datasets_root, representation, args.dataset, args.property, args.file)
        if not path.exists():
            raise FileNotFoundError(f"{representation} file not found: {path}")

        df = pd.read_csv(path)
        rep_target = require_single_target(df, representation)
        if len(df) != len(psmiles):
            raise ValueError(
                f"{representation} row count {len(df)} does not match PSMILES row count {len(psmiles)}."
            )
        if rep_target != target_col:
            raise ValueError(
                f"{representation} target column '{rep_target}' does not match canonical '{target_col}'."
            )

        rep_y = pd.to_numeric(df[rep_target], errors="raise").to_numpy()
        if not np.allclose(rep_y, canonical_y, rtol=0.0, atol=1e-12, equal_nan=False):
            raise ValueError(f"{representation} target values are not row-aligned with PSMILES.")
        frames[representation] = df

    if "RDKit_descriptors" in args.representations and Chem is None:
        raise ImportError(
            "RDKit is required to generate RDKit_descriptors. Install rdkit or omit "
            "RDKit_descriptors from --representations."
        )

    return frames, target_col


def build_rdkit_frame(
    psmiles: pd.DataFrame, target_col: str, cache_path: Path | None = None
) -> pd.DataFrame:
    print(f"Generating RDKit descriptors from {len(psmiles)} canonical PSMILES rows...")
    if cache_path is None:
        # Kept for direct library callers; production CLI paths always supply the
        # shared persistent cache below.
        from rdkit_descriptor_cache import calculate_descriptors, sanitize_descriptors

        descriptor_rows = psmiles["smiles"].apply(calculate_descriptors)
        features = sanitize_descriptors(pd.DataFrame(descriptor_rows))
        invalid_rows = [int(index) for index, values in descriptor_rows.items() if not values]
    else:
        with DescriptorCache(cache_path) as cache:
            features, invalid_rows = cache.descriptor_frame(psmiles["smiles"])
    if invalid_rows:
        preview = invalid_rows[:10]
        raise ValueError(
            "RDKit could not parse canonical PSMILES rows "
            f"{preview}{'...' if len(invalid_rows) > len(preview) else ''}."
        )
    return pd.concat([features, psmiles[[target_col]].reset_index(drop=True)], axis=1)


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def split_indices(n_rows: int, fold: int, seed: int, val_frac: float, test_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("Require 0 < val_frac, 0 < test_frac, and val_frac + test_frac < 1.")

    rng = np.random.default_rng(seed + fold)
    shuffled = np.arange(n_rows, dtype=np.int64)
    rng.shuffle(shuffled)

    n_test = int(round(n_rows * test_frac))
    n_val = int(round(n_rows * val_frac))
    if n_test <= 0 or n_val <= 0:
        raise ValueError("Validation and test splits must each contain at least one row.")

    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train_pool = shuffled[n_test + n_val :]
    return train_pool, val, test


def indices_frame(indices: Iterable[int], split: str, psmiles: pd.DataFrame, target_col: str) -> pd.DataFrame:
    idx = np.asarray(list(indices), dtype=np.int64)
    return pd.DataFrame(
        {
            "order": np.arange(len(idx), dtype=np.int64),
            "row_index": idx,
            "split": split,
            "smiles": psmiles.iloc[idx]["smiles"].to_numpy(),
            target_col: psmiles.iloc[idx][target_col].to_numpy(),
        }
    )


def output_base(args: argparse.Namespace) -> Path:
    return args.output_root / args.dataset / args.property / Path(args.file).stem / f"seed_{args.seed}"


def load_published_indices(
    args: argparse.Namespace,
    psmiles_path: Path,
    n_rows: int,
    target_col: str,
) -> tuple[dict, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] | None:
    base = (
        args.indices_root
        / args.dataset
        / args.property
        / Path(args.file).stem
        / f"seed_{args.seed}"
    )
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": args.dataset,
        "property": args.property,
        "file": args.file,
        "target_column": target_col,
        "n_rows": n_rows,
        "seed": args.seed,
        "num_folds": args.num_folds,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Published index manifest {manifest_path} has {key}={manifest.get(key)!r}; "
                f"expected {value!r}."
            )
    if not np.isclose(manifest["val_frac"], args.val_frac) or not np.isclose(
        manifest["test_frac"], args.test_frac
    ):
        raise ValueError(f"Published index fractions do not match requested fractions: {manifest_path}")
    if not set(args.train_sizes) <= set(manifest["train_sizes"]):
        raise ValueError(f"Requested train sizes are absent from {manifest_path}")
    if sha256_file(psmiles_path) != manifest["canonical_sha256"]:
        raise ValueError(f"Canonical PSMILES checksum does not match {manifest_path}")

    folds = []
    if len(manifest["folds"]) != args.num_folds:
        raise ValueError(f"Published fold count does not match {args.num_folds}: {manifest_path}")
    for expected_fold, fold_record in enumerate(manifest["folds"]):
        if fold_record.get("fold") != expected_fold:
            raise ValueError(
                f"Published fold record {expected_fold} has fold="
                f"{fold_record.get('fold')!r}: {manifest_path}"
            )
        if fold_record.get("seed") != args.seed + expected_fold:
            raise ValueError(
                f"Published fold {expected_fold} seed does not match "
                f"{args.seed + expected_fold}: {manifest_path}"
            )
        archive_path = base / fold_record["indices_file"]
        with np.load(archive_path, allow_pickle=False) as archive:
            train_pool = archive["train_pool"].astype(np.int64, copy=True)
            val = archive["val"].astype(np.int64, copy=True)
            test = archive["test"].astype(np.int64, copy=True)
        validate_partition(train_pool, val, test, n_rows, str(archive_path))
        folds.append((train_pool, val, test))
    return manifest, folds


def main() -> None:
    args = parse_args()
    args.datasets_root = args.datasets_root.resolve()
    args.output_root = args.output_root.resolve()
    args.indices_root = args.indices_root.resolve()
    args.descriptor_cache = (
        args.descriptor_cache.resolve()
        if args.descriptor_cache is not None
        else default_cache_path(args.datasets_root)
    )
    args.train_sizes = sorted(set(args.train_sizes))
    if not args.train_sizes or any(size <= 0 for size in args.train_sizes):
        raise ValueError("Training sizes must be positive integers.")
    if args.num_folds <= 0:
        raise ValueError("--num-folds must be a positive integer.")

    frames, target_col = load_and_validate_inputs(args)
    psmiles = frames["PSMILES"]
    n_rows = len(psmiles)
    max_train_size = max(args.train_sizes)
    psmiles_path = dataset_path(
        args.datasets_root, "PSMILES", args.dataset, args.property, args.file
    )
    published = load_published_indices(args, psmiles_path, n_rows, target_col)

    rdkit_frame = None
    if "RDKit_descriptors" in args.representations:
        rdkit_frame = build_rdkit_frame(psmiles, target_col, args.descriptor_cache)
        frames["RDKit_descriptors"] = rdkit_frame

    base = output_base(args)
    manifest = {
        "dataset": args.dataset,
        "property": args.property,
        "file": args.file,
        "target_column": target_col,
        "seed": args.seed,
        "num_folds": args.num_folds,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "train_sizes": args.train_sizes,
        "representations": args.representations,
        "n_rows": n_rows,
        "split_source": "published_indices" if published else "generated_from_seed",
        "splits": [],
        "source_paths": {
            rep: str(Path(rep) / args.dataset / args.property / args.file)
            for rep in args.representations
            if rep != "RDKit_descriptors"
        },
        "rdkit_descriptor_cache": (
            str(args.descriptor_cache.relative_to(args.datasets_root))
            if "RDKit_descriptors" in args.representations
            and args.descriptor_cache.is_relative_to(args.datasets_root)
            else None
        ),
        "notes": [
            "PSMILES is the canonical chemical order.",
            "wPSMILES and polyBERT were validated against PSMILES targets row-by-row.",
            "RDKit descriptors were regenerated from canonical PSMILES rows when requested.",
        ],
    }

    for fold in range(args.num_folds):
        if published:
            train_pool, val, test = published[1][fold]
        else:
            train_pool, val, test = split_indices(
                n_rows, fold, args.seed, args.val_frac, args.test_frac
            )
        if len(train_pool) < max_train_size:
            raise ValueError(
                f"Fold {fold} train pool has {len(train_pool)} rows, smaller than requested "
                f"max train size {max_train_size}."
            )

        fold_dir = base / f"fold_{fold:02d}"
        write_csv(indices_frame(train_pool, "train_pool", psmiles, target_col), fold_dir / "indices" / "train_pool.csv", args.overwrite)
        write_csv(indices_frame(val, "val", psmiles, target_col), fold_dir / "indices" / "val.csv", args.overwrite)
        write_csv(indices_frame(test, "test", psmiles, target_col), fold_dir / "indices" / "test.csv", args.overwrite)

        fold_info = {
            "fold": fold,
            "seed": args.seed + fold,
            "n_train_pool": int(len(train_pool)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "train_sizes": [],
        }

        for train_size in args.train_sizes:
            train = train_pool[:train_size]
            size_dir = fold_dir / f"train_{train_size}"

            write_csv(indices_frame(train, "train", psmiles, target_col), size_dir / "indices" / "train.csv", args.overwrite)
            write_csv(indices_frame(val, "val", psmiles, target_col), size_dir / "indices" / "val.csv", args.overwrite)
            write_csv(indices_frame(test, "test", psmiles, target_col), size_dir / "indices" / "test.csv", args.overwrite)

            for representation in args.representations:
                frame = frames[representation]
                rep_dir = size_dir / representation
                write_csv(frame.iloc[train].reset_index(drop=True), rep_dir / "train.csv", args.overwrite)
                write_csv(frame.iloc[val].reset_index(drop=True), rep_dir / "val.csv", args.overwrite)
                write_csv(frame.iloc[test].reset_index(drop=True), rep_dir / "test.csv", args.overwrite)

            fold_info["train_sizes"].append(
                {
                    "train_size": int(train_size),
                    "n_train": int(len(train)),
                    "n_val": int(len(val)),
                    "n_test": int(len(test)),
                    "path": str(size_dir.relative_to(base)),
                }
            )

        manifest["splits"].append(fold_info)

    base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "manifest.json"
    if not manifest_path.exists() or args.overwrite:
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)

    print(f"Wrote scaling splits to: {base}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
