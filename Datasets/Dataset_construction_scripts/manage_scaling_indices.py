#!/usr/bin/env python3
"""Export and verify compact canonical indices for scaling experiments."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_row_indices(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["row_index"])
    values = frame["row_index"].to_numpy(dtype=np.int64)
    if len(values) != len(np.unique(values)):
        raise ValueError(f"Duplicate row_index values in {path}")
    return values


def validate_partition(
    train_pool: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    n_rows: int,
    context: str,
) -> None:
    combined = np.concatenate([train_pool, val, test])
    if len(combined) != n_rows:
        raise ValueError(f"{context}: partition contains {len(combined)} rows, expected {n_rows}")
    if len(np.unique(combined)) != n_rows:
        raise ValueError(f"{context}: partition contains duplicate row indices")
    if not np.array_equal(np.sort(combined), np.arange(n_rows, dtype=np.int64)):
        raise ValueError(f"{context}: partition does not cover canonical rows 0..{n_rows - 1}")


def write_deterministic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an np.load-compatible archive without variable ZIP timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asarray(arrays[name], dtype=np.int64),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def experiment_relative_path(manifest: dict) -> Path:
    return (
        Path(manifest["dataset"])
        / manifest["property"]
        / Path(manifest["file"]).stem
        / f"seed_{manifest['seed']}"
    )


def export_experiment(
    split_manifest_path: Path,
    datasets_root: Path,
    output_root: Path,
    overwrite: bool,
) -> Path:
    split_base = split_manifest_path.parent
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    relative = experiment_relative_path(manifest)
    output_base = output_root / relative
    output_manifest_path = output_base / "manifest.json"

    canonical_relative = (
        Path("PSMILES") / manifest["dataset"] / manifest["property"] / manifest["file"]
    )
    canonical_path = datasets_root / canonical_relative
    if not canonical_path.is_file():
        raise FileNotFoundError(f"Canonical PSMILES source not found: {canonical_path}")

    canonical = pd.read_csv(canonical_path)
    target_column = manifest["target_column"]
    if list(canonical.columns) != ["smiles", target_column]:
        raise ValueError(
            f"Unexpected canonical columns in {canonical_path}: {list(canonical.columns)}"
        )
    if len(canonical) != manifest["n_rows"]:
        raise ValueError(
            f"Canonical row count {len(canonical)} does not match split manifest "
            f"{manifest['n_rows']}: {canonical_path}"
        )

    if output_manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Published index manifest already exists: {output_manifest_path}; use --overwrite"
        )

    fold_records = []
    for fold in range(manifest["num_folds"]):
        fold_dir = split_base / f"fold_{fold:02d}"
        train_pool = read_row_indices(fold_dir / "indices" / "train_pool.csv")
        val = read_row_indices(fold_dir / "indices" / "val.csv")
        test = read_row_indices(fold_dir / "indices" / "test.csv")
        validate_partition(train_pool, val, test, manifest["n_rows"], f"{relative}/fold_{fold:02d}")

        for train_size in manifest["train_sizes"]:
            materialized = read_row_indices(
                fold_dir / f"train_{train_size}" / "indices" / "train.csv"
            )
            if not np.array_equal(materialized, train_pool[:train_size]):
                raise ValueError(
                    f"{relative}/fold_{fold:02d}/train_{train_size} is not the expected "
                    "train-pool prefix"
                )

        indices_file = f"fold_{fold:02d}.npz"
        write_deterministic_npz(
            output_base / indices_file,
            train_pool=train_pool,
            val=val,
            test=test,
        )
        fold_records.append(
            {
                "fold": fold,
                "seed": manifest["seed"] + fold,
                "indices_file": indices_file,
                "n_train_pool": int(len(train_pool)),
                "n_val": int(len(val)),
                "n_test": int(len(test)),
            }
        )

    published = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "property": manifest["property"],
        "file": manifest["file"],
        "target_column": target_column,
        "canonical_source": str(canonical_relative),
        "canonical_sha256": sha256_file(canonical_path),
        "n_rows": manifest["n_rows"],
        "seed": manifest["seed"],
        "num_folds": manifest["num_folds"],
        "val_frac": manifest["val_frac"],
        "test_frac": manifest["test_frac"],
        "train_sizes": manifest["train_sizes"],
        "folds": fold_records,
    }
    output_base.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(published, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_manifest_path


def verify_experiment(
    manifest_path: Path,
    datasets_root: Path,
    allow_missing_sources: bool,
) -> None:
    base = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema version in {manifest_path}")

    canonical_path = datasets_root / manifest["canonical_source"]
    if canonical_path.exists():
        if sha256_file(canonical_path) != manifest["canonical_sha256"]:
            raise ValueError(f"Canonical source checksum mismatch: {canonical_path}")
        canonical = pd.read_csv(canonical_path)
        if len(canonical) != manifest["n_rows"]:
            raise ValueError(f"Canonical source row count mismatch: {canonical_path}")
        if list(canonical.columns) != ["smiles", manifest["target_column"]]:
            raise ValueError(f"Canonical source columns mismatch: {canonical_path}")
    elif not allow_missing_sources:
        raise FileNotFoundError(f"Canonical source not found: {canonical_path}")

    if len(manifest["folds"]) != manifest["num_folds"]:
        raise ValueError(f"Fold count mismatch in {manifest_path}")
    for expected_fold, fold_record in enumerate(manifest["folds"]):
        if fold_record.get("fold") != expected_fold:
            raise ValueError(
                f"Fold record {expected_fold} has fold={fold_record.get('fold')!r} "
                f"in {manifest_path}"
            )
        if fold_record.get("seed") != manifest["seed"] + expected_fold:
            raise ValueError(
                f"Fold {expected_fold} seed mismatch in {manifest_path}"
            )
        archive_path = base / fold_record["indices_file"]
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != {"train_pool", "val", "test"}:
                raise ValueError(f"Unexpected arrays in {archive_path}: {archive.files}")
            train_pool = archive["train_pool"]
            val = archive["val"]
            test = archive["test"]
        validate_partition(
            train_pool,
            val,
            test,
            manifest["n_rows"],
            str(archive_path),
        )
        for key, values in (
            ("n_train_pool", train_pool),
            ("n_val", val),
            ("n_test", test),
        ):
            if fold_record[key] != len(values):
                raise ValueError(f"{key} mismatch in {archive_path}")
        if max(manifest["train_sizes"]) > len(train_pool):
            raise ValueError(f"Training size exceeds train pool in {archive_path}")


def discover_manifests(root: Path) -> list[Path]:
    return sorted(root.rglob("manifest.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Export compact indices from materialized splits.")
    export.add_argument("--split-root", type=Path, default=REPO_ROOT / "Datasets/scaling_splits")
    export.add_argument("--datasets-root", type=Path, default=REPO_ROOT / "Datasets")
    export.add_argument("--output-root", type=Path, default=REPO_ROOT / "Datasets/scaling_indices")
    export.add_argument("--overwrite", action="store_true")

    verify = subparsers.add_parser("verify", help="Verify published compact index packages.")
    verify.add_argument("--indices-root", type=Path, default=REPO_ROOT / "Datasets/scaling_indices")
    verify.add_argument("--datasets-root", type=Path, default=REPO_ROOT / "Datasets")
    verify.add_argument("--allow-missing-sources", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        manifests = discover_manifests(args.split_root.resolve())
        if not manifests:
            raise FileNotFoundError(f"No split manifests found under {args.split_root}")
        for manifest_path in manifests:
            output = export_experiment(
                manifest_path,
                args.datasets_root.resolve(),
                args.output_root.resolve(),
                args.overwrite,
            )
            print(f"Exported {output}")
        print(f"Exported {len(manifests)} compact scaling-index packages.")
    else:
        manifests = discover_manifests(args.indices_root.resolve())
        if not manifests:
            raise FileNotFoundError(f"No index manifests found under {args.indices_root}")
        for manifest_path in manifests:
            verify_experiment(
                manifest_path,
                args.datasets_root.resolve(),
                args.allow_missing_sources,
            )
        print(f"Verified {len(manifests)} compact scaling-index packages.")


if __name__ == "__main__":
    main()
