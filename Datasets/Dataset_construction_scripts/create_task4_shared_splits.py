#!/usr/bin/env python3
"""Materialize the aligned PolyBench26 Task 4 architecture-transfer protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from polymer_conversions import wPSMILES_to_PSMILES_alternating
from rdkit_descriptor_cache import DescriptorCache, default_cache_path
from task4_architecture_transfer import parse_wpsmiles


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "Datasets"
DEFAULT_OUTPUT = DATASETS / "task4_architecture_transfer" / "generated" / "shared_splits"
POLYVERSE_PSMILES = DATASETS / "PSMILES/polyVERSE/electron_affinity/electron_affinity_data_polymers_v4.csv"
POLYVERSE_WPSMILES = DATASETS / "wPSMILES/polyVERSE/electron_affinity/electron_affinity_data_polymers_v4.csv"
VIPEA_ROOT = DATASETS / "wPSMILES/Vipea/EA"
TARGET = "electron_affinity"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def read_vipea(path: Path, architecture: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"VIPEA wPSMILES input is missing: {path}. Run "
            "python Datasets/Dataset_construction_scripts/process_Vipea_data.py first."
        )
    frame = pd.read_csv(path)
    require_columns(frame, {"smiles", "EA"}, path)
    result = pd.DataFrame(
        {
            "psmiles": frame["smiles"].map(wPSMILES_to_PSMILES_alternating),
            "wpsmiles": frame["smiles"],
            TARGET: pd.to_numeric(frame["EA"], errors="raise"),
            "architecture": architecture,
            "source": "Vipea",
            "source_row": np.arange(len(frame), dtype=np.int64),
        }
    )
    if result["psmiles"].isna().any() or not np.isfinite(result[TARGET]).all():
        raise ValueError(f"Invalid PSMILES conversion or EA target in {path}")
    return result


def read_polyverse(psmiles_path: Path, wpsmiles_path: Path) -> pd.DataFrame:
    psmiles = pd.read_csv(psmiles_path)
    wpsmiles = pd.read_csv(wpsmiles_path)
    require_columns(psmiles, {"smiles", "electron_aff"}, psmiles_path)
    require_columns(wpsmiles, {"smiles", "electron_aff"}, wpsmiles_path)
    if len(psmiles) != len(wpsmiles):
        raise ValueError("polyVERSE PSMILES and wPSMILES row counts differ")
    p_target = pd.to_numeric(psmiles["electron_aff"], errors="raise").to_numpy()
    w_target = pd.to_numeric(wpsmiles["electron_aff"], errors="raise").to_numpy()
    if not np.array_equal(p_target, w_target):
        raise ValueError("polyVERSE PSMILES and wPSMILES targets are not row-aligned")
    return pd.DataFrame({
        "psmiles": psmiles["smiles"], "wpsmiles": wpsmiles["smiles"], TARGET: p_target,
        "architecture": "homopolymer", "source": "polyVERSE",
        "source_row": np.arange(len(psmiles), dtype=np.int64),
    })


def canonical_training_frame(vipea_path: Path, polyverse_psmiles: Path, polyverse_wpsmiles: Path) -> pd.DataFrame:
    """Return the 6,478-row PSMILES-aligned training table.

    The 28 duplicate VIPEA PSMILES pairs get their mean EA target; the earliest
    source row supplies the representative wPSMILES.
    """
    source = pd.concat(
        [read_vipea(vipea_path, "alternating"), read_polyverse(polyverse_psmiles, polyverse_wpsmiles)],
        ignore_index=True,
    ).sort_values(["psmiles", "source", "source_row"], kind="stable")
    canonical = source.groupby("psmiles", sort=False, as_index=False).agg(
        wpsmiles=("wpsmiles", "first"), electron_affinity=(TARGET, "mean"),
        architecture=("architecture", "first"), source=("source", "first"),
        source_rows=("source_row", lambda values: ";".join(map(str, values))),
        n_source_rows=("source_row", "size"),
    )
    counts = canonical["architecture"].value_counts().to_dict()
    if counts != {"alternating": 6110, "homopolymer": 368}:
        raise ValueError(f"Unexpected Task 4 canonical architecture counts: {counts}")
    canonical.insert(0, "row_index", np.arange(len(canonical), dtype=np.int64))
    return canonical


def holdout_frame(vipea_root: Path) -> pd.DataFrame:
    frames = []
    for architecture in ("random", "block"):
        frame = read_vipea(vipea_root / f"{architecture}_EA.csv", architecture)
        parsed = frame["wpsmiles"].map(parse_wpsmiles)
        frame[["psmiles_a", "psmiles_b", "molar_fraction_a", "molar_fraction_b"]] = pd.DataFrame(parsed.tolist(), index=frame.index)
        frames.append(frame)
    holdout = pd.concat(frames, ignore_index=True)
    holdout.insert(0, "holdout_row_index", np.arange(len(holdout), dtype=np.int64))
    counts = holdout["architecture"].value_counts().to_dict()
    if counts != {"random": 18414, "block": 18414}:
        raise ValueError(f"Unexpected Task 4 OOD holdout architecture counts: {counts}")
    return holdout


def assign_folds(train: pd.DataFrame, num_folds: int, seed: int) -> np.ndarray:
    """Assign whole PSMILES groups to balanced partitions within architecture."""
    if not {"psmiles", "architecture"} <= set(train):
        raise ValueError("Training data requires psmiles and architecture columns")
    assignments = np.empty(len(train), dtype=np.int64)
    for architecture, rows in train.groupby("architecture", sort=True):
        groups = [(name, rows.index[positions].to_numpy()) for name, positions in rows.groupby("psmiles", sort=True).indices.items()]
        if len(groups) < num_folds:
            raise ValueError(f"Not enough {architecture} PSMILES groups for {num_folds} folds")
        rng = np.random.default_rng(seed + sum(map(ord, architecture)))
        rng.shuffle(groups)
        loads = np.zeros(num_folds, dtype=np.int64)
        for _, indices in sorted(groups, key=lambda item: len(item[1]), reverse=True):
            fold = int(np.argmin(loads))
            assignments[indices] = fold
            loads[fold] += len(indices)
    return assignments


def index_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return pd.DataFrame({
        "order": np.arange(len(frame), dtype=np.int64), "row_index": frame["row_index"].to_numpy(),
        "split": split, "smiles": frame["psmiles"].to_numpy(), TARGET: frame[TARGET].to_numpy(),
    })


def embedding_frame(frame: pd.DataFrame, dictionary_path: Path) -> pd.DataFrame:
    with dictionary_path.open("rb") as handle:
        dictionary = pickle.load(handle)
    missing = [smiles for smiles in frame["psmiles"] if smiles not in dictionary]
    if missing:
        raise ValueError(f"polyBERT dictionary lacks {len(missing)} Task 4 PSMILES values (first: {missing[0]!r}).")
    values = np.asarray([dictionary[smiles] for smiles in frame["psmiles"]], dtype=float)
    if values.ndim != 2 or values.shape[1] != 600:
        raise ValueError("Task 4 polyBERT embeddings must have exactly 600 columns")
    result = pd.DataFrame(values)
    result[TARGET] = frame[TARGET].to_numpy()
    return result


def write_representations(frame: pd.DataFrame, split_dir: Path, split: str, cache: DescriptorCache | None, dictionary_path: Path | None) -> None:
    index_frame(frame, split).to_csv(split_dir / "indices" / f"{split}.csv", index=False)
    frame[["psmiles", TARGET]].rename(columns={"psmiles": "smiles"}).to_csv(split_dir / "PSMILES" / f"{split}.csv", index=False)
    frame[["wpsmiles", TARGET]].rename(columns={"wpsmiles": "smiles"}).to_csv(split_dir / "wPSMILES" / f"{split}.csv", index=False)
    if cache is not None:
        features, invalid = cache.descriptor_frame(frame["psmiles"])
        if invalid:
            raise ValueError(f"RDKit could not parse Task 4 PSMILES rows {invalid[:10]}")
        pd.concat([features, frame[[TARGET]].reset_index(drop=True)], axis=1).to_csv(split_dir / "RDKit_descriptors" / f"{split}.csv", index=False)
    if dictionary_path is not None:
        embedding_frame(frame, dictionary_path).to_csv(split_dir / "polyBERT" / f"{split}.csv", index=False)


def create_shared_splits(output_dir: Path, vipea_root: Path = VIPEA_ROOT, polyverse_psmiles: Path = POLYVERSE_PSMILES, polyverse_wpsmiles: Path = POLYVERSE_WPSMILES, num_folds: int = 5, seed: int = 42, dictionary_path: Path | None = None, with_rdkit: bool = True, overwrite: bool = False) -> None:
    if num_folds != 5:
        raise ValueError("Task 4 is defined as five shared folds.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    train = canonical_training_frame(vipea_root / "alternating_EA.csv", polyverse_psmiles, polyverse_wpsmiles)
    holdout = holdout_frame(vipea_root)
    train.to_csv(output_dir / "canonical_train.csv", index=False)
    holdout.to_csv(output_dir / "architecture_holdout.csv", index=False)
    holdout[["wpsmiles", TARGET]].rename(columns={"wpsmiles": "smiles"}).to_csv(
        output_dir / "ood_wpsmiles.csv", index=False
    )
    pd.DataFrame({"psmiles": pd.unique(pd.concat([holdout["psmiles_a"], holdout["psmiles_b"]], ignore_index=True))}).to_csv(
        output_dir / "ood_components.csv", index=False
    )
    train = train.copy()
    train["partition"] = assign_folds(train, num_folds * 2, seed)
    cache = DescriptorCache(default_cache_path(DATASETS)) if with_rdkit else None
    manifests = []
    try:
        for fold in range(num_folds):
            train_frame = train[~train["partition"].isin([fold, fold + num_folds])].drop(columns="partition")
            val_frame = train[train["partition"] == fold + num_folds].drop(columns="partition")
            test_frame = train[train["partition"] == fold].drop(columns="partition")
            run_dir = output_dir / f"fold_{fold:02d}" / f"train_{len(train_frame)}"
            for directory in ("indices", "PSMILES", "wPSMILES", "RDKit_descriptors", "polyBERT"):
                (run_dir / directory).mkdir(parents=True, exist_ok=True)
            for name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame)):
                write_representations(frame, run_dir, name, cache, dictionary_path)
            manifests.append({"fold": fold, "n_train": len(train_frame), "n_val": len(val_frame), "n_in_distribution_test": len(test_frame)})
    finally:
        if cache is not None:
            cache.close()
    inputs = [vipea_root / f"{name}_EA.csv" for name in ("alternating", "random", "block")] + [polyverse_psmiles, polyverse_wpsmiles]
    manifest = {
        "schema_version": 2, "seed": seed, "num_folds": num_folds,
        "train_fraction": 0.8, "validation_fraction": 0.1, "in_distribution_test_fraction": 0.1,
        "target_column": TARGET, "canonical_rows": len(train),
        "canonical_architecture_counts": train["architecture"].value_counts().sort_index().to_dict(),
        "deduplication": "group by PSMILES; mean EA; earliest source wPSMILES",
        "fixed_holdout": "architecture_holdout.csv",
        "holdout_architecture_counts": holdout["architecture"].value_counts().sort_index().to_dict(),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs}, "folds": manifests,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vipea-root", type=Path, default=VIPEA_ROOT)
    parser.add_argument("--polyverse-psmiles", type=Path, default=POLYVERSE_PSMILES)
    parser.add_argument("--polyverse-wpsmiles", type=Path, default=POLYVERSE_WPSMILES)
    parser.add_argument("--polybert-dictionary", type=Path, help="Optional 600-dimensional PSMILES embedding dictionary")
    parser.add_argument("--without-rdkit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    create_shared_splits(args.output_dir, args.vipea_root, args.polyverse_psmiles, args.polyverse_wpsmiles, args.num_folds, args.seed, args.polybert_dictionary, not args.without_rdkit, args.overwrite)


if __name__ == "__main__":
    main()
