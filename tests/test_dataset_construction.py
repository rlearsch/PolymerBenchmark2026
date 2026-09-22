from __future__ import annotations

import json
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import RDLogger


REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION_DIR = REPO_ROOT / "Datasets" / "Dataset_construction_scripts"
sys.path.insert(0, str(CONSTRUCTION_DIR))

from polymer_conversions import (  # noqa: E402
    PSMILES_to_wPSMILES,
    construct_alternating_wPSMILES,
    construct_random_wPSMILES,
    wPSMILES_to_PSMILES_alternating,
    wPSMILES_to_PSMILES_homopolymer_simple,
)
from create_scaling_splits import build_rdkit_frame, split_indices  # noqa: E402
from process_Vipea_data import process_vipea_data  # noqa: E402


def run_cli(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CONSTRUCTION_DIR / script), *args],
        cwd=cwd,
        text=True,
        capture_output=True,
    )


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ConversionUnitTests(unittest.TestCase):
    def test_homopolymer_round_trip(self) -> None:
        psmiles = "*CC*"
        weighted = PSMILES_to_wPSMILES(psmiles)
        self.assertEqual(
            weighted,
            "[*:1]CC[*:2]|1|<1-2:1:1<1-1:0:0<2-2:0:0",
        )
        self.assertEqual(wPSMILES_to_PSMILES_homopolymer_simple(weighted), psmiles)

    def test_copolymer_conversions(self) -> None:
        alternating = construct_alternating_wPSMILES("0.5,0.5", "*CC*,*O*")
        self.assertIn("[*:1]", alternating)
        self.assertIn("[*:4]", alternating)
        self.assertIn("|0.5|0.5|", alternating)
        converted = wPSMILES_to_PSMILES_alternating(alternating)
        self.assertEqual(converted.count("*"), 2)

        random = construct_random_wPSMILES("0.25,0.75", "*CC*,*O*")
        self.assertIn("|0.25|0.75|", random)
        self.assertIn("<1-1:0.25:0.25", random)

    def test_split_indices_are_deterministic_and_disjoint(self) -> None:
        first = split_indices(40, fold=2, seed=42, val_frac=0.2, test_frac=0.2)
        second = split_indices(40, fold=2, seed=42, val_frac=0.2, test_frac=0.2)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        train, val, test = (set(values.tolist()) for values in first)
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(val.isdisjoint(test))
        self.assertEqual(len(train | val | test), 40)
        with self.assertRaisesRegex(ValueError, "Require 0 < val_frac"):
            split_indices(40, fold=0, seed=42, val_frac=0.0, test_frac=0.2)

    def test_invalid_psmiles_are_rejected_before_descriptor_generation(self) -> None:
        frame = pd.DataFrame(
            {"smiles": ["*CC*", "not valid smiles"], "target": [1.0, 2.0]}
        )
        RDLogger.DisableLog("rdApp.error")
        try:
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    ValueError, "could not parse canonical PSMILES rows"
                ):
                    build_rdkit_frame(frame, "target")
        finally:
            RDLogger.EnableLog("rdApp.error")


class ConstructionIntegrationTests(unittest.TestCase):
    def test_vipea_processor_creates_vipea_output_directories(self) -> None:
        weighted = pd.DataFrame(
            {
                "poly_chemprop_input": [
                    "[*:1]CC[*:2].[*:3]O[*:4]|0.5|0.5|"
                    "<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5",
                    "random-weighted",
                    "block-weighted",
                ]
            }
        )
        information = pd.DataFrame(
            {
                "poly_type": ["alternating", "random", "block"],
                "EA (eV)": [1.0, 2.0, 3.0],
                "IP (eV)": [4.0, 5.0, 6.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp:
            datasets_root = Path(temp) / "Datasets"
            with patch(
                "process_Vipea_data.pd.read_csv", side_effect=[weighted, information]
            ):
                process_vipea_data(CONSTRUCTION_DIR, datasets_root)

            for architecture in ("alternating", "random", "block"):
                for quantity in ("EA", "IP"):
                    self.assertTrue(
                        (
                            datasets_root
                            / "wPSMILES"
                            / "Vipea"
                            / quantity
                            / f"{architecture}_{quantity}.csv"
                        ).is_file()
                    )
            for quantity in ("EA", "IP"):
                self.assertTrue(
                    (
                        datasets_root
                        / "PSMILES"
                        / "Vipea"
                        / quantity
                        / f"alternating_{quantity}.csv"
                    ).is_file()
                )
            self.assertFalse((datasets_root / "PSMILES" / "Coley_2022").exists())
            self.assertFalse((datasets_root / "wPSMILES" / "Coley_2022").exists())

    def test_converter_is_non_mutating_and_strict_missing_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            input_root = temp_path / "input"
            output_root = temp_path / "output"
            input_root.mkdir()
            source = input_root / "sample.csv"
            source.write_text("smiles,target\n*C*,1\n*C*,1\n*CC*,2\n", encoding="utf-8")
            original = source.read_bytes()

            result = run_cli(
                "PSMILES_to_wPSMILES.py",
                str(input_root),
                str(output_root),
                "--strict",
                cwd=temp_path,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source.read_bytes(), original)
            converted = pd.read_csv(output_root / "sample.csv")
            self.assertEqual(len(converted), 2)
            self.assertTrue(converted["smiles"].str.contains(r"\|", regex=True).all())

            missing = run_cli(
                "convert_web_datasets.py",
                "--psmiles-root",
                str(temp_path / "missing"),
                "--wpsmiles-root",
                str(output_root),
                "--datasets",
                "required_dataset",
                "--strict-missing",
                cwd=temp_path,
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("Missing required_dataset", missing.stderr)

    def test_tiny_pipeline_alignment_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            datasets_root = temp_path / "datasets"
            relative = Path("Fixture") / "target" / "fixture.csv"
            psmiles_path = datasets_root / "PSMILES" / relative
            psmiles_path.parent.mkdir(parents=True)

            rows = 20
            psmiles = [f"*{'C' * length}*" for length in range(1, rows + 1)]
            targets = np.arange(rows, dtype=float) / 10.0
            canonical = pd.DataFrame({"smiles": psmiles, "target": targets})
            canonical.to_csv(psmiles_path, index=False)
            canonical_bytes = psmiles_path.read_bytes()

            conversion = run_cli(
                "convert_web_datasets.py",
                "--psmiles-root",
                str(datasets_root / "PSMILES"),
                "--wpsmiles-root",
                str(datasets_root / "wPSMILES"),
                "--datasets",
                "Fixture",
                "--strict-missing",
                cwd=temp_path,
            )
            self.assertEqual(conversion.returncode, 0, conversion.stderr)
            self.assertEqual(psmiles_path.read_bytes(), canonical_bytes)

            embedding = np.add.outer(np.arange(rows, dtype=float), np.arange(600) / 1000)
            embedding_frame = pd.DataFrame(embedding, columns=[str(i) for i in range(600)])
            embedding_frame["target"] = targets
            embedding_path = datasets_root / "polyBERT" / relative
            embedding_path.parent.mkdir(parents=True)
            embedding_frame.to_csv(embedding_path, index=False)

            outputs = [temp_path / "splits_one", temp_path / "splits_two"]
            for output in outputs:
                result = run_cli(
                    "create_scaling_splits.py",
                    "--dataset",
                    "Fixture",
                    "--property",
                    "target",
                    "--file",
                    "fixture.csv",
                    "--train-sizes",
                    "4",
                    "8",
                    "--num-folds",
                    "2",
                    "--val-frac",
                    "0.2",
                    "--test-frac",
                    "0.2",
                    "--seed",
                    "7",
                    "--datasets-root",
                    str(datasets_root),
                    "--output-root",
                    str(output),
                    cwd=temp_path,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            self.assertEqual(tree_bytes(outputs[0]), tree_bytes(outputs[1]))

            base = outputs[0] / "Fixture" / "target" / "fixture" / "seed_7"
            manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_sizes"], [4, 8])
            self.assertEqual(
                manifest["source_paths"]["PSMILES"],
                "PSMILES/Fixture/target/fixture.csv",
            )
            self.assertNotIn(temp, json.dumps(manifest))

            for fold in range(2):
                fold_dir = base / f"fold_{fold:02d}"
                smaller = set(
                    pd.read_csv(fold_dir / "train_4" / "indices" / "train.csv")[
                        "row_index"
                    ]
                )
                larger = set(
                    pd.read_csv(fold_dir / "train_8" / "indices" / "train.csv")[
                        "row_index"
                    ]
                )
                self.assertTrue(smaller < larger)

                fixed_val = None
                fixed_test = None
                for size in (4, 8):
                    size_dir = fold_dir / f"train_{size}"
                    index_frames = {
                        split: pd.read_csv(size_dir / "indices" / f"{split}.csv")
                        for split in ("train", "val", "test")
                    }
                    index_sets = {
                        split: set(frame["row_index"])
                        for split, frame in index_frames.items()
                    }
                    self.assertTrue(index_sets["train"].isdisjoint(index_sets["val"]))
                    self.assertTrue(index_sets["train"].isdisjoint(index_sets["test"]))
                    self.assertTrue(index_sets["val"].isdisjoint(index_sets["test"]))

                    val_order = tuple(index_frames["val"]["row_index"])
                    test_order = tuple(index_frames["test"]["row_index"])
                    if fixed_val is None:
                        fixed_val, fixed_test = val_order, test_order
                    else:
                        self.assertEqual(val_order, fixed_val)
                        self.assertEqual(test_order, fixed_test)

                    for split in ("train", "val", "test"):
                        expected = index_frames[split]["target"].to_numpy()
                        for representation in (
                            "PSMILES",
                            "wPSMILES",
                            "polyBERT",
                            "RDKit_descriptors",
                        ):
                            frame = pd.read_csv(
                                size_dir / representation / f"{split}.csv"
                            )
                            self.assertEqual(len(frame), len(expected))
                            np.testing.assert_allclose(frame.iloc[:, -1], expected)

            packages = [temp_path / "indices_one", temp_path / "indices_two"]
            for package in packages:
                export = run_cli(
                    "manage_scaling_indices.py",
                    "export",
                    "--split-root",
                    str(outputs[0]),
                    "--datasets-root",
                    str(datasets_root),
                    "--output-root",
                    str(package),
                    cwd=temp_path,
                )
                self.assertEqual(export.returncode, 0, export.stderr)
            self.assertEqual(tree_bytes(packages[0]), tree_bytes(packages[1]))

            verification = run_cli(
                "manage_scaling_indices.py",
                "verify",
                "--indices-root",
                str(packages[0]),
                "--datasets-root",
                str(datasets_root),
                cwd=temp_path,
            )
            self.assertEqual(verification.returncode, 0, verification.stderr)

            published_output = temp_path / "splits_from_published_indices"
            rematerialized = run_cli(
                "create_scaling_splits.py",
                "--dataset",
                "Fixture",
                "--property",
                "target",
                "--file",
                "fixture.csv",
                "--train-sizes",
                "4",
                "8",
                "--num-folds",
                "2",
                "--val-frac",
                "0.2",
                "--test-frac",
                "0.2",
                "--seed",
                "7",
                "--datasets-root",
                str(datasets_root),
                "--indices-root",
                str(packages[0]),
                "--output-root",
                str(published_output),
                cwd=temp_path,
            )
            self.assertEqual(rematerialized.returncode, 0, rematerialized.stderr)

            published_base = (
                published_output / "Fixture" / "target" / "fixture" / "seed_7"
            )
            published_manifest = json.loads(
                (published_base / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(published_manifest["split_source"], "published_indices")
            original_indices = {
                str(path.relative_to(base)): path.read_bytes()
                for path in sorted(base.rglob("indices/*.csv"))
            }
            published_indices = {
                str(path.relative_to(published_base)): path.read_bytes()
                for path in sorted(published_base.rglob("indices/*.csv"))
            }
            self.assertEqual(original_indices, published_indices)

    def test_scaling_builder_rejects_misaligned_representations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            datasets_root = temp_path / "datasets"
            relative = Path("Fixture") / "target" / "fixture.csv"
            canonical = pd.DataFrame(
                {
                    "smiles": [f"*{'C' * length}*" for length in range(1, 11)],
                    "target": np.arange(10, dtype=float),
                }
            )
            weighted = canonical.copy()
            weighted["smiles"] = weighted["smiles"].map(PSMILES_to_wPSMILES)
            weighted.loc[3, "target"] = -999.0

            for representation, frame in (
                ("PSMILES", canonical),
                ("wPSMILES", weighted),
            ):
                path = datasets_root / representation / relative
                path.parent.mkdir(parents=True)
                frame.to_csv(path, index=False)

            result = run_cli(
                "create_scaling_splits.py",
                "--dataset",
                "Fixture",
                "--property",
                "target",
                "--file",
                "fixture.csv",
                "--train-sizes",
                "4",
                "--num-folds",
                "1",
                "--datasets-root",
                str(datasets_root),
                "--output-root",
                str(temp_path / "splits"),
                "--representations",
                "PSMILES",
                "wPSMILES",
                cwd=temp_path,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not row-aligned", result.stderr)

    def test_cli_help_works_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            cwd = Path(temp)
            for script in (
                "PSMILES_to_wPSMILES.py",
                "convert_web_datasets.py",
                "create_scaling_splits.py",
                "manage_scaling_indices.py",
            ):
                with self.subTest(script=script):
                    result = run_cli(script, "--help", cwd=cwd)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
