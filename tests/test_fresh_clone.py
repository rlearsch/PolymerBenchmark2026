from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "Datasets" / "dataset_manifest.json"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


class FreshCloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_required_sources_are_tracked_and_present_in_git_archive(self) -> None:
        tracked = set(run("git", "ls-files").stdout.splitlines())
        archive = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
            archived = set(handle.getnames())

        for source in self.manifest["required_sources"]:
            relative = source["path"]
            with self.subTest(path=relative):
                self.assertIn(relative, tracked)
                self.assertIn(relative, archived)
                self.assertGreater((REPO_ROOT / relative).stat().st_size, 0)

    def test_required_source_schemas(self) -> None:
        for source in self.manifest["required_sources"]:
            path = REPO_ROOT / source["path"]
            with self.subTest(path=source["path"]):
                if source["format"] == "csv":
                    with path.open(newline="", encoding="utf-8") as handle:
                        header = next(csv.reader(handle))
                    self.assertTrue(set(source["required_columns"]) <= set(header))
                    continue

                with path.open(encoding="utf-8") as handle:
                    first = json.loads(next(handle))
                    self.assertTrue(
                        set(source["required_input_keys"]) <= set(first["inputs"])
                    )
                    self.assertTrue(
                        set(source["required_output_keys"]) <= set(first["outputs"])
                    )
                    found = set(first["outputs"])
                    wanted = set(source.get("required_output_keys_anywhere", []))
                    for line in handle:
                        if wanted <= found:
                            break
                        found.update(json.loads(line)["outputs"])
                    self.assertTrue(wanted <= found)

    def test_generated_artifacts_are_ignored(self) -> None:
        for relative in self.manifest["generated_paths_that_must_be_ignored"]:
            with self.subTest(path=relative):
                result = run("git", "check-ignore", "--quiet", relative, check=False)
                self.assertEqual(result.returncode, 0)

    def test_task3_complexity_metadata_is_complete_and_consistent(self) -> None:
        path = REPO_ROOT / "Datasets" / "task3_complexity" / "alternating_refractive_index.csv"
        frame = pd.read_csv(path)

        self.assertFalse(frame.isna().any().any())
        self.assertTrue(frame["original_index"].is_unique)
        self.assertTrue(frame["psmiles"].is_unique)
        self.assertTrue(frame["legacy_test_fold"].isin(range(1, 6)).all())

        fragments = frame["wpsmiles"].str.split("|").str[0].str.count(r"\.") + 1
        pd.testing.assert_series_equal(
            fragments.reset_index(drop=True),
            frame["n_distinct_monomers"].reset_index(drop=True),
            check_names=False,
        )

        task3 = frame[frame["n_distinct_monomers"].between(5, 10)]
        self.assertEqual(len(task3), 2445)
        self.assertTrue(task3["n_distinct_monomers"].isin(range(5, 11)).all())

    def test_task4_legacy_training_source_is_complete(self) -> None:
        path = REPO_ROOT / "Datasets" / "task4_architecture_transfer" / "legacy_train.csv"
        frame = pd.read_csv(path)
        self.assertEqual(list(frame.columns), ["SMILES", "EA (eV)"])
        self.assertEqual(len(frame), 5855)
        self.assertFalse(frame.isna().any().any())
        self.assertTrue(frame["SMILES"].is_unique)

    def test_tracked_construction_code_has_no_machine_specific_paths(self) -> None:
        tracked = run(
            "git",
            "ls-files",
            "Datasets/Dataset_construction_scripts/*.py",
            "Datasets/Dataset_construction_scripts/*.sh",
        ).stdout.splitlines()
        absolute_home = re.compile(r"/Users/|/home/|[A-Za-z]:\\Users\\")
        retired_name = re.compile(r"polymer[_ -]?genome", re.IGNORECASE)
        for relative in tracked:
            with self.subTest(path=relative):
                text = (REPO_ROOT / relative).read_text(encoding="utf-8")
                self.assertIsNone(absolute_home.search(text))
                self.assertIsNone(retired_name.search(text))

    def test_scaling_driver_matches_manifest(self) -> None:
        result = run(
            "bash",
            "Datasets/Dataset_construction_scripts/generate_scaling_datasets.sh",
            "--dry-run",
        )
        actual = []
        for line in result.stdout.splitlines():
            if not line.startswith("DRY RUN:"):
                continue
            args = shlex.split(line.removeprefix("DRY RUN:").strip())

            def value(option: str) -> str:
                return args[args.index(option) + 1]

            size_start = args.index("--train-sizes") + 1
            size_end = next(
                index
                for index in range(size_start, len(args))
                if args[index].startswith("--")
            )
            actual.append(
                {
                    "dataset": value("--dataset"),
                    "property": value("--property"),
                    "file": value("--file"),
                    "train_sizes": [int(item) for item in args[size_start:size_end]],
                }
            )

        self.assertEqual(actual, self.manifest["scaling_experiments"])

    def test_published_scaling_packages_match_manifest_and_are_trackable(self) -> None:
        settings = self.manifest["published_scaling_indices"]
        root = REPO_ROOT / settings["root"]
        manifests = sorted(root.rglob("manifest.json"))
        expected = {
            (item["dataset"], item["property"], item["file"], tuple(item["train_sizes"]))
            for item in self.manifest["scaling_experiments"]
        }
        actual = set()
        tracked = set(run("git", "ls-files", settings["root"]).stdout.splitlines())

        for manifest_path in manifests:
            package = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual.add(
                (
                    package["dataset"],
                    package["property"],
                    package["file"],
                    tuple(package["train_sizes"]),
                )
            )
            self.assertEqual(package["schema_version"], settings["schema_version"])
            self.assertEqual(package["seed"], settings["seed"])
            self.assertEqual(package["num_folds"], settings["num_folds"])
            for path in [manifest_path, *sorted(manifest_path.parent.glob("fold_*.npz"))]:
                relative = str(path.relative_to(REPO_ROOT))
                with self.subTest(path=relative):
                    self.assertIn(relative, tracked)
                    ignored = run("git", "check-ignore", "--quiet", relative, check=False)
                    self.assertNotEqual(ignored.returncode, 0)

        self.assertEqual(actual, expected)

    def test_real_condition_materializes_published_indices(self) -> None:
        experiment = self.manifest["scaling_experiments"][0]
        settings = self.manifest["published_scaling_indices"]
        package_base = (
            REPO_ROOT
            / settings["root"]
            / experiment["dataset"]
            / experiment["property"]
            / Path(experiment["file"]).stem
            / f"seed_{settings['seed']}"
        )
        package = json.loads((package_base / "manifest.json").read_text(encoding="utf-8"))
        canonical_path = REPO_ROOT / "Datasets" / package["canonical_source"]
        if not canonical_path.is_file():
            self.skipTest(
                "requires generated canonical PSMILES; fresh-clone CI validates "
                "the published index package without generated datasets"
            )
        digest = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
        self.assertEqual(digest, package["canonical_sha256"])

        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "scaling_splits"
            result = run(
                sys.executable,
                "Datasets/Dataset_construction_scripts/create_scaling_splits.py",
                "--dataset",
                experiment["dataset"],
                "--property",
                experiment["property"],
                "--file",
                experiment["file"],
                "--train-sizes",
                *(str(size) for size in experiment["train_sizes"]),
                "--num-folds",
                str(settings["num_folds"]),
                "--seed",
                str(settings["seed"]),
                "--representations",
                "PSMILES",
                "--output-root",
                str(output_root),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output_base = (
                output_root
                / experiment["dataset"]
                / experiment["property"]
                / Path(experiment["file"]).stem
                / f"seed_{settings['seed']}"
            )
            materialized_manifest = json.loads(
                (output_base / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(materialized_manifest["split_source"], "published_indices")

            for expected_fold, fold_record in enumerate(package["folds"]):
                self.assertEqual(fold_record["fold"], expected_fold)
                with np.load(package_base / fold_record["indices_file"], allow_pickle=False) as archive:
                    expected_arrays = {
                        name: archive[name].copy() for name in ("train_pool", "val", "test")
                    }
                fold_dir = output_base / f"fold_{expected_fold:02d}"
                for split, expected_indices in expected_arrays.items():
                    actual_indices = pd.read_csv(
                        fold_dir / "indices" / f"{split}.csv",
                        usecols=["row_index"],
                    )["row_index"].to_numpy()
                    np.testing.assert_array_equal(actual_indices, expected_indices)
                for train_size in experiment["train_sizes"]:
                    actual_train = pd.read_csv(
                        fold_dir / f"train_{train_size}" / "indices" / "train.csv",
                        usecols=["row_index"],
                    )["row_index"].to_numpy()
                    np.testing.assert_array_equal(
                        actual_train, expected_arrays["train_pool"][:train_size]
                    )


if __name__ == "__main__":
    unittest.main()
