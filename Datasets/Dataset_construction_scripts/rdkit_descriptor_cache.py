"""Persistent, provenance-validated cache for RDKit molecular descriptors.

The cache key is the exact PSMILES input string. This deliberately avoids
changing the chemical identity convention used by the canonical PSMILES
datasets while eliminating repeated descriptor calculations across property
files and scaling-split materializations.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Descriptors import CalcMolDescriptors


CACHE_SCHEMA_VERSION = 1
DESCRIPTOR_ALGORITHM_VERSION = 1
DROP_COLUMNS = (
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "Ipc",
)


def calculate_descriptors(psmiles: str) -> dict[str, float]:
    """Calculate the un-sanitized descriptor mapping for one PSMILES string."""
    molecule = Chem.MolFromSmiles(psmiles)
    if molecule is None:
        return {}
    return CalcMolDescriptors(molecule, missingVal=np.nan, silent=True)


def descriptor_columns() -> list[str]:
    """Return stable output columns after the repository sanitization policy."""
    values = calculate_descriptors("C")
    return [name for name in values if name not in DROP_COLUMNS]


def sanitize_descriptors(
    features: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Apply the existing descriptor cleaning policy with a fixed column order."""
    features = features.reindex(columns=columns or descriptor_columns())
    features = features.apply(pd.to_numeric, errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan)


def default_cache_path(datasets_root: Path) -> Path:
    """Return the ignored shared cache location for a datasets root."""
    return datasets_root / "RDKit_descriptors" / ".descriptor_cache.sqlite3"


def _json_value(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


class DescriptorCache:
    """SQLite cache whose contents are valid only for matching provenance."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.columns = descriptor_columns()
        self._initialize_or_validate()

    def __enter__(self) -> "DescriptorCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def metadata(self) -> dict[str, object]:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "descriptor_algorithm_version": DESCRIPTOR_ALGORITHM_VERSION,
            "rdkit_version": rdBase.rdkitVersion,
            "descriptor_columns": self.columns,
            "dropped_columns": list(DROP_COLUMNS),
        }

    def _initialize_or_validate(self) -> None:
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS descriptors ("
            "psmiles TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'provenance'"
        ).fetchone()
        expected = self.metadata()
        if row is None:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('provenance', ?)",
                (json.dumps(expected, sort_keys=True),),
            )
            self.connection.commit()
            return

        actual = json.loads(row[0])
        if actual != expected:
            raise ValueError(
                f"Descriptor cache provenance does not match this environment: {self.path}. "
                "Remove this generated cache and rebuild it with the current descriptor code."
            )

    def descriptor_rows(self, psmiles: Iterable[str]) -> list[dict[str, float]]:
        """Return descriptor mappings in input order, calculating cache misses once."""
        values = list(psmiles)
        unique = list(dict.fromkeys(values))
        cached: dict[str, dict[str, float]] = {}
        for start in range(0, len(unique), 900):
            batch = unique[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT psmiles, payload FROM descriptors WHERE psmiles IN ({placeholders})",
                batch,
            ).fetchall()
            cached.update({key: json.loads(payload) for key, payload in rows})

        missing = [value for value in unique if value not in cached]
        calculated = {value: calculate_descriptors(value) for value in missing}
        records = []
        for value, descriptors in calculated.items():
            # Invalid PSMILES are intentionally not cached, so a later corrected
            # RDKit installation never inherits a prior parse failure.
            if descriptors:
                payload = {key: _json_value(number) for key, number in descriptors.items()}
                records.append((value, json.dumps(payload, sort_keys=True)))
                cached[value] = payload
            else:
                cached[value] = {}
        if records:
            self.connection.executemany(
                "INSERT OR IGNORE INTO descriptors(psmiles, payload) VALUES (?, ?)", records
            )
            self.connection.commit()

        return [cached[value] for value in values]

    def descriptor_frame(self, psmiles: Iterable[str]) -> tuple[pd.DataFrame, list[int]]:
        """Return sanitized features and positions that RDKit could not parse."""
        rows = self.descriptor_rows(psmiles)
        invalid = [index for index, values in enumerate(rows) if not values]
        return sanitize_descriptors(pd.DataFrame(rows), self.columns), invalid
