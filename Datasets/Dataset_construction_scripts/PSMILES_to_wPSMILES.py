#!/usr/bin/env python3
"""
PSMILES_to_wPSMILES.py

Recursively process CSV files with pandas, preserving folder structure.

Usage:
  python PSMILES_to_wPSMILES.py /path/to/input /path/to/output

Optional flags:
  --dry-run           Show what would be processed without writing files
  --strict            Fail on first error (default is to continue)
  --encoding ENC      File encoding (default: utf-8; try latin-1 if needed)
  --sep SEP           CSV delimiter (default: auto-detect, else ',')
  --pattern PAT       Limit to files whose name contains this substring
"""
from __future__ import annotations

import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Optional

# Import centralized conversion function
from polymer_conversions import PSMILES_to_wPSMILES


def is_csv_like(path: Path) -> bool:
    """
    Treat files that end with '.csv' or compressed variants as CSV-like.
    (pandas can infer compression from the extension.)
    """
    name = path.name.lower()
    return (
        name.endswith(".csv")
    )

def process_file(
    in_path: Path,
    in_root: Path,
    out_root: Path,
    *,
    encoding: str,
    sep: Optional[str],
    dry_run: bool,
    ) -> None:
    rel = in_path.relative_to(in_root)
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to auto-detect sep if not provided
    _sep = sep
    #if _sep is None:
    #    try:
    #        with in_path.open("r", encoding=encoding, errors="replace") as fh:
    #            head = fh.read(20000)
    #        #_sep = detect_sep(head) or ","
    #        _sep = ','
    #    except Exception:
    #        _sep = ","

    _sep = ","
    #_sep="\t"
    print(f"Processing: {in_path}  ->  {out_path} (sep='{_sep}')")

    if dry_run:
        return

    # Read
    df = pd.read_csv(in_path, encoding=encoding, sep=_sep, low_memory=False)
    if "smiles" not in df.columns:
        raise ValueError(f"Missing required 'smiles' column: {in_path}")
    # Source datasets are immutable inputs. Deduplicate only the generated copy.
    df_in = df.drop_duplicates("smiles").copy()
    # Transform
    df_out = df_in
    df_out["smiles"] = df_out["smiles"].apply(PSMILES_to_wPSMILES)
    #df_out['polymer_psmiles_guess'] = df_out.polymer_psmiles_guess.apply(PSMILES_to_wPSMILES)

    # Write (compression inferred from extension)
    df_out.to_csv(out_path, index=False, encoding=encoding)

def main() -> int:
    ap = argparse.ArgumentParser(description="Recursively transform CSVs with pandas.")
    ap.add_argument("input_dir", type=Path, help="Directory to crawl for CSVs")
    ap.add_argument("output_dir", type=Path, help="Directory to write transformed CSVs")
    ap.add_argument("--dry-run", action="store_true", help="List actions only")
    ap.add_argument("--strict", action="store_true", help="Stop on first error")
    ap.add_argument("--encoding", default="utf-8", help="Text encoding (default: utf-8)")
    ap.add_argument("--sep", default=None, help="CSV delimiter (default: auto-detect)")
    ap.add_argument("--pattern", default=None, help="Substring filter on filenames")
    args = ap.parse_args()
    in_root = args.input_dir.resolve()
    out_root = args.output_dir.resolve()

    if not in_root.exists() or not in_root.is_dir():
        print(f"ERROR: Input directory not found: {in_root}", file=sys.stderr)
        return 1

    # Walk and filter CSV-like files
    files = [p for p in in_root.rglob("*") if p.is_file() and is_csv_like(p)]
    if args.pattern:
        files = [p for p in files if args.pattern.lower() in p.name.lower()]

    if not files:
        print("No matching CSV files found.")
        return 0

    errors = 0
    for f in files:
        try:
            process_file(
                f,
                in_root,
                out_root,
                encoding=args.encoding,
                sep=args.sep,
                dry_run=args.dry_run,
            )
        except Exception as e:
            errors += 1
            print(f"[ERROR] {f}: {e}", file=sys.stderr)
            if args.strict:
                raise

    if errors:
        print(f"Completed with {errors} error(s).")
        return 1
    else:
        print("All done.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
