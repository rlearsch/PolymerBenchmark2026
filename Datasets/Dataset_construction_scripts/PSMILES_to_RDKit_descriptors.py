#!/usr/bin/env python3
"""
PSMILES_to_RDKit_descriptors.py

Convert PSMILES datasets to RDKit molecular descriptors for traditional ML models.

This script recursively processes CSV files containing PSMILES, calculates all RDKit
descriptors, and saves them in the same directory structure under RDKit_descriptors/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Descriptors import CalcMolDescriptors
import sys


# Use paths relative to this script's location
script_dir = Path(__file__).parent


def calc_all_descriptors(psmi: str, cap='[H]', missing=float('nan')):
    """
    Cap PSMILES, build RDKit Mol, and compute all RDKit descriptors.
    
    Args:
        psmi: PSMILES string
        cap: Capping group for attachment points (default '[H]')
        missing: Value to use for missing descriptors
    
    Returns:
        Dictionary of descriptor name -> value
    """
    m = Chem.MolFromSmiles(psmi)
    mol = m 
    if mol is None:
        return {}
    return CalcMolDescriptors(mol, missingVal=np.nan, silent=True)


def sanitize(X):
    """
    Clean descriptor dataframe by removing problematic columns and handling inf/nan.
    
    Args:
        X: DataFrame of descriptors
    
    Returns:
        Cleaned DataFrame
    """
    # Drop columns known to cause issues
    columns_to_drop = ['MaxPartialCharge', 'MinPartialCharge', 
                       'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'Ipc']
    try:
        X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])
    except Exception as e:
        print(f'Warning: Failed to drop some columns in sanitize: {e}')
    
    # Convert to numeric and handle inf values
    X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def is_csv_like(path: Path) -> bool:
    """Check if file is a CSV."""
    return path.name.lower().endswith(".csv")


def process_file(input_path: Path, output_path: Path):
    """
    Process a single PSMILES CSV file and generate RDKit descriptors.
    
    Args:
        input_path: Path to input PSMILES CSV
        output_path: Path to output RDKit descriptors CSV
    """
    try:
        print(f"  Processing: {input_path.name}")
        
        # Read input CSV
        df = pd.read_csv(input_path)
        
        # Check if 'smiles' column exists
        if 'smiles' not in df.columns:
            print(f"    Warning: No 'smiles' column found, skipping")
            return False
        
        # Calculate descriptors for each SMILES
        print(f"    Calculating descriptors for {len(df)} polymers...")
        X = df["smiles"].apply(calc_all_descriptors).apply(pd.Series)
        X = sanitize(X)
        
        # Get target column(s) - everything except 'smiles'
        target_cols = [col for col in df.columns if col != 'smiles']
        y = df[target_cols]
        
        # Combine descriptors with targets
        result = pd.concat([X, y], axis=1)
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        result.to_csv(output_path, index=False)
        print(f"    ✓ Saved {len(X.columns)} descriptors to {output_path.name}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error processing {input_path.name}: {e}")
        return False


def generate_rdkit_descriptors():
    """
    Main function to recursively process all PSMILES datasets and generate RDKit descriptors.
    """
    psmiles_root = script_dir / '../../Datasets/PSMILES'
    rdkit_root = script_dir / '../../Datasets/RDKit_descriptors'
    
    print("=" * 70)
    print("Generating RDKit Molecular Descriptors from PSMILES Datasets")
    print("=" * 70)
    print()
    
    # Check if PSMILES directory exists
    if not psmiles_root.exists():
        print(f"ERROR: PSMILES directory not found at {psmiles_root}")
        print("Please run generate_basic_datasets.sh first.")
        return False
    
    # Find all CSV files in PSMILES directory
    csv_files = [p for p in psmiles_root.rglob("*.csv") if p.is_file()]
    
    if not csv_files:
        print("No CSV files found in PSMILES directory.")
        return False
    
    print(f"Found {len(csv_files)} CSV files to process")
    print()
    
    success_count = 0
    fail_count = 0
    
    for input_path in csv_files:
        # Calculate relative path from PSMILES root
        rel_path = input_path.relative_to(psmiles_root)
        output_path = rdkit_root / rel_path
        
        # Check if output already exists
        if output_path.exists():
            print(f"  Skipping {rel_path} (already exists)")
            success_count += 1
            continue
        
        # Process the file
        if process_file(input_path, output_path):
            success_count += 1
        else:
            fail_count += 1
        print()
    
    print("=" * 70)
    print("RDKit Descriptor Generation Complete")
    print("=" * 70)
    print(f"Successfully processed: {success_count}/{len(csv_files)} files")
    if fail_count > 0:
        print(f"Failed: {fail_count} files")
    print()
    print(f"Output directory: {rdkit_root}")
    print()
    
    return fail_count == 0


if __name__ == '__main__':
    success = generate_rdkit_descriptors()
    sys.exit(0 if success else 1)
