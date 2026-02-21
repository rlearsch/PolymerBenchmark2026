#!/usr/bin/env python3
"""
Convert web-sourced PSMILES datasets to wPSMILES format.

This script processes small datasets that were obtained from the web/literature
and are already in PSMILES format. It uses the PSMILES_to_wPSMILES.py script
to generate corresponding wPSMILES versions.

Source datasets (in PSMILES/, tracked in git):
- Polymer_Genome
- OpenPoly_2025
- PolyMetriX
"""

import subprocess
import sys
from pathlib import Path

# Use paths relative to this script's location
script_dir = Path(__file__).parent

def convert_web_datasets():
    """
    Convert web-sourced datasets from PSMILES to wPSMILES using PSMILES_to_wPSMILES.py
    """
    psmiles_root = script_dir / '../../Datasets/PSMILES'
    wpsmiles_root = script_dir / '../../Datasets/wPSMILES'
    
    # Directories to process
    datasets_to_convert = [
        'Polymer_Genome',
        'OpenPoly_2025',
        'PolyMetriX',
    ]
    
    print("=" * 60)
    print("Converting Web-Sourced Datasets to wPSMILES")
    print("=" * 60)
    print()
    
    success = True
    
    for dataset_name in datasets_to_convert:
        input_dir = psmiles_root / dataset_name
        output_dir = wpsmiles_root / dataset_name
        
        # Check if input exists
        if not input_dir.exists():
            print(f"⚠ Skipping {dataset_name} (not found at {input_dir})")
            continue
        
        print(f"Converting: {dataset_name}")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print()
        
        # Run PSMILES_to_wPSMILES.py script
        try:
            result = subprocess.run(
                [sys.executable, 'PSMILES_to_wPSMILES.py', str(input_dir), str(output_dir)],
                cwd=script_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"✗ Error converting {dataset_name}:")
            print(e.stdout)
            print(e.stderr)
            success = False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            success = False
    
    print()
    print("=" * 60)
    if success:
        print("✓ All conversions complete!")
    else:
        print("✗ Some conversions failed. Check output above.")
    print("=" * 60)
    print()
    
    return success


if __name__ == '__main__':
    success = convert_web_datasets()
    sys.exit(0 if success else 1)
