"""
wPSMILES to PSMILES conversion functions.

These functions are now centralized in polymer_conversions.py.
This file is kept for backwards compatibility and provides DataFrame-friendly wrappers.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# Import the centralized functions
from polymer_conversions import wPSMILES_to_PSMILES_alternating


def wPSMILES_to_PSMILES_homopolymer(dataset):
    """
    Takes a dataframe, returns a dataframe.
    Converts wPSMILES to PSMILES for homopolymer column.
    
    This is a DataFrame wrapper that maintains backwards compatibility.
    """
    dataset.poly_chemprop_input = dataset.poly_chemprop_input.str.split('|').str[0]
    dataset.poly_chemprop_input = dataset.poly_chemprop_input.str.replace('[*:1]','*')
    dataset.poly_chemprop_input = dataset.poly_chemprop_input.str.replace('[*:2]','*')
    return dataset


# wPSMILES_to_PSMILES_alternating is imported from polymer_conversions
# It's available for use directly
