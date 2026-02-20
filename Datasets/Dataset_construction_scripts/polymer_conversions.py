"""
Polymer notation conversion utilities.

This module provides functions to convert between different polymer notation formats:
- PSMILES (Polymer SMILES)
- wPSMILES (Weighted Polymer SMILES with connectivity information)
- polyBERT embeddings

All conversion functions are centralized here to avoid duplication across scripts.
"""

import re
import numpy as np
from rdkit import Chem
from typing import Optional


# ============================================================================
# PSMILES to wPSMILES Conversions
# ============================================================================

def replace_astrix(input_string: str, unit_number: int = 0) -> str:
    """
    Replace asterisks with numbered attachment points for wPSMILES format.
    
    Args:
        input_string: PSMILES string with asterisks
        unit_number: Unit number for multi-unit polymers (default 0)
    
    Returns:
        String with [*:n] attachment points
    """
    parts = input_string.split('[*]', 2)
    
    if len(parts) < 3:
        parts = input_string.split('*', 2)
        if len(parts) < 3:
            print(f'Unable to convert {input_string} to wPSMILES.')
            return input_string
    
    connection_numbers = [(unit_number * 2) + 1, (unit_number * 2) + 2]
    return f"{parts[0]}[*:{connection_numbers[0]}]{parts[1]}[*:{connection_numbers[1]}]{parts[2]}"


def PSMILES_to_wPSMILES(input_string: str) -> str:
    """
    Convert PSMILES to wPSMILES format for homopolymers.
    
    Replaces asterisks with numbered attachment points and adds connectivity suffix.
    
    Args:
        input_string: PSMILES string
    
    Returns:
        wPSMILES string with attachment points and connectivity
    """
    replaced_smi = replace_astrix(input_string)
    return f"{replaced_smi}|1|<1-2:1:1<1-1:0:0<2-2:0:0"


def construct_alternating_wPSMILES(copolymer_ratio: str, smiles_list: str) -> str:
    """
    Construct alternating copolymer wPSMILES from ratio and SMILES list.
    
    Args:
        copolymer_ratio: Comma-separated ratio string (e.g., "0.5,0.5")
        smiles_list: Comma-separated SMILES strings
    
    Returns:
        wPSMILES string for alternating copolymer
    """
    ratios = [float(ratio) for ratio in copolymer_ratio.split(',')]
    smiles = smiles_list.split(',')
    wPSMILES_units = [replace_astrix(smi, unit_number=index) for index, smi in enumerate(smiles)]
    
    alternating_suffix = ""
    if len(ratios) == 2:
        alternating_suffix = '<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5'
    else:
        connections = [(i + 1) for i in range(2 * len(ratios))]
        length = len(connections)
        suffix_list = []
        
        for connection_index in range(len(connections)):
            if connections[connection_index] % 2 != 0:  # odd
                connection_list = [
                    connections[(connection_index - 2) % length],
                    connections[(connection_index - 1) % length],
                    connections[(connection_index + 2) % length],
                    connections[(connection_index + 3) % length]
                ]
                for c in connection_list:
                    suffix_list.append(f"<{connections[connection_index]}-{c}:.25:.25")
            
            if connections[connection_index] % 2 == 0:  # even
                connection_list = [
                    connections[(connection_index - 3) % length],
                    connections[(connection_index - 2) % length],
                    connections[(connection_index + 1) % length],
                    connections[(connection_index + 2) % length]
                ]
                for c in connection_list:
                    suffix_list.append(f"<{connections[connection_index]}-{c}:.25:.25")
        
        alternating_suffix = ''.join(suffix_list)
    
    return f"{'.'.join(wPSMILES_units)}|{'|'.join(map(str, ratios))}|{alternating_suffix}"


def construct_random_wPSMILES(copolymer_ratio: str, smiles_list: str) -> str:
    """
    Construct random copolymer wPSMILES from ratio and SMILES list.
    
    Args:
        copolymer_ratio: Comma-separated ratio string
        smiles_list: Comma-separated SMILES strings
    
    Returns:
        wPSMILES string for random copolymer
    """
    ratios = [float(ratio) for ratio in copolymer_ratio.split(',')]
    smiles = smiles_list.split(',')
    wPSMILES_units = [replace_astrix(smi, unit_number=index) for index, smi in enumerate(smiles)]
    
    connections = [(i + 1) for i in range(2 * len(ratios))]
    probability = 1 / len(connections)
    suffix_list = []
    
    for connection in connections:
        for connection_ii in range(connection, connections[-1] + 1):
            suffix_list.append(f'<{connection}-{connection_ii}:{probability}:{probability}')
    
    random_suffix = ''.join(suffix_list)
    return f"{'.'.join(wPSMILES_units)}|{'|'.join(map(str, ratios))}|{random_suffix}"


# ============================================================================
# wPSMILES to PSMILES Conversions
# ============================================================================

def wPSMILES_to_PSMILES_homopolymer_simple(wpsmiles: str) -> str:
    """
    Convert wPSMILES to PSMILES for homopolymers (simple string replacement).
    
    Args:
        wpsmiles: wPSMILES string
    
    Returns:
        PSMILES string
    """
    # Toss MW data and connectivity
    molecule = wpsmiles.split('|')[0]
    # Replace numbered attachment points with simple asterisks
    molecule = molecule.replace('[*:1]', '*').replace('[*:2]', '*')
    return molecule


def join_by_pairs(mols_or_smiles, pairs, bond_orders=None):
    """
    Join fragments by connecting neighbors of dummy atoms [*:<id>] for specified pairs.
    
    This is the primary method for converting wPSMILES to PSMILES for alternating copolymers.
    Uses RDKit's molecular graph operations for robust handling of complex polymers.
    
    Args:
        mols_or_smiles: Mol objects or SMILES strings
        pairs: Map-number pairs to connect, e.g. [(2,3), (4,5)]
        bond_orders: Bond order for connections (default SINGLE)
    
    Returns:
        Combined and sanitized Chem.Mol object
    """
    # Normalize inputs -> list of mols
    if isinstance(mols_or_smiles, (str, Chem.Mol)):
        mols_or_smiles = [mols_or_smiles]
    
    mols = []
    for x in mols_or_smiles:
        if isinstance(x, str):
            m = Chem.MolFromSmiles(x)
            if m is None:
                raise ValueError(f"Bad SMILES: {x}")
            mols.extend(Chem.GetMolFrags(m, asMols=True))
        else:
            mols.append(x)
    
    # Combine into one to get stable global indices
    combo = mols[0]
    for m in mols[1:]:
        combo = Chem.CombineMols(combo, m)
    
    # Collect sites: mapnum -> (dummy_idx, neighbor_idx)
    sites = {}
    for a in combo.GetAtoms():
        if a.GetAtomicNum() == 0 and a.HasProp("molAtomMapNumber"):
            mapnum = int(a.GetProp("molAtomMapNumber"))
            nbrs = [n.GetIdx() for n in a.GetNeighbors()]
            if len(nbrs) != 1:
                raise ValueError(f"[*:{mapnum}] must have exactly one neighbor")
            if mapnum in sites:
                raise ValueError(f"Map number {mapnum} appears more than once")
            sites[mapnum] = (a.GetIdx(), nbrs[0])
    
    # Helper to get bond order for a given pair
    def get_bond_order(p):
        if bond_orders is None:
            return Chem.BondType.SINGLE
        if isinstance(bond_orders, Chem.BondType):
            return bond_orders
        return bond_orders.get(p, bond_orders.get((p[1], p[0]), Chem.BondType.SINGLE))
    
    rw = Chem.RWMol(combo)
    to_delete = []
    
    for a_id, b_id in pairs:
        if a_id not in sites or b_id not in sites:
            raise ValueError(f"Missing map numbers in structure: {a_id}, {b_id}")
        dA, nA = sites[a_id]
        dB, nB = sites[b_id]
        rw.AddBond(nA, nB, get_bond_order((a_id, b_id)))
        to_delete.extend([dA, dB])
    
    # Remove dummies (descending order so indices stay valid)
    for idx in sorted(set(to_delete), reverse=True):
        rw.RemoveAtom(idx)
    
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def wPSMILES_to_PSMILES_alternating(input_smiles: str) -> str:
    """
    Convert wPSMILES to PSMILES for alternating copolymers using RDKit joining.
    
    This is the primary method, using RDKit's molecular graph operations for
    robust handling of complex polymers including those with challenging ring systems.
    
    Args:
        input_smiles: wPSMILES string
    
    Returns:
        PSMILES string
    
    Raises:
        ValueError: If input has fewer than 2 repeat units or invalid SMILES
    """
    # Toss MW data
    input_smiles = input_smiles.split('|')[0]
    
    # Separate repeat units
    repeat_units = input_smiles.split('.')
    if len(repeat_units) < 2:
        raise ValueError("Input SMILES must contain at least two repeat units separated by '.'")
    
    # Create pairs: [(2,3), (4,5), (6,7), ...]
    pairs = [(i * 2, (i * 2) + 1) for i in range(1, len(repeat_units))]
    
    # Join using RDKit
    mol = join_by_pairs(input_smiles, pairs=pairs)
    PSMILES = Chem.MolToSmiles(mol)
    
    # Replace numbered attachment points with simple asterisks
    pattern = re.compile(r"\[\*:\d+\]")
    PSMILES = pattern.sub("*", PSMILES)
    
    return PSMILES


# ============================================================================
# Utility Functions
# ============================================================================

def is_homopolymer(wpsmiles: str) -> bool:
    """
    Check if a wPSMILES string represents a homopolymer.
    
    Args:
        wpsmiles: wPSMILES string
    
    Returns:
        True if homopolymer, False if copolymer
    """
    parts = wpsmiles.split('|')[0]
    return '.' not in parts


def get_architecture(wpsmiles: str) -> str:
    """
    Determine the copolymer architecture from wPSMILES.
    
    Args:
        wpsmiles: wPSMILES string
    
    Returns:
        'homopolymer', 'alternating', 'random', or 'block'
    """
    if is_homopolymer(wpsmiles):
        return 'homopolymer'
    
    # Parse connectivity patterns to determine architecture
    # This is a simplified heuristic - actual implementation may be more complex
    connectivity = wpsmiles.split('|')[-1] if '|' in wpsmiles else ''
    
    if '<1-3:0.5:0.5' in connectivity:
        return 'alternating'
    elif '<1-1:' in connectivity and '<2-2:' in connectivity:
        return 'random'
    else:
        return 'block'
