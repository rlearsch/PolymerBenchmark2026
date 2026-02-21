#from psmiles import PolymerSmiles as PS
import pickle
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer

# Use paths relative to this script's location
script_dir = Path(__file__).parent

dictionary_path = script_dir / 'files/PSMILES_pBERT_dict.pkl'

# Load existing dictionary or create new empty one
if dictionary_path.exists():
    with open(dictionary_path,'rb') as dictionary:
        PSMILES_pBERT_dict = pickle.load(dictionary)
    keys_length = len(PSMILES_pBERT_dict.keys())
    print(f'Initial dictionary loaded with {keys_length} entries')
else:
    PSMILES_pBERT_dict = {}
    print('No existing dictionary found. Creating new dictionary from scratch.')

in_root = script_dir / '../../Datasets/PSMILES/'

# NOTE: Requires local polyBERT model at ../../Models/polyBERT/
# Download from: https://huggingface.co/kuelumbus/polyBERT
polyBERT = SentenceTransformer(script_dir / '../../Models/polyBERT')

def PSMILES_to_FP(smiles):
    #ps = PS(smiles)
    #time.sleep(2)
    #fingerprint = ps.fingerprint('polyBERT')
    fingerprint = polyBERT.encode(smiles)
    return fingerprint

def save_dictionary(dictionary):
    file_path = script_dir / 'files/PSMILES_pBERT_dict.pkl'
    with open(file_path, "wb") as file_handler:
        pickle.dump(dictionary, file_handler)

def is_csv_like(path: Path) -> bool:
    """
    Treat files that end with '.csv' or compressed variants as CSV-like.
    (pandas can infer compression from the extension.)
    """
    name = path.name.lower()
    return (
        name.endswith(".csv")
    )

dataset_list = [p for p in in_root.rglob("*") if p.is_file() and is_csv_like(p)]

count = 0
for dataset in dataset_list:
    df = pd.read_csv(dataset)
    #print(df.columns)
    try:
        smiles_list = df.smiles.tolist()
        for smiles in smiles_list:
            if smiles not in PSMILES_pBERT_dict:
                count = count+1
                try:
                    PSMILES_pBERT_dict[smiles] = PSMILES_to_FP(smiles)
                except Exception as e:
                    print(f'Error processing {smiles}: {e}')
                if count%100==0:
                    save_dictionary(PSMILES_pBERT_dict)
                    keys_length = len(PSMILES_pBERT_dict.keys())
                    print(f'Dictionary saved with {keys_length} entries')
            else: 
                pass 

    except Exception as e:
        print(f'Error processing {dataset}: {e}')
    
    save_dictionary(PSMILES_pBERT_dict)
    keys_length = len(PSMILES_pBERT_dict.keys())
    print(f'Dictionary saved with {keys_length} entries')
    