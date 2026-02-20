import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Use paths relative to this script's location
script_dir = Path(__file__).parent

def is_csv_like(path: Path) -> bool:
    """
    Treat files that end with '.csv' or compressed variants as CSV-like.
    (pandas can infer compression from the extension.)
    """
    name = path.name.lower()
    return (
        name.endswith(".csv")
    )

dictionary_path = script_dir / 'files/PSMILES_pBERT_dict.pkl'
with open(dictionary_path,'rb') as dictionary:
    PSMILES_pBERT_dict = pickle.load(dictionary)

in_root = script_dir / '../../PSMILES/'
out_root = script_dir / '../../polyBERT/'
dataset_list = [p for p in in_root.rglob("*") if p.is_file() and is_csv_like(p)]
for dataset in dataset_list:
    data = pd.read_csv(dataset)
    rel = dataset.relative_to(in_root)
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try: 
        pBERT_FPs = data.smiles.map(PSMILES_pBERT_dict).apply(pd.Series)
        value_column_name = data.columns[1]
        values = data[value_column_name]
        pBERT_data = pd.concat([pBERT_FPs, values], axis=1)

        pBERT_data.to_csv(out_path,index=False)
    except Exception as e:
        print(f'Error processing {dataset}: {e}')
        #os.mkdir('./../../Datasets/polyBERT/Polymer_Genome/ionization_energy')
        #pBERT_data.to_csv(out_path,index=False)
