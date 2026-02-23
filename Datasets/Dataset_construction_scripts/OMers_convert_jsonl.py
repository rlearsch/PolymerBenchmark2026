import json
import pandas as pd
import numpy as np
from rdkit import Chem
import re 
import ast
from pathlib import Path

# Import centralized conversion functions
from polymer_conversions import (
    PSMILES_to_wPSMILES,
    construct_alternating_wPSMILES,
    construct_random_wPSMILES,
    wPSMILES_to_PSMILES_alternating,
    join_by_pairs
)

# Use path relative to this script's location
script_dir = Path(__file__).parent
filename = str(script_dir / "files/Cleaned_OMersBench.jsonl")


def construct_alternating_wPSMILES_from_row(OMers_row):
    """Wrapper to call construct_alternating_wPSMILES with DataFrame row."""
    return construct_alternating_wPSMILES(
        OMers_row['copolymer_ratio'],
        OMers_row['smiles_list']
    )


def construct_random_wPSMILES_from_row(OMers_row):
    """Wrapper to call construct_random_wPSMILES with DataFrame row."""
    return construct_random_wPSMILES(
        OMers_row['copolymer_ratio'],
        OMers_row['smiles_list']
    )


homopolymer_inputs, homopolymer_outputs = [], []
random_copolymer_inputs, random_copolymer_outputs = [], []
alternating_copolymer_inputs, alternating_copolymer_outputs = [], []
with open(filename, 'r') as f:
    for line in f:
        try:
            json_object = json.loads(line.strip()) # strip() removes leading/trailing whitespace and newline characters
            inputs = json_object['inputs']
            outputs = json_object['outputs']
            if inputs['copolymer_architecture'] == "homopolymer":
                homopolymer_inputs.append(inputs)
                homopolymer_outputs.append(outputs)
            if inputs['copolymer_architecture'] == "alternating":
                alternating_copolymer_inputs.append(inputs)
                alternating_copolymer_outputs.append(outputs)
            if inputs['copolymer_architecture'] == "random":
                random_copolymer_inputs.append(inputs)
                random_copolymer_outputs.append(outputs)                
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line.strip()} - {e}")
# 'data' now contains a list of Python dictionaries, each representing a JSON object from the file.
copolymer_architectures = ['alternating','random','homopolymer']
for index, dictionaries in enumerate([[alternating_copolymer_inputs, alternating_copolymer_outputs],
                                      [random_copolymer_inputs, random_copolymer_outputs],
                                      [homopolymer_inputs, homopolymer_outputs]]):
    architecture = copolymer_architectures[index]
    dfs = [pd.DataFrame(dictionaries[0]), pd.DataFrame(dictionaries[1])]
    df = pd.concat(dfs, axis=1)
    # Save intermediate CSV in script directory
    df.to_csv(script_dir / f'OMersBench_{architecture}.csv', index=False)
    ## split into 300 atom and 5000 atom simulations
    ## if n_chains < 10: 300 atom
    ## if n_chains ==10: 5000 atom

    if architecture=='homopolymer':
        df['smiles'] = df['smiles_list']
        for measurement in ['Cp', 'Rg','density','refractive_index']: # save PSMILES
            df[df['n_chains'].astype(int) < 10][['smiles', measurement]].dropna().to_csv(script_dir / f'../../Datasets/PSMILES/MD_300/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().to_csv(script_dir / f'../../Datasets/PSMILES/MD_5000/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().sample(n=300,random_state=0).to_csv(script_dir / f'../../Datasets/PSMILES/MD_5000/{measurement}/{architecture}_300_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().sample(n=1000,random_state=0).to_csv(script_dir / f'../../Datasets/PSMILES/MD_5000/{measurement}/{architecture}_1000_{measurement}.csv', index=False)
        df['smiles'] = df['smiles'].apply(PSMILES_to_wPSMILES)
        for measurement in ['Cp', 'Rg','density','refractive_index']: #save wPSMILES
            df[df['n_chains'].astype(int) < 10][['smiles', measurement]].dropna().to_csv(script_dir / f'../../Datasets/wPSMILES/MD_300/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().to_csv(script_dir / f'../../Datasets/wPSMILES/MD_5000/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().sample(n=300,random_state=0).to_csv(script_dir / f'../../Datasets/wPSMILES/MD_5000/{measurement}/{architecture}_300_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().sample(n=1000,random_state=0).to_csv(script_dir / f'../../Datasets/wPSMILES/MD_5000/{measurement}/{architecture}_1000_{measurement}.csv', index=False)
    if architecture=='alternating':
        df['smiles'] = df.apply(construct_alternating_wPSMILES_from_row, axis=1)
        for measurement in ['Cp', 'Rg','density','refractive_index']: # save wPSMILES
            df[df['n_chains'].astype(int) < 10][['smiles', measurement]].dropna().to_csv(script_dir / f'../../Datasets/wPSMILES/MD_300/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().to_csv(script_dir / f'../../Datasets/wPSMILES/MD_5000/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().sample(n=300,random_state=0).to_csv(script_dir / f'../../Datasets/wPSMILES/MD_5000/{measurement}/{architecture}_300_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().sample(n=3000,random_state=0).to_csv(script_dir / f'../../Datasets/wPSMILES/MD_5000/{measurement}/{architecture}_3000_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().sample(n=10000,random_state=0).to_csv(script_dir / f'../../Datasets/wPSMILES/MD_5000/{measurement}/{architecture}_10000_{measurement}.csv', index=False)
        df['smiles']=df.smiles.apply(wPSMILES_to_PSMILES_alternating)
        for measurement in ['Cp', 'Rg','density','refractive_index']: # save PSMILES
            df[df['n_chains'].astype(int) < 10][['smiles', measurement]].dropna().to_csv(script_dir / f'../../Datasets/PSMILES/MD_300/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().to_csv(script_dir / f'../../Datasets/PSMILES/MD_5000/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().sample(n=300,random_state=0).to_csv(script_dir / f'../../Datasets/PSMILES/MD_5000/{measurement}/{architecture}_300_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().sample(n=3000,random_state=0).to_csv(script_dir / f'../../Datasets/PSMILES/MD_5000/{measurement}/{architecture}_3000_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().sample(n=10000,random_state=0).to_csv(script_dir / f'../../Datasets/PSMILES/MD_5000/{measurement}/{architecture}_10000_{measurement}.csv', index=False)
    if architecture=='random':
        df['smiles'] = df.apply(construct_random_wPSMILES_from_row, axis=1)
        for measurement in ['Cp', 'Rg','density','refractive_index']: # save wPSMILES
            df[df['n_chains'].astype(int) < 10][['smiles', measurement]].dropna().to_csv(script_dir / f'../../Datasets/wPSMILES/MD_300/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles', measurement]].dropna().to_csv(script_dir / f'../../Datasets/wPSMILES/MD_5000/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) < 10][['smiles_list','copolymer_ratio','copolymer_architecture', measurement]].dropna().to_csv(script_dir / f'../../Datasets/PSMILES/MD_300/{measurement}/{architecture}_{measurement}.csv', index=False)
            df[df['n_chains'].astype(int) == 10][['smiles_list','copolymer_ratio','copolymer_architecture', measurement]].dropna().to_csv(script_dir / f'../../Datasets/PSMILES/MD_5000/{measurement}/{architecture}_{measurement}.csv', index=False)
