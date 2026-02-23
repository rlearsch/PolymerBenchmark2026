from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from pathlib import Path

# Import centralized conversion functions
from polymer_conversions import wPSMILES_to_PSMILES_alternating

# Use paths relative to this script's location
script_dir = Path(__file__).parent


wPSMILES_data = pd.read_csv(script_dir / 'files/polymer-chemprop-data/dataset-poly_chemprop.csv')
polymer_information = pd.read_csv(script_dir / 'files/polymer-chemprop-data/dataset.csv')

df = wPSMILES_data.reset_index().merge(polymer_information.reset_index(), on='index')
df['EA']=df['EA (eV)']
df['IP']=df['IP (eV)']
#block copolymers
copolymer_architectures = ['alternating','random','block']
copolymer_quantities = ['EA','IP']
for architecture in copolymer_architectures:
    for quantity in copolymer_quantities:
        df_temp = df[df.poly_type==architecture]
        df_temp = df_temp.reset_index(drop=True)
        df_temp['smiles'] = df_temp.poly_chemprop_input
        df_temp[['smiles',quantity]].to_csv(script_dir / f'../../Datasets/wPSMILES/Coley_2022/{quantity}/{architecture}_{quantity}.csv', index=False)
        if architecture == 'alternating':
            df_temp['smiles']=df_temp.smiles.apply(wPSMILES_to_PSMILES_alternating)
            df_temp[['smiles',quantity]].to_csv(script_dir / f'../../Datasets/PSMILES/Coley_2022/{quantity}/{architecture}_{quantity}.csv', index=False)


#random copolymers        
        
        
