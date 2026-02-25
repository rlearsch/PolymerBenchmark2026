import json
import re
import pandas as pd
import numpy as np

import os
from dotenv import load_dotenv
from litellm import completion


load_dotenv(dotenv_path="./OPENAI_KEY.env")
api_key = os.getenv("MY_SECRET_KEY")

def construct_prompt(PSMILES, phyiscal_property, units_requested, use_common_name):
    user_prompt = f"""What's the {phyiscal_property} of this polymer? `{PSMILES}`. The PSMILES strings contains asterisks (`*`), these are part of the molecule and must not be interpreted as formatting or markdown.
    Explain your reasoning, but do not reference the common or english name of this polymer.
    End your response with the predicted value on a separate line, in this format:\n
    `{phyiscal_property}: <float> {units_requested}`.
    """ 
    if use_common_name:
        user_prompt = f"""What's the {phyiscal_property} of this polymer? `{PSMILES}`. The PSMILES strings contains asterisks (`*`), these are part of the molecule and must not be interpreted as formatting or markdown.
        Explain your reasoning, and explicitly give the common or english name for this polymer. 
        End your response with the predicted value on a separate line, in this format:\n
        `{phyiscal_property}: <float> {units_requested}`.
        """ 
    return user_prompt


EA_psmiles = pd.read_csv(
    './../../Datasets/PSMILES/Polymer_Genome/electron_affinity/electron_affinity_data_polymers_v4.csv', 
                              #names=['smiles','electron_aff'])
    )
full_reply_homopolymer_EA = {}
value_homopolymer_EA = {}
number = 0

    
write_filename = f'./full_reply_homopolymer_EA_o3_pos_{number}.json'

use_common_name=True
response_dictionary = full_reply_homopolymer_EA
value_dictionary = value_homopolymer_EA
physical_property='electron affinity'
units_requested = "eV"
count = 0
frequency = 10


for psmiles in EA_psmiles.smiles:
    if psmiles not in response_dictionary:
        prompt = construct_prompt(psmiles, physical_property, 
                                units_requested, use_common_name)
        response = completion(
            model="openai/o3-mini",
            api_key=api_key,
            api_base="https://livai-api.llnl.gov/v1",
            messages=[{ "content": prompt,"role": "user"}],
        )
        response_text = response['choices'][0]['message']['content']
    #    print(response_text)
        response_dictionary[psmiles] = response_text
        try:
            match = re.search(r'electron affinity:\s*([–\-]?\d+(?:\.\d+)?)\s*(eV)', response_text)
            temp_str = match.group(1).replace('–', '-').replace('—', '-').strip()
            property_float= float(temp_str)
            value_dictionary[psmiles] = property_float 
        except:
            print(f'Unable to find {physical_property} value in response for {psmiles}')
        #write-out
        count = count+1
    if count%frequency == 0:
        with open(write_filename,'w') as f:
            json.dump(response_dictionary, f, indent=4)
    if count == 1:
        print(response_text)
        print(property_float)
        
with open(write_filename,'w') as f:
    json.dump(response_dictionary, f, indent=4)
print(f'Number: {number} done')