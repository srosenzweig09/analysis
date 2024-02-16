"""
As of Feb 2024, when a model is used to obtain predictions, the output files are a JSON file and a ROOT file that are stored on EOS. The current structure is such that the json and root files share a hash and information about the masspoint is contained in the json file.

This script will merge the json files and create a dictionary 
"""

import os
import glob
import json
import sys
from utils.analysis.feyn import new_model_path
from tqdm import tqdm

directory_path = f"{new_model_path}/Summer2018UL"

def merge_json_files(directory):
    merged_data = {}
    json_files = glob.glob(os.path.join(directory, '*.json'))
    
    for file_path in tqdm(json_files):
        with open(file_path, 'r') as file:
            file_data = json.load(file)
            for key,val in file_data.items():
                # obtain hash from val and mass filename from key
                # key, val = key.split('/')[-2], val.split('/')[-1]
                hash_key = val.split('/')[-1]
                file_dict = {key : f"{directory_path}/{hash_key}.root"}
                # file_dict = {key.replace("root://cmseos.fnal.gov/", "/eos/uscms") : f"{directory_path}/{hash_key}.root"}
            merged_data.update(file_dict)
    
    return merged_data

# Example usage:
merged_result = merge_json_files(directory_path)
# for key in merged_result:
    # print(key, merged_result[key])

with open(f"{directory_path}/samples.json", 'w') as file:
    json.dump(merged_result, file)