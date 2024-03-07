# parallel -j 4 "python scripts/generate_combine_root_files.py {}" ::: $(cat filelists/central.txt) --eta
"""
Skim jobs fail quite often and there's no good way to know besides reading through all of the danged skim_job logs. Here's a way to check them and print which ones need to be rerun again or predicted with FeynNet.
"""

import os, sys
from utils.analysis.signal import SixB
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

fname = args.filename
masspoint = fname.split('/')[-2]

base = "/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/syst"
systematics = os.listdir(base)

for syst in systematics:
    f_up = f"{base}/{syst}/up/{masspoint}/ntuple.root"
    f_down = f"{base}/{syst}/down/{masspoint}/ntuple.root"
    
    try: tree = SixB(f_up)
    except IndexError: print(f"Prediction:\n{f_up}")
    except: print(f"Resubmit:\n{f_up}")

    try: tree = SixB(f_down)
    except IndexError: print(f"Prediction:\n{f_down}")
    except: print(f"Resubmit:\n{f_down}")