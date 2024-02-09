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




# syst_up = [f"{base}/{syst}/up" for syst in systematics if 'analysis_tar' not in syst]
# syst_down = [f"{base}/{syst}/down" for syst in systematics if 'analysis_tar' not in syst]

# masses = os.listdir(syst_up[0])

# bad_masses = []
# needs_prediction = []

# for up,down in tqdm(zip(syst_up,syst_down)):
#     for mass in masses:
#         f_up = f"{up}/{mass}/ntuple.root"
#         f_down = f"{down}/{mass}/ntuple.root"
        
#         try: 
#             tree = SixB(f_up)
#             del tree
#         except IndexError: needs_prediction.append(f_up)
#         except: bad_masses.append(f_up)

#         try: 
#             tree = SixB(f_down)
#             del tree
#         except IndexError: needs_prediction.append(f_down)
#         except: bad_masses.append(f_down)

# print("To resubmit: sent to resubmissions.txt")
# with open("resubmissions.txt", "w") as f:
#     f.writelines(bad_masses)

# ' '.join(needs_prediction)
# print("To predict: sent to predictions.txt")
# with open("predictions.txt", "w") as f:
#     f.writelines(needs_prediction)