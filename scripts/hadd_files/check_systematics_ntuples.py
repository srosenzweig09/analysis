from colorama import Fore, Style
import os
import re
import shlex
import subprocess
import sys
import uproot

def check_mass_point(f):
    try: 
        tree = uproot.open(f)
        check1 = len(tree['sixBtree']['n_jet'].array()) == tree['h_cutflow']..to_numpy()[0][-1]
        check2 = len(tree['sixBtree']['n_jet'].array()) == tree['h_cutflow_unweighted']..to_numpy()[0][-1]
        del tree
        return (check1 or check2)
    except:
        return False

cfg=f"config/skim_ntuple_2018_106X_NanoAODv9.cfg"
sh_cmd="scripts/6b_scripts/submit_all_6b_signal.sh"
file_loc = "/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_6_19_patch2/src/MultiHiggs/analysis/MultiHAnalysis"

with open(f"{file_loc}/data/jec/RegroupedV2_Summer19UL18_V5_MC_UncertaintySources_AK4PFchs.txt") as f:
   output = f.read()
   result = re.findall('\[(.*)\]', output, re.MULTILINE)
systematics = result[:-1]

print(f"[INFO] {', '.join(systematics)}")

input_dir = 'input/Run2_UL/RunIISummer20UL18NanoAODv9/NMSSM_XToYHTo6B'
cmd = f'ls {file_loc}/{input_dir}'
output = subprocess.check_output(shlex.split(cmd)).decode('utf-8').split('\n')
output = [out for out in output if 'NMSSM' in out]
signals = [out for out in output if int(out.split('_')[2].split('-')[1]) < 1300]

cmds = []
for signal in signals:
   print(f"[INFO] Processing signal: {Fore.MAGENTA}{signal}{Style.RESET_ALL}")
   for syst in systematics:
        print(f"[INFO] Processing systematic: {Fore.CYAN}{syst}{Style.RESET_ALL}:up")
        tag = f'Summer2018UL/maxbtag_4b/Official_NMSSM/syst/{syst}/up'
        if not check_mass_point(f"/eos/uscms/store/user/srosenzw/sixb/ntuples/{tag}/{signal.replace('.txt', '')}/ntuple.root"):
            submit_cmd = f'python scripts/submitSkimOnBatch.py --tag {tag} --jes {syst}:up --outputDir /store/user/srosenzw/sixb/ntuples --cfg config/skim_ntuple_2018_106X_NanoAODv9.cfg --njobs 100 --input {input_dir}/{signal} --is-signal --memory 4000 --forceOverwrite'
            cmds.append(submit_cmd)
            # print(f"{Style.DIM}{submit_cmd}{Style.RESET_ALL}")
        # subprocess.run(shlex.split(submit_cmd))


        print(f"[INFO] Processing systematic: {Fore.CYAN}{syst}{Style.RESET_ALL}:down")
        tag = f'Summer2018UL/maxbtag_4b/Official_NMSSM/syst/{syst}/down'
        if not check_mass_point(f"/eos/uscms/store/user/srosenzw/sixb/ntuples/{tag}/{signal.replace('.txt', '')}/ntuple.root"):
            submit_cmd = f'python scripts/submitSkimOnBatch.py --tag {tag} --jes {syst}:down --outputDir /store/user/srosenzw/sixb/ntuples --cfg config/skim_ntuple_2018_106X_NanoAODv9.cfg --njobs 100 --input {input_dir}/{signal} --is-signal --memory 4000 --forceOverwrite'
            cmds.append(submit_cmd)
            # print(f"{Style.DIM}{submit_cmd}{Style.RESET_ALL}")
            # subprocess.run(shlex.split(submit_cmd))
         
for cmd in cmds: print(cmd)