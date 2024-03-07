# python scripts/minimum_diagonal/efficiencies/calculate_efficiency.py /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/NMSSM_XToYHTo6B_MX-1000_MY-250_TuneCP5_13TeV-madgraph-pythia8/ntuple.root

from utils import *
import os
import json

from argparse import ArgumentParser
parser = ArgumentParser()
# parser.add_argument("filename")
parser.add_argument("filelist")
parser.add_argument("--nbatches", type=int, default=1)
parser.add_argument("--batch", type=int, default=0)
args = parser.parse_args()

with open(args.filelist) as f:
    filelist = f.readlines()
filelist = [f.strip('\n') for f in filelist]

batches = np.array_split(filelist, args.nbatches)
batch = batches[args.batch]

def raiseError(fname):
    print(f"{Fore.RED}[FAILED]{Style.RESET_ALL} Cannot open file:\n{fname}")
    raise


def main(fname):
    try: tree = SixB(fname, feyn=False)
    except: raiseError(fname)

    savepath = 'plots/minimum_diagonal'

    resolved_efficiency = ak.sum(tree.resolved_mask)/len(tree.resolved_mask)
    sr_efficiency = ak.sum(tree.asr_hs_mask)/len(tree.asr_hs_mask)
    higgs_efficiency = round(ak.sum(tree.n_h_found[tree.resolved_mask] == 3) / ak.sum(tree.resolved_mask),3)
    n_possible_higgs = np.average(tree.n_h_possible)

    metric_dict = {
        'higgs_efficiency' : higgs_efficiency,
        'resolved_efficiency': resolved_efficiency,
        'sr_efficiency': sr_efficiency,
        'n_possible_higgs': n_possible_higgs
    }   

    tmpdir = f"{savepath}/tmp"
    if not os.path.exists(tmpdir): os.makedirs(tmpdir)

    with open(f"{tmpdir}/{tree.mxmy}_efficiency.json", 'w') as f:
        json.dump(metric_dict, f)

    print("Done!")  

failed_files = []
for filename in batch:
    try:
        main(filename)
    except:
        import traceback
        failed_files.append(filename)
        print(f"Failed to process {filename}")
        traceback.print_exc()

if any(failed_files):
    print(f"{Fore.RED}Failed to process the following files:{Style.RESET_ALL}")
    for f in failed_files:
        print(f)

    exit(1)

print(f"{Fore.GREEN}All files processed successfully!{Style.RESET_ALL}")
