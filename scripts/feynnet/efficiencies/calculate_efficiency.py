# Main script:
#    sbatch scripts/feynnet/efficiencies/run
# To run alone:
#    python scripts/feynnet/efficiencies/calculate_efficiency.py /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/NMSSM_XToYHTo6B_MX-1000_MY-250_TuneCP5_13TeV-madgraph-pythia8/ntuple.root


print("Importing libraries")
from utils import *
import os, re
import json

from argparse import ArgumentParser
parser = ArgumentParser()
# parser.add_argument("filenames", nargs="?")
parser.add_argument("filelist")
parser.add_argument("--nbatches", type=int, default=1)
parser.add_argument("--batch", type=int, default=0)
parser.add_argument("--config", default="config/feynnet.cfg")
args = parser.parse_args()

with open(args.filelist) as f:
    filelist = f.readlines()
filelist = [f.strip('\n') for f in filelist]

batches = np.array_split(filelist, args.nbatches)
batch = batches[args.batch]
cfg = args.config
rand_num = re.search("feynnet_(.*).cfg", args.config).group(1)

def main(fname, cfg):
    try: tree = SixB(fname, feyn=cfg)
    except: 
        import traceback
        print(f"Failed to process {filename}")
        traceback.print_exc()

    savepath = tree.model.savepath

    feynnet_efficiency = round(ak.sum(tree.n_h_found[tree.resolved_mask] == 3) / ak.sum(tree.resolved_mask),3)
    resolved_efficiency = ak.sum(tree.resolved_mask)/len(tree.resolved_mask)
    sr_efficiency = ak.sum(tree.asr_hs_mask)/len(tree.asr_hs_mask)
    n_possible_higgs = np.average(tree.n_h_possible)

    metric_dict = {
        'feynnet_efficiency': feynnet_efficiency,
        'resolved_efficiency': resolved_efficiency,
        'sr_efficiency': sr_efficiency,
        'n_possible_higgs': n_possible_higgs
    }

    # tmpdir = f"{savepath}/tmp"
    # if not os.path.exists(tmpdir): os.makedirs(tmpdir)

    # mxmy = re.search("(MX_.*)/", tree.filepath).group(1)

    with open(f"tmp/{rand_num}_{tree.mxmy}_efficiency.json", 'w') as f:
    # with open(f"{tmpdir}/{mxmy}_efficiency.json", 'w') as f:
        json.dump(metric_dict, f)

    print("Done!")

failed_files = []
for filename in batch:
    try:
        main(filename, cfg)
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
