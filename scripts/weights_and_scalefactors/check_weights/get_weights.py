# python scripts/feynnet/efficiencies/calculate_efficiency.py /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/NMSSM_XToYHTo6B_MX-1000_MY-250_TuneCP5_13TeV-madgraph-pythia8/ntuple.root

print("Importing libraries")
from utils import *
import os

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

def main(fname, cfg):
    try: tree = SixB(fname, feyn=cfg)
    except: 
        import traceback
        print(f"Failed to process {filename}")
        traceback.print_exc()

    savepath = "plots"
    if not os.path.exists(savepath): os.makedirs(savepath)

    metric_dict = {
        'X_m' : tree.X.m,
        'genWeight' : tree.genWeight,
        'w_pu': tree.w_pu,
        'w_pu_up': tree.w_pu_up,
        'w_pu_down': tree.w_pu_down,
        'w_puid' : tree.w_puid,
        'w_puid_up' : tree.w_puid_up,
        'w_puid_down' : tree.w_puid_down,
        'w_trigger' : tree.w_trigger,
        'w_trigger_up' : tree.w_trigger_up,
        'w_trigger_down' : tree.w_trigger_down,
    }

    np.savez(f"{savepath}/tmp/{tree.mxmy}_weights.npz", **metric_dict)

    tmpdir = f"{savepath}/tmp"
    if not os.path.exists(tmpdir): os.makedirs(tmpdir)

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
