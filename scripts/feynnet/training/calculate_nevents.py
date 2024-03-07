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
args = parser.parse_args()

with open(args.filelist) as f:
    filelist = f.readlines()
filelist = [f.strip('\n') for f in filelist]

batches = np.array_split(filelist, args.nbatches)
batch = batches[args.batch]

def main(fname):
    # try: tree = SixB(fname, feyn=False)
    try: tree = uproot.open(f"{fname}:sixBtree")
    except: 
        import traceback
        print(f"Failed to process {filename}")
        traceback.print_exc()

    # savepath = 'plots/feynnet'

    metric_dict = {
        'nevents': len(tree['jet_pt'].array())
    }

    # tmpdir = f"{savepath}/tmp"
    # if not os.path.exists(tmpdir): os.makedirs(tmpdir)

    mxmy = re.search("(MX_.*MY_.*)[_/]", fname).group(1)

    with open(f"tmp/{mxmy}_nevents.json", 'w') as f:
    # with open(f"{tmpdir}/{mxmy}_efficiency.json", 'w') as f:
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
