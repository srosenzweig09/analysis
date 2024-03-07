# python scripts/feynnet/ranking/calculate_efficiency.py /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/NMSSM_XToYHTo6B_MX-1000_MY-250_TuneCP5_13TeV-madgraph-pythia8/ntuple.root

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

    temp = tree.n_h_found[tree.resolved_mask]
    maxlength = tree.combos.to_numpy().shape[1]
    print(maxlength)
    # scan through the different ranking combinations to see where all three higgs are paired correctly
    for i in range(1, maxlength):
        tree.model.init_particles(tree, combo=i)
        temp = np.column_stack((temp, tree.model.n_h_found[tree.resolved_mask]))
    ind = ak.argsort(temp == 3, ascending=False, axis=1)
    correct_rank = ak.where(ak.sum(temp == 3, axis=1) > 0, ind[:,0], maxlength) + 1

    # tmpdir = f"{savepath}/tmp"
    # if not os.path.exists(tmpdir): os.makedirs(tmpdir)
    
    np.save(f"tmp/{rand_num}_{tree.mxmy}_ranking.npy", correct_rank)
    # avg = np.average(correct_rank)

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
