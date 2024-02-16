# parallel -j 4 "python scripts/calculate_2d_sf_corrs.py {}" ::: $(cat filelists/Summer2018UL/private.txt) --eta

# python scripts/calculate_2d_sf_corrs.py /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag/NMSSM/NMSSM_XYH_YToHH_6b_MX_1000_MY_250_2M/ntuple.root

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import uproot
from utils.analysis.signal import SixB
from utils.plotter import Hist2d
from colorama import Fore, Style
import re, sys

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

def get_2d_ratio(ratio, n, ht):
    i = np.digitize(n, n_bins) - 1
    j = np.digitize(ht, ht_bins) - 1
    if i < 0: i = 0
    if j < 0: j = 0
    print(i,j)
    return ratio[i,j]

def calculate_ratio(n_jet, ht, bsf, scale):
    fig, axs = plt.subplots(ncols=2, figsize=(16,6))

    ax = axs[0]
    n1, xe, xy, im = Hist2d(n_jet, ht, bins=[n_bins, ht_bins], ax=ax, weights=scale)
    fig.colorbar(im, ax=ax)

    ax = axs[1]
    n2, xe, xy, im = Hist2d(n_jet, ht, bins=[n_bins, ht_bins], ax=ax, weights=bsf*scale)
    fig.colorbar(im, ax=ax)

    ratio = np.nan_to_num(n1/n2, nan=1.0)
    plt.close()

    return ratio

filename = args.filename
year = re.search("ntuples/(.*?)/", args.filename).group(1)

try: tree = SixB(filename, feyn=False)
except uproot.exceptions.KeyInFileError: print(Fore.RED + f"File {filename} not found" + Style.RESET_ALL)

n_jet = tree.n_jet.to_numpy()
ht = tree.get('PFHT', library='np')
bsf_names = [key for key in tree.keys() if key.startswith('bSFshape')]
# print("bsf_names", bsf_names)
scale = np.repeat(tree.scale, tree.nevents)

n_bins = np.arange(6, n_jet.max() + 2)
x_n = (n_bins[1:] + n_bins[:-1]) / 2
bin_w = 25
ht_min = bin_w*(round(ht.min()/bin_w) - 1)
ht_max = bin_w*(round(ht.max()/bin_w) - 1)
ht_bins = np.arange(ht_min, ht_max, bin_w)
x_ht = (ht_bins[1:] + ht_bins[:-1]) / 2

ratios = {name:calculate_ratio(n_jet, ht, tree.get(name, library='np'), scale) for name in bsf_names}

with uproot.recreate(f'data/{year}/btag/MX_{tree.mx}_MY_{tree.my}.root') as f:
    for name in bsf_names:
        h_name = f"MX_{tree.mx}_MY_{tree.my}_{name}"
        f[h_name] = (ratios[name], n_bins, ht_bins)