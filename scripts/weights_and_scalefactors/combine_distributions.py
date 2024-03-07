# sh scripts/parallel/runPythonWithFilelist.sh scripts/combine_distributions.py filelists/Summer2018UL/central.txt

"""
This script will build the individual root files for each mass point, containing the nominal MX distribution, as well as the distributions for each systematic variation.

python scripts/weights_and_scalefactors/combine_distributions.py /cmsuf/data/store/user/srosenzw/root/cmseos.fnal.gov/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/NMSSM_XToYHTo6B_MX-1000_MY-250_TuneCP5_13TeV-madgraph-pythia8/ntuple.root
"""

import awkward as ak
from colorama import Fore, Style
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
# from utils.plotter import Hist
from utils.analysis.signal import SixB
from utils.filelists import hpg_base
import os, sys
import uproot
import boost_histogram as bh

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

config = ConfigParser()
config.read("config/bdt_Summer2018UL.cfg")

style = config['plot']['style']
minMX = int(config['plot']['minMX'])
maxMX = int(config['plot']['maxMX'])
if style == 'linspace': 
    edges = int(config['plot']['edges'])
    bins = np.linspace(minMX, maxMX, edges)
elif style == 'arange':
    step = int(config['plot']['steps'])
    bins = np.arange(minMX, maxMX, step)
else: raise ValueError(f"Unknown style: {style}")
nbins = len(bins) - 1

systematics = os.listdir(f"{hpg_base}/Summer2018UL/maxbtag_4b/Official_NMSSM/syst")

def raiseError(fname):
    print(f"{Fore.RED}[FAILED]{Style.RESET_ALL} Cannot open file:\n{fname}")
    raise

def get_fname(fname, systematic, var):
    tmp = fname.split('/')
    tmp.insert(16, "syst")
    tmp.insert(17, systematic)
    tmp.insert(18, var)
    return '/'.join(tmp)

def btag_weight(tree, var, systematic):
    if "jer" in systematic.lower(): btag_weight = getattr(tree, f'bSFshape_central')
    elif '2018' in systematic: btag_weight = getattr(tree, f'bSFshape_{var}_jes{systematic.replace("_2018","")}')
    else: btag_weight = getattr(tree, f'bSFshape_{var}_jes{systematic}')
    return btag_weight

def get_boost(x, bins, weight):
    x = ak.to_numpy(x)
    weight = ak.to_numpy(weight)
    n, _ = np.histogram(x, bins=bins, weights=weight)
    v, _ = np.histogram(x, bins=bins, weights=weight**2)

    H = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
    H[...] = np.stack([n, v]).T
    return H

def makeDir(path):
    if not os.path.exists(path): os.makedirs(path)

fname = args.filename

try: tree = SixB(fname)
except: raiseError(fname)
MX, MY = tree.mx, tree.my
fsave = f'combine/feynnet/{tree.model.model_name}'
makeDir(fsave)
fout = f"{fsave}/MX_{MX}_MY_{MY}.root"
print(fout)

asr_mask = tree.asr_hs_mask
mx = tree.X.m[asr_mask]


boost_dict = {}
fig, ax = plt.subplots()
boost_dict['signal'] = get_boost(mx, bins, tree.w_nominal[asr_mask])
boost_dict['signal_TriggerUp'] = get_boost(mx, bins, tree.w_triggerUp[asr_mask])
boost_dict['signal_TriggerDown'] = get_boost(mx, bins, tree.w_triggerDown[asr_mask])
boost_dict['signal_PileupUp'] = get_boost(mx, bins, tree.w_PUUp[asr_mask])
boost_dict['signal_PileupDown'] = get_boost(mx, bins, tree.w_PUDown[asr_mask])
boost_dict['signal_PUIDUp'] = get_boost(mx, bins, tree.w_PUIDUp[asr_mask])
boost_dict['signal_PUIDDown'] = get_boost(mx, bins, tree.w_PUIDDown[asr_mask])
boost_dict['signal_BTagHFUp'] = get_boost(mx, bins, tree.w_HFUp[asr_mask])
boost_dict['signal_BTagHFDown'] = get_boost(mx, bins, tree.w_HFDown[asr_mask])
boost_dict['signal_BTagLFUp'] = get_boost(mx, bins, tree.w_LFUp[asr_mask])
boost_dict['signal_BTagLFDown'] = get_boost(mx, bins, tree.w_LFDown[asr_mask])
boost_dict['signal_BTagHFStats1Up'] = get_boost(mx, bins, tree.w_HFStats1Up[asr_mask])
boost_dict['signal_BTagHFStats1Down'] = get_boost(mx, bins, tree.w_HFStats1Down[asr_mask])
boost_dict['signal_BTagHFStats2Up'] = get_boost(mx, bins, tree.w_HFStats2Up[asr_mask])
boost_dict['signal_BTagHFStats2Down'] = get_boost(mx, bins, tree.w_HFStats2Down[asr_mask])
boost_dict['signal_BTagLFStats1Up'] = get_boost(mx, bins, tree.w_LFStats1Up[asr_mask])
boost_dict['signal_BTagLFStats1Down'] = get_boost(mx, bins, tree.w_LFStats1Down[asr_mask])
boost_dict['signal_BTagLFStats2Up'] = get_boost(mx, bins, tree.w_LFStats2Up[asr_mask])
boost_dict['signal_BTagLFStats2Down'] = get_boost(mx, bins, tree.w_LFStats2Down[asr_mask])

for systematic in systematics:
    fUp = get_fname(fname, systematic, "up")
    try: up = SixB(fUp)
    except: raiseError(fUp)

    w_btag = btag_weight(up, "up", systematic)
    weight = (up.genWeight*up.w_pu*up.w_puid*up.triggerSF*w_btag)[up.asr_hs_mask]
    H_syst = get_boost(up.X.m[up.asr_hs_mask], bins=bins, weight=weight)
    boost_dict[f"signal_{systematic}Up"] = H_syst
    

    fDown = get_fname(fname, systematic, "down")
    try: down = SixB(fDown)
    except: raiseError(fDown)

    w_btag = btag_weight(down, "down", systematic)
    weight = (down.genWeight*down.w_pu*down.w_puid*down.triggerSF*w_btag)[down.asr_hs_mask]
    H_syst = get_boost(down.X.m[down.asr_hs_mask], bins=bins, weight=weight)
    boost_dict[f"signal_{systematic}Down"] = H_syst


with uproot.recreate(fout) as file:
    for key,val in boost_dict.items():
        file[key] = val