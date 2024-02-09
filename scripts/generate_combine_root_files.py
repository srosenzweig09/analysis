# parallel -j 10 "python scripts/generate_combine_root_files.py {}" ::: $(cat filelists/central.txt) --eta
"""
This script will build the individual root files for each mass point, containing the nominal MX distribution, as well as the distributions for each systematic variation.

python scripts/generate_combine_root_files.py /eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/NMSSM_XToYHTo6B_MX-500_MY-300_TuneCP5_13TeV-madgraph-pythia8/ntuple.root
python scripts/generate_combine_root_files.py /eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/NMSSM_XToYHTo6B_MX-950_MY-700_TuneCP5_13TeV-madgraph-pythia8/ntuple.root
python scripts/generate_combine_root_files.py /eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM/NMSSM_XToYHTo6B_MX-1000_MY-800_TuneCP5_13TeV-madgraph-pythia8/ntuple.root
"""

from argparse import ArgumentParser
from colorama import Fore, Style
import ROOT
ROOT.gROOT.SetBatch(True)
from array import array
import numpy as np
import matplotlib.pyplot as plt
# import subprocess
from configparser import ConfigParser
from utils.plotter import Hist
from utils.analysis.signal import SixB
from utils.analysis.feyn import model_name
fsave = f'combine/feynnet/{model_name}'
import sys

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

config = ConfigParser()
config.read("config/bdt_params.cfg")

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

systematics = ['JERpt', 'bJER','Absolute_2018', 'Absolute', 'BBEC1', 'BBEC1_2018', 'EC2', 'EC2_2018', 'FlavorQCD', 'HF', 'HF_2018', 'RelativeBal', 'RelativeSample_2018']

def raiseError(fname):
    print(f"{Fore.RED}[FAILED]{Style.RESET_ALL} Cannot open file:\n{fname}")
    raise

def writeHist(h_title, n):
    ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins,array('d',list(bins)))
    for i,(val) in enumerate(n):
        ROOT_hist.SetBinContent(i+1, val)

    ROOT_hist.Draw("hist")
    ROOT_hist.Write()

# base = "/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/Official_NMSSM"
# output = subprocess.check_output(f"ls {base}", shell=True).decode('utf-8').split('\n')
# signals = [out for out in output if 'NMSSM' in out]
# signal = "NMSSM_XToYHTo6B_MX-850_MY-350_TuneCP5_13TeV-madgraph-pythia8"
# fname = f"{base}/{signal}/ntuple.root"

fname = args.filename

try: tree = SixB(fname)
except: raiseError(fname)
tree.spherical_region()
mx, my = tree.mx, tree.my
MX = tree.X.m[tree.asr_hs_mask]

genWeight = tree.genWeight

PUWeight = tree.PUWeight
PUWeight_up = tree.PUWeight_up
PUWeight_down = tree.PUWeight_down

PUIDWeight = tree.PUIDWeight
PUIDWeight_up = tree.PUIDWeight_up
PUIDWeight_down = tree.PUIDWeight_down

btag_central = tree.bSFshape_central
btag_hf_up = tree.bSFshape_up_hf
btag_hf_down = tree.bSFshape_down_hf
btag_lf_up = tree.bSFshape_up_lf
btag_lf_down = tree.bSFshape_down_lf
btag_hfstats1_up = tree.bSFshape_up_hfstats1
btag_hfstats1_down = tree.bSFshape_down_hfstats1
btag_hfstats2_up = tree.bSFshape_up_hfstats2
btag_hfstats2_down = tree.bSFshape_down_hfstats2
btag_lfstats1_up = tree.bSFshape_up_lfstats1
btag_lfstats1_down = tree.bSFshape_down_lfstats1
btag_lfstats2_up = tree.bSFshape_up_lfstats2
btag_lfstats2_down = tree.bSFshape_down_lfstats2

triggerSF = tree.triggerSF
triggerSF_up = tree.triggerSF_up
triggerSF_down = tree.triggerSF_down

nominal_weight = (genWeight*PUWeight*PUIDWeight*btag_central*triggerSF)[tree.asr_hs_mask]
PUWeight_up_weight = (genWeight*PUWeight_up*PUIDWeight*btag_central*triggerSF)[tree.asr_hs_mask]
PUWeight_down_weight = (genWeight*PUWeight_down*PUIDWeight*btag_central*triggerSF)[tree.asr_hs_mask]
triggerSF_up_weight = (genWeight*PUWeight*PUIDWeight*btag_central*triggerSF_up)[tree.asr_hs_mask]
triggerSF_down_weight = (genWeight*PUWeight*PUIDWeight*btag_central*triggerSF_down)[tree.asr_hs_mask]
btag_hf_up_weight = (genWeight*PUWeight*PUIDWeight*btag_hf_up*triggerSF)[tree.asr_hs_mask]
btag_hf_down_weight = (genWeight*PUWeight*PUIDWeight*btag_hf_down*triggerSF)[tree.asr_hs_mask]
btag_lf_up_weight = (genWeight*PUWeight*PUIDWeight*btag_lf_up*triggerSF)[tree.asr_hs_mask]
btag_lf_down_weight = (genWeight*PUWeight*PUIDWeight*btag_lf_down*triggerSF)[tree.asr_hs_mask]
btag_hfstats1_up_weight = (genWeight*PUWeight*PUIDWeight*btag_hfstats1_up*triggerSF)[tree.asr_hs_mask]
btag_hfstats1_down_weight = (genWeight*PUWeight*PUIDWeight*btag_hfstats1_down*triggerSF)[tree.asr_hs_mask]
btag_hfstats2_up_weight = (genWeight*PUWeight*PUIDWeight*btag_hfstats2_up*triggerSF)[tree.asr_hs_mask]
btag_hfstats2_down_weight = (genWeight*PUWeight*PUIDWeight*btag_hfstats2_down*triggerSF)[tree.asr_hs_mask]
btag_lfstats1_up_weight = (genWeight*PUWeight*PUIDWeight*btag_lfstats1_up*triggerSF)[tree.asr_hs_mask]
btag_lfstats1_down_weight = (genWeight*PUWeight*PUIDWeight*btag_lfstats1_down*triggerSF)[tree.asr_hs_mask]
btag_lfstats2_up_weight = (genWeight*PUWeight*PUIDWeight*btag_lfstats2_up*triggerSF)[tree.asr_hs_mask]
btag_lfstats2_down_weight = (genWeight*PUWeight*PUIDWeight*btag_lfstats2_down*triggerSF)[tree.asr_hs_mask]
PUIDWeight_up_weight = (genWeight*PUWeight*PUIDWeight_up*btag_central*triggerSF)[tree.asr_hs_mask]
PUIDWeight_down_weight = (genWeight*PUWeight*PUIDWeight_down*btag_central*triggerSF)[tree.asr_hs_mask]

fout = f"{fsave}/MX_{mx}_MY_{my}.root"
print(f"Saving to {fout}")
# numbered to avoid annoying message about canvas with same name
# canvas = ROOT.TCanvas(f'c{i}',f'c{i}', 600, 600)
canvas = ROOT.TCanvas(f'c0',f'c0', 600, 600)
canvas.SetFrameLineWidth(3)
canvas.Draw()

fout = ROOT.TFile(fout,"recreate")
fout.cd()

fig, ax = plt.subplots()
n_nominal = Hist(MX, bins=bins, weights=nominal_weight, ax=ax)
n_trigger_up = Hist(MX, bins=bins, weights=triggerSF_up_weight, ax=ax)
n_trigger_down = Hist(MX, bins=bins, weights=triggerSF_down_weight, ax=ax)
n_PU_up = Hist(MX, bins=bins, weights=PUWeight_up_weight, ax=ax)
n_PU_down = Hist(MX, bins=bins, weights=PUWeight_down_weight, ax=ax)
n_hf_up = Hist(MX, bins=bins, weights=btag_hf_up_weight, ax=ax)
n_hf_down = Hist(MX, bins=bins, weights=btag_hf_down_weight, ax=ax)
n_lf_up = Hist(MX, bins=bins, weights=btag_lf_up_weight, ax=ax)
n_lf_down = Hist(MX, bins=bins, weights=btag_lf_down_weight, ax=ax)
n_hfstats1_up = Hist(MX, bins=bins, weights=btag_hfstats1_up_weight, ax=ax)
n_hfstats1_down = Hist(MX, bins=bins, weights=btag_hfstats1_down_weight, ax=ax)
n_hfstats2_up = Hist(MX, bins=bins, weights=btag_hfstats2_up_weight, ax=ax)
n_hfstats2_down = Hist(MX, bins=bins, weights=btag_hfstats2_down_weight, ax=ax)
n_lfstats1_up = Hist(MX, bins=bins, weights=btag_lfstats1_up_weight, ax=ax)
n_lfstats1_down = Hist(MX, bins=bins, weights=btag_lfstats1_down_weight, ax=ax)
n_lfstats2_up = Hist(MX, bins=bins, weights=btag_lfstats2_up_weight, ax=ax)
n_lfstats2_down = Hist(MX, bins=bins, weights=btag_lfstats2_down_weight, ax=ax)
n_puid_up = Hist(MX, bins=bins, weights=PUIDWeight_up_weight, ax=ax)
n_puid_down = Hist(MX, bins=bins, weights=PUIDWeight_down_weight, ax=ax)

writeHist("signal", n_nominal)
writeHist("signal_TriggerUp", n_trigger_up)
writeHist("signal_TriggerDown", n_trigger_down)
writeHist("signal_PileupUp", n_PU_up)
writeHist("signal_PileupDown", n_PU_down)
writeHist("signal_BTagHFUp", n_hf_up)
writeHist("signal_BTagHFDown", n_hf_down)
writeHist("signal_BTagLFUp", n_lf_up)
writeHist("signal_BTagLFDown", n_lf_down)
writeHist("signal_BTagHFStats1Up", n_hfstats1_up)
writeHist("signal_BTagHFStats1Down", n_hfstats1_down)
writeHist("signal_BTagHFStats2Up", n_hfstats2_up)
writeHist("signal_BTagHFStats2Down", n_hfstats2_down)
writeHist("signal_BTagLFStats1Up", n_lfstats1_up)
writeHist("signal_BTagLFStats1Down", n_lfstats1_down)
writeHist("signal_BTagLFStats2Up", n_lfstats2_up)
writeHist("signal_BTagLFStats2Down", n_lfstats2_down)
writeHist("signal_PUIDUp", n_puid_up)
writeHist("signal_PUIDDown", n_puid_down)

for systematic in systematics:
    tmp = fname.split('/')
    tmp.insert(11, "syst")
    tmp.insert(12, systematic)
    tmp.insert(13, "up")
    up = '/'.join(tmp)
    # try: 
    try: up = SixB(up)
    except: raiseError(up)
    up.spherical_region()
    genWeight = up.genWeight
    PUWeight = up.PUWeight
    PUIDWeight = up.PUIDWeight
    triggerSF = up.triggerSF

    if "jer" in systematic.lower(): btag_weight = getattr(up, f'bSFshape_central')
    elif '2018' in systematic: btag_weight = getattr(up, f'bSFshape_up_jes{systematic.replace("_2018","")}')
    else: btag_weight = getattr(up, f'bSFshape_up_jes{systematic}')

    weight = (genWeight*PUWeight*PUIDWeight*triggerSF*btag_weight)[up.asr_hs_mask]
    n_syst = Hist(up.X.m[up.asr_hs_mask], bins=bins, weights=weight, ax=ax)
    writeHist(f'signal_{systematic}Up', n_syst)
    # except: print(f"[FAILED] Cannot open file: {up}")

    tmp[13] = "down"
    down = '/'.join(tmp)
    # try:
    try: down = SixB(down)
    except: raiseError(down)
    down.spherical_region()
    genWeight = down.genWeight
    PUWeight = down.PUWeight
    PUIDWeight = down.PUIDWeight
    triggerSF = down.triggerSF
    if "jer" in systematic.lower(): btag_weight = getattr(down, f'bSFshape_central')
    elif '2018' in systematic: btag_weight = getattr(down, f'bSFshape_down_jes{systematic.replace("_2018","")}')
    else: btag_weight = getattr(down, f'bSFshape_down_jes{systematic}')
    weight = (genWeight*PUWeight*PUIDWeight*triggerSF*btag_weight)[down.asr_hs_mask]
    n_syst = Hist(down.X.m[down.asr_hs_mask], bins=bins, weights=weight, ax=ax)
    writeHist(f'signal_{systematic}Down', n_syst)
    # except: print(f"{Fore.RED}[FAILED]{Style.RESET_ALL} Cannot open file: {down}")


ROOT.gStyle.SetOptStat(0)
fout.Close()
del canvas
