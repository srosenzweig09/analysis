print("---------------------------------")
print("STARTING PROGRAM")
print("---------------------------------")

from argparse import ArgumentParser
from array import array
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np
import os
# import re
import ROOT
import sys
from utils.analysis import Signal
from utils.plotter import Hist, Ratio

def getROOTCanvas(h_title, bin_values, outFile):
    """Builds the canvas, draws the histogram, and saves to ROOT file"""
    print(f".. generating root hist for {h_title}")
    canvas = ROOT.TCanvas('c1','c1', 600, 600)
    canvas.SetFrameLineWidth(3)

    ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",len(bin_values),array('d',list(mBins)))

    for i,(bin_vals) in enumerate(bin_values):
        ROOT_hist.SetBinContent(i+1, bin_vals)

    ROOT_hist.Draw("hist")
    canvas.Draw()
    ROOT.gStyle.SetOptStat(0)

    canvas.Print(f"{outFile}.pdf)","Title:Signal Region");

    fout = ROOT.TFile(f"{outFile}.root","recreate")
    fout.cd()
    ROOT_hist.Write()
    fout.Close()

## ------------------------------------------------------------------------------------
## Implement command line parser

print(".. parsing command line arguments")

parser = ArgumentParser(description='Command line parser of model options and tags')

# parser.add_argument('--cfg', dest='cfg', help='config file', default='')
parser.add_argument('--testing', dest='testing', action='store_true', default=False)
parser.add_argument('--plot', dest='plot', action='store_true', default=False)

# jets
parser.add_argument('--btag', dest='btag', action='store_true', default=False)

# region shapes
parser.add_argument('--rectangular', dest='rectangular', help='', action='store_true', default=False)

# bdt parameters
parser.add_argument('--Nestimators', dest='N', type=int)
parser.add_argument('--learningRate', dest='lr', type=float)
parser.add_argument('--maxDepth', dest='depth', type=int)
parser.add_argument('--minLeaves', dest='minLeaves', type=int)
parser.add_argument('--GBsubsample', dest='gbsub', type=float)
parser.add_argument('--randomState', dest='rand', default=2020, type=int)

# stats parameters
parser.add_argument('--no-stats', dest='no_stats', action='store_true', default=False)
parser.add_argument('--no-MCstats', dest='MCstats', action='store_false', default=True)

args = parser.parse_args()

jets = 'bias'
if args.btag: jets = 'btag'

### ------------------------------------------------------------------------------------
## Implement config parser

print(".. parsing config file")

region_type = 'sphere'
cfg = f'config/{jets}_config.cfg'
if args.rectangular: 
   region_type = 'rect'
   cfg = f'config/rectConfig.cfg'

config = ConfigParser()
config.optionxform = str
config.read(cfg)

base = config['file']['base']
data = config['file']['data']

minMX = int(config['plot']['minMX'])
maxMX = int(config['plot']['maxMX'])
if config['plot']['style'] == 'linspace':
   nbins = int(config['plot']['nbins'])
   mBins = np.linspace(minMX,maxMX,nbins)
if config['plot']['style'] == 'arange':
   step = int(config['plot']['steps'])
   mBins = np.arange(minMX,maxMX,step)


indir = f"root://cmseos.fnal.gov/{base}"
outDir = f"combine/{jets}_{region_type}"
loc = f"{os.getcwd()}/{outDir}"

datFileName = f"{indir}{data}"

### ------------------------------------------------------------------------------------
## Set BDT parameters

# BDT_dict = {
#     'Nestimators' : args.N,
#     'learningRate' : args.lr,
#     'maxDepth' : args.depth,
#     'minLeaves' : args.minLeaves,
#     'GBsubsample' : args.gbsub,
#     'randomState' : args.rand
# }

# overwrite_log = []
# for hyper in BDT_dict:
#     if BDT_dict[hyper] is not None:
#         config['BDT'][hyper] = BDT_dict[hyper]
#         overwrite_log.append(f"Overwriting [{hyper}={BDT_dict[hyper]}] from config file with command line argument.")
# if len(overwrite_log) > 0: overwrite_log.insert(0, '')
# for line in overwrite_log:
#     print(overwrite_log)
# print()

# datacardDir = f"{combineDir}datacards/{jets}_{region_type}"
# print(datacardDir)

### ------------------------------------------------------------------------------------
## Obtain data regions

datTree = Signal(datFileName)

if args.rectangular:
    print("\n ---RECTANGULAR---")
    datTree.rectangular_region(config)
else:
    print("\n ----SPHERICAL----")
    datTree.spherical_region(config)

dat_mX_V_CRls = datTree.dat_mX_V_CRls
dat_mX_V_CRhs = datTree.dat_mX_V_CRhs
dat_mX_V_SRls = datTree.dat_mX_V_SRls
dat_mX_V_SRhs = datTree.dat_mX_V_SRhs
dat_mX_A_CRls = datTree.dat_mX_A_CRls
dat_mX_A_CRhs = datTree.dat_mX_A_CRhs
dat_mX_A_SRls = datTree.dat_mX_A_SRls


if args.plot:

   ## CONSTANT WEIGHTING   
   ratio = len(dat_mX_V_CRhs) / len(dat_mX_V_CRls)
   print("ratio",ratio)

   fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
   fig.suptitle("Validation Control Region")
   axs = Ratio([dat_mX_V_CRhs, dat_mX_V_CRls], bins=mBins, xlabel=r"M$_\mathrm{X}$ [GeV]", axs=axs, labels=['Target', 'Ratio Reweight'], ratio_ylabel='Target/Reweight', weights=[None,ratio])
   fig.savefig('plots/TF_target_comparison_VCR.pdf', bbox_inches='tight')

   fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
   fig.suptitle("Validation Signal Region")
   axs = Ratio([dat_mX_V_SRhs, dat_mX_V_SRls], bins=mBins, xlabel=r"M$_\mathrm{X}$ [GeV]", axs=axs, labels=['Target', 'Ratio Reweight'], ratio_ylabel='Target/Reweight', weights=[None,ratio])
   fig.savefig('plots/TF_target_comparison_VSR.pdf', bbox_inches='tight')

   fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
   fig.suptitle("Analysis Control Region")
   axs = Ratio([dat_mX_A_CRhs, dat_mX_A_CRls], bins=mBins, xlabel=r"M$_\mathrm{X}$ [GeV]", axs=axs, labels=['Target', 'Ratio Reweight'], ratio_ylabel='Target/Reweight', weights=[None,ratio])
   fig.savefig('plots/TF_target_comparison_ACR.pdf', bbox_inches='tight')

### ------------------------------------------------------------------------------------
## train bdt and predict data SR weights

datTree.bdt_process(region_type, config)

if args.testing: sys.exit()

V_CR_weights = datTree.V_CR_weights
V_SR_weights = datTree.V_SR_weights
A_CR_weights = datTree.A_CR_weights
A_SR_weights = datTree.A_SR_weights

err_dict = {
    'bkg_crtf' : datTree.bkg_crtf,
    'bkg_vrpred' : datTree.bkg_vrpred,
    'bkg_vr_normval' : datTree.bkg_vr_normval
}

print(err_dict)

# print("datTree.V_CR_kstest",datTree.V_CR_kstest)
# print("datTree.V_CR_prob_w",datTree.V_CR_prob_w)

if args.no_stats: err_dict = {key:1 for key in err_dict.keys()}

if args.plot:
   ratio = len(dat_mX_V_CRhs)/sum(datTree.V_CR_weights)
   # fig, axs = plt.subplots()
   fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
   fig.suptitle("Validation Control Region")
   axs, n_target, n_model = Ratio([dat_mX_V_CRhs, dat_mX_V_CRls], weights=[None, datTree.V_CR_weights*ratio], bins=mBins, axs=axs, labels=['Target', 'Model'], xlabel=r"M$_\mathrm{X}$ [GeV]", density=True, ratio_ylabel='Target/Model')
   fig.savefig(f"plots/model_VCR.pdf", bbox_inches='tight')

   ratio = len(dat_mX_V_SRhs)/sum(datTree.V_SR_weights)
   fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
   fig.suptitle("Validation Signal Region")
   axs, n_target, n_model = Ratio([dat_mX_V_SRhs, dat_mX_V_SRls], weights=[None, V_SR_weights*ratio], bins=mBins, axs=axs, labels=['Target', 'Model'], xlabel=r"M$_\mathrm{X}$ [GeV]", density=True, ratio_ylabel='Target/Model')
   fig.savefig(f"plots/model_VSR.pdf", bbox_inches='tight')

   ratio = len(dat_mX_A_CRhs)/sum(datTree.A_CR_weights)
   fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
   fig.suptitle("Analysis Control Region")
   axs, n_target, n_model = Ratio([dat_mX_A_CRhs, dat_mX_A_CRls], weights=[None, datTree.A_CR_weights*ratio], bins=mBins, axs=axs, labels=['Target', 'Model'], xlabel=r"M$_\mathrm{X}$ [GeV]", density=True, ratio_ylabel='Target/Model')
   fig.savefig(f"plots/model_ACR.pdf", bbox_inches='tight')

# n_VRtarget = Hist(dat_mX_V_CRhs, bins=mBins, ax=ax, label='Target')
# n_VRestimate = Hist(dat_mX_V_SRls, weights=VR_weights, bins=mBins, ax=ax, label='Estimation')

# print(kstest(n_VRtarget,n_VRestimate))
# sys.exit()

# ax.set_xlabel(r"M$_\mathrm{X}$ [GeV]")
# ax.set_ylabel("Events")
# ax.set_title("BDT Estimation of Data Yield in Validation Region")

fig, ax = plt.subplots()

n_dat_VRhs_obs = Hist(dat_mX_V_SRhs, bins=mBins, ax=ax, label='V_SR')

V_SR_weights = V_SR_weights * n_dat_VRhs_obs.sum() / V_SR_weights[(dat_mX_V_SRls >= mBins[0]) & (dat_mX_V_SRls < mBins[-1])].sum()
n_dat_VRhs_pred = Hist(dat_mX_V_SRls, weights=V_SR_weights, bins=mBins, ax=ax, label='V_SR')

n_dat_SRhs_pred = Hist(dat_mX_A_SRls, weights=A_SR_weights, bins=mBins, ax=ax, label='A_SR')

### ------------------------------------------------------------------------------------
## signal region estimate

ROOT.gROOT.SetBatch(True)

print(f"Estimated data = {n_dat_SRhs_pred.sum()}")
getROOTCanvas("data_exp", n_dat_SRhs_pred, f"{outDir}/a_sr/data")
print(f"ASR ROOT file saved to {outDir}/a_sr/data.root")

vr_exp_out = f"{outDir}/v_sr/data_exp"
getROOTCanvas("data_exp", n_dat_VRhs_pred, vr_exp_out)
print(f"VSR ROOT file saved to {vr_exp_out}")

vr_obs_out = f"{outDir}/v_sr/data_obs"
getROOTCanvas("data_obs", n_dat_VRhs_obs, vr_obs_out)
print(f"VSR ROOT file saved to {vr_obs_out}")



# print()
# print("PLEASE RUN THE FOLLOWING COMMAND:")
# print("---------------------------------")
# print("ssh srosenzw@cmslpc-sl7.fnal.gov 'cd /uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/; source env_standalone.sh; make -j 8; make; python3 runCombine.py'")
print("---------------------------------")
print("END PROCESS\n")