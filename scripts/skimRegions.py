print("---------------------------------")
print("STARTING PROGRAM")
print("---------------------------------")

from argparse import ArgumentParser
from array import array
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import ROOT
from scipy.stats import kstest
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# from sklearn.model_selection import StratifiedKFold
import sys
from tqdm import tqdm
from utils.analysis import Signal
from utils.analysis.data import createDataCard
from utils.fileUtils import mx_my_masses, combineDir
from utils.plotter import Hist

def getROOTCanvas(h_title, bin_values, outFile):
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

def getROOTCanvas_VR(h_title, bin_values, outFile):
   h1_title = h_title[0]
   h2_title = h_title[1]
   assert(h1_title == "h_obs")
   assert(h2_title == "h_est")
   h1_bin_vals = bin_values[0]
   h2_bin_vals = bin_values[1]
   # print("h1_bin_vals",h1_bin_vals)
   # print("h2_bin_vals",h2_bin_vals)
   print(f".. generating root hist for VR")
   canvas = ROOT.TCanvas('c1','c1', 600, 600)
   canvas.SetFrameLineWidth(3)

   h1 = ROOT.TH1D(h1_title,";m_{X} [GeV];Events",len(h1_bin_vals),array('d',list(mBins)))
   h2 = ROOT.TH1D(h2_title,";m_{X} [GeV];Events",len(h2_bin_vals),array('d',list(mBins)))

   for i,(vals1,vals2) in enumerate(zip(h1_bin_vals,h2_bin_vals)):
      h1.SetBinContent(i+1, vals1)
      h2.SetBinContent(i+1, vals2)

   h1.Draw("hist")
   h2.Draw("hist same")
   # h1.SetMarkerStyle(20)
   # h1.SetMarkerSize(1)
   h1.SetLineWidth(1)
   h2.SetLineWidth(2)
   h1.SetLineColor(1)
   h2.SetLineColor(38)

   print(h1.KolmogorovTest(h2))

## ------------------------------------------------------------------------------------
## Implement command line parser

print(".. parsing command line arguments")

parser = ArgumentParser(description='Command line parser of model options and tags')

# parser.add_argument('--cfg', dest='cfg', help='config file', default='')
parser.add_argument('--testing', dest='testing', action='store_true', default=False)
parser.add_argument('--all-signal', dest='allSignal', action='store_true', default=False)

# region shapes
parser.add_argument('--rectangular', dest='rectangular', help='', action='store_true', default=True)
parser.add_argument('--spherical', dest='spherical', help='', action='store_true', default=False)

# bdt parameters
parser.add_argument('--nestimators', dest='N')
parser.add_argument('--learningRate', dest='lr')
parser.add_argument('--maxDepth', dest='depth')
parser.add_argument('--minLeaves', dest='minLeaves')
parser.add_argument('--GBsubsample', dest='gbsub')
parser.add_argument('--randomState', dest='rand')

# stats parameters
parser.add_argument('--no-stats', dest='no_stats', action='store_true', default=False)
parser.add_argument('--no-MCstats', dest='MCstats', action='store_false', default=True)

args = parser.parse_args()

BDT_dict = {
    'Nestimators' : args.N,
    'learningRate' : args.lr,
    'maxDepth' : args.depth,
    'minLeaves' : args.minLeaves,
    'GBsubsample' : args.gbsub,
    'randomState' : args.rand
}

### ------------------------------------------------------------------------------------
## Implement config parser

print(".. parsing config file")

if args.spherical and args.rectangular: args.rectangular = False
if args.rectangular: 
   region_type = 'rect'
   cfg = 'config/rectConfig.cfg'
elif args.spherical: 
   region_type = 'sphere'
   cfg = 'config/sphereConfig.cfg'

config = ConfigParser()
config.optionxform = str
config.read(cfg)

overwrite_log = []
for hyper in BDT_dict:
    if BDT_dict[hyper] is not None:
        config['BDT'][hyper] = BDT_dict[hyper]
        overwrite_log.append(f"Overwriting [{hyper}={BDT_dict[hyper]}] from config file with command line argument.")
if len(overwrite_log) > 0: overwrite_log.insert(0, '')
for line in overwrite_log:
    print(overwrite_log)
print()

base = config['file']['base']
signal = config['file']['signal']
data = config['file']['data']
treename = config['file']['tree']
year = int(config['file']['year'])
pairing = config['pairing']['scheme']
pairing_type = pairing.split('_')[0]

minMX = int(config['plot']['minMX'])
maxMX = int(config['plot']['maxMX'])
nbins = int(config['plot']['nbins'])
mBins = np.linspace(minMX,maxMX,nbins)

score = float(config['score']['threshold'])

# BDT parameters
Nestimators  = int(config['BDT']['Nestimators'])
learningRate = float(config['BDT']['learningRate'])
maxDepth     = int(config['BDT']['maxDepth'])
minLeaves    = int(config['BDT']['minLeaves'])
GBsubsample  = float(config['BDT']['GBsubsample'])
randomState  = int(config['BDT']['randomState'])

variables    = config['BDT']['variables'].split(", ")

indir = f"root://cmseos.fnal.gov/{base}/{pairing}/"
outDir = f"combine/{pairing_type}/{region_type}"
datacardDir = f"{combineDir}datacards/{pairing_type}/{region_type}"
loc = f"{os.getcwd()}/{outDir}"

### ------------------------------------------------------------------------------------
## Obtain data regions

datFileName = f"{indir}{data}"
datTree = Signal(datFileName)

if args.rectangular:
    print("\n ---RECTANGULAR---")
    datTree.rectangular_region(config)
elif args.spherical:
    print("\n ----SPHERICAL----")
    datTree.spherical_region(config)
else:
    raise AttributeError("No mass region definition!")

dat_mX_VRls = datTree.dat_mX_VRls
dat_mX_VRhs = datTree.dat_mX_VRhs
dat_mX_SRls = datTree.dat_mX_SRls

### ------------------------------------------------------------------------------------
## train bdt and predict data SR weights

VR_weights, SR_weights = datTree.bdt_process(region_type, config)

err_dict = {
    'bkg_crtf' : datTree.bkg_crtf,
    'bkg_vrpred' : datTree.bkg_vrpred,
    'bkg_vr_normval' : datTree.bkg_vr_normval
}

if args.no_stats: err_dict = {key:1 for key in err_dict.keys()}

fig, ax = plt.subplots()
n_VRtarget = Hist(dat_mX_VRhs, bins=mBins, ax=ax, label='Target')
n_VRestimate = Hist(dat_mX_VRls, weights=VR_weights, bins=mBins, ax=ax, label='Estimation')

print(kstest(n_VRtarget,n_VRestimate))
# sys.exit()

ax.set_xlabel(r"$m_X$ [GeV]")
ax.set_ylabel("Events")
ax.set_title("BDT Estimation of Data Yield in Validation Region")
fig.savefig(f"{outDir}/data_validation.pdf", bbox_inches='tight')

getROOTCanvas_VR(["h_obs", "h_est"], [n_VRtarget,n_VRestimate], f"{outDir}/data_VR")

fig, ax = plt.subplots()
n_dat_SRhs_pred = Hist(dat_mX_SRls, weights=SR_weights, bins=mBins, ax=ax, label='Estimation')


### ------------------------------------------------------------------------------------
## Obtain MC signal counts

def get_sig_SRhs(sigFileName):
    sigTree = Signal(sigFileName)
    if args.rectangular:
        sig_mX_SRhs = sigTree.rectangular_region(config)
    elif args.spherical:
        sig_mX_SRhs = sigTree.spherical_region(config)
    n_sig_SRhs, _ = np.histogram(sig_mX_SRhs, bins=mBins)
    n_sig_SRhs = n_sig_SRhs * sigTree.scale
    return n_sig_SRhs

print("\n.. obtaining MC signal yields")
signal_nSR = []
signal_out = []
if args.allSignal:
    for mx,my in tqdm(mx_my_masses, ncols=50):
        signal = f'NMSSM_XYH_YToHH_6b_MX_{mx}_MY_{my}'
        sigFileName = f"{indir}NMSSM/{signal}/ntuple.root"
        n_sig_SRhs = get_sig_SRhs(sigFileName)
        signal_nSR.append(n_sig_SRhs)
        sigFileName = re.search('MX_.+/', sigFileName).group()[:-1]
        signal_out.append(sigFileName)
else:
    sigFileName = f"{indir}NMSSM/{signal}"
    n_sig_SRhs = get_sig_SRhs(sigFileName)
    mx_my_masses = []
    signal_nSR.append(n_sig_SRhs)
    sigFileName = re.search('MX_.+/', sigFileName).group()[:-1]
    signal_out.append(sigFileName)

### ------------------------------------------------------------------------------------
## signal region estimate

ROOT.gROOT.SetBatch(True)

print(f"Estimated data = {n_dat_SRhs_pred.sum()}")
getROOTCanvas("data", n_dat_SRhs_pred, f"{outDir}/data_{region_type}")
for sig_bin_vals,sig_name in tqdm(zip(signal_nSR, signal_out), ncols=50):
    hist_name = f"h_{sig_name}"
    getROOTCanvas(hist_name, sig_bin_vals, f"{outDir}/{sig_name}_nosyst")
   #  createDataCard(location=loc, sigROOT=sig_name, h_name=hist_name, err_dict=err_dict, outdir=datacardDir, no_bkg_stats=args.no_stats, MCstats=args.MCstats)

# print()
# print("PLEASE RUN THE FOLLOWING COMMAND:")
# print("---------------------------------")
# print("ssh srosenzw@cmslpc-sl7.fnal.gov 'cd /uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/; source env_standalone.sh; make -j 8; make; python3 runCombine.py'")
print("---------------------------------")
print("END PROCESS\n")