print("[INFO] .. starting program")

from argparse import ArgumentParser
from array import array
import ast
import awkward as ak
from configparser import ConfigParser
from hep_ml import reweight
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import re
import ROOT
import sys
import uproot
# https://pypi.org/project/uproot-tree-utils/
from uproot_tree_utils import clone_tree
# from utils.analysis import Signal, Particle
from utils.analysis import Signal, Particle
from utils.plotter import Hist
from utils.xsecUtils import lumiMap, xsecMap
import vector

### ------------------------------------------------------------------------------------
## Implement command line parser

print(".. parsing command line arguments")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--cfg',    dest='cfg',    help='config file', required=True)
parser.add_argument('--rectangular',    dest='rectangular',    help='', action='store_true', default=False)
parser.add_argument('--spherical',    dest='spherical',    help='', action='store_true', default=False)
# parser.add_argument('--signal', dest='signal', help='signal file')
# parser.add_argument('--data' ,  dest='data',   help='data file')
# parser.add_argument('--output', dest='output', help='output file', required=True)

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Implement config parser

print(".. parsing config file")

config = ConfigParser()
config.optionxform = str
config.read(args.cfg)

base = config['file']['base']
signal = config['file']['signal']
data = config['file']['data']
treename = config['file']['tree']
year = int(config['file']['year'])
pairing = config['pairing']['scheme']

if args.rectangular: region_type = 'rect'
elif args.spherical: region_type = 'sphere'

if 'dHHH' in pairing: pairing_type = 'dHHH'
elif 'mH' in pairing: pairing_type = 'mH'

# Assumptions here:
# Signal file is saved in an NMSSM folder with the following format:
# /NMSSM/NMSSM_XYH_YToHH_6b_MX_700_MY_400
# Most importantly, we are looking for the NMSSM_XYH string and the /ntuple.root string
start = re.search('NMSSM_XYH',signal).start()
end = re.search('/ntuple.root',signal).start()
outputFile = f"{signal[start:end]}_{pairing_type}_{region_type}"

if args.rectangular:
    maxSR = float(config['rectangular']['maxSR'])
    maxVR = float(config['rectangular']['maxVR'])
    maxCR = float(config['rectangular']['maxCR'])
    if maxCR == -1: maxCR = 9999
elif args.spherical:
    ARcenter =float(config['spherical']['ARcenter'])
    VRcenter =float(config['spherical']['VRcenter'])
    rInner   =float(config['spherical']['rInner'])
    rOuter   =float(config['spherical']['rOuter'])
else:
    raise AttributeError("No mass region definition!")

score = float(config['score']['threshold'])

# BDT parameters
Nestimators  = int(config['BDT']['Nestimators'])
learningRate = float(config['BDT']['learningRate'])
maxDepth     = int(config['BDT']['maxDepth'])
minLeaves    = int(config['BDT']['minLeaves'])
GBsubsample  = float(config['BDT']['GBsubsample'])
randomState  = int(config['BDT']['randomState'])

variables    = config['BDT']['variables'].split(", ")

### ------------------------------------------------------------------------------------
## build region masks

mH = 125 # GeV

indir = f"root://cmseos.fnal.gov/{base}/{pairing}/"

sigFileName = f"{indir}NMSSM/{signal}"
sigTree = Signal(sigFileName)
sigTree.rectangular_region()

datFileName = f"{indir}{data}"
datTree = Signal(datFileName)
datTree.rectangular_region()

print("N(data,SR) =",sum(sigTree.SRhs_mask))

### ------------------------------------------------------------------------------------
## train BDT

print(".. preparing inputs to train BDT")

def create_dict(mask):
    features = {}
    for var in variables:
        features[var] = datTree.get(var)[mask]
    return features

df_cr_ls = DataFrame(create_dict(datTree.CRls_mask))
df_cr_hs = DataFrame(create_dict(datTree.CRhs_mask))

df_vr_ls = DataFrame(create_dict(datTree.VRls_mask))
df_vr_hs = DataFrame(create_dict(datTree.VRhs_mask))

df_sr_ls = DataFrame(create_dict(datTree.SRls_mask))

TF = sum(datTree.CRhs_mask)/sum(datTree.CRls_mask)
print("TF",TF)

# Train BDT on CR data
# Use low-score CR to estimate high-score CR

ls_weights = np.ones(ak.sum(datTree.CRls_mask))*TF
hs_weights = np.ones(ak.sum([datTree.CRhs_mask]))

np.random.seed(randomState) #Fix any random seed using numpy arrays
print(".. calling reweight.GBReweighter")
reweighter_base = reweight.GBReweighter(
    n_estimators=Nestimators, 
    learning_rate=learningRate, 
    max_depth=maxDepth, 
    min_samples_leaf=minLeaves,
    gb_args={'subsample': GBsubsample})
print(".. calling reweight.FoldingReweighter")
reweighter = reweight.FoldingReweighter(reweighter_base, random_state=randomState, n_folds=2, verbose=False)
print(".. fitting BDT")
print(".. calling reweighter.fit")
reweighter.fit(df_cr_ls,df_cr_hs,ls_weights,hs_weights)

### ------------------------------------------------------------------------------------
## predict weights using BDT

print(".. predicting weights in validation region")
weights_pred = reweighter.predict_weights(df_vr_ls,np.ones(ak.sum(datTree.VRls_mask))*TF,lambda x: np.mean(x, axis=0))

nbins = 60
mBins = np.linspace(0,2000,nbins)

fig, ax = plt.subplots()
n_dat_VRls, _ = np.histogram(datTree.X_m[datTree.VRls_mask].to_numpy(), bins=mBins)
n_dat_VRls, _ = np.histogram(datTree.np('X_m')[datTree.VRls_mask], bins=mBins)

n_dat_VRls_transformed, e = Hist(datTree.X_m[datTree.VRls_mask], weights=weights_pred, bins=mBins, ax=ax, label='Estimation')
n_dat_VRhs, e = Hist(datTree.X_m[datTree.VRhs_mask], bins=mBins, ax=ax, label='Target')
ax.set_xlabel(r"$m_X$ [GeV]")
ax.set_ylabel("Events")
ax.set_title("BDT Estimation of Data Yield in Validation Region")
fig.savefig(f"combine/{outputFile}.pdf", bbox_inches='tight')

n_sig_SRhs, _ = np.histogram(sigTree.X_m[sigTree.SRhs_mask].to_numpy(), bins=mBins)
n_sig_SRhs = n_sig_SRhs * sigTree.scale

### ------------------------------------------------------------------------------------
## add branches and prepare to save

weights_pred = reweighter.predict_weights(df_sr_ls,np.ones(ak.sum(datTree.SRls_mask))*TF,lambda x: np.mean(x, axis=0))
n_dat_SRls, _ = np.histogram(datTree.np('X_m')[datTree.SRls_mask], bins=mBins)

n_dat_SRls_transformed, e = Hist(datTree.np('X_m')[datTree.SRls_mask], weights=weights_pred, bins=mBins, ax=ax, label='Estimation')

print("N(data,SR,est) =",n_dat_SRls_transformed.sum())

canvas = ROOT.TCanvas('c1','c1', 600, 600)
canvas.SetFrameLineWidth(3)

h_dat = ROOT.TH1D("h_dat",";m_{X} [GeV];Events",len(n_dat_SRls_transformed),array('d',list(e)))
h_sig = ROOT.TH1D("h_sig",";m_{X} [GeV];Events",len(n_sig_SRhs),array('d',list(e)))

for i,(bin_sig,bin_dat) in enumerate(zip(n_sig_SRhs, n_dat_SRls_transformed)):
    h_dat.SetBinContent(i+1, bin_dat)
    h_sig.SetBinContent(i+1, bin_sig)

h_sig.Draw("hist")
h_dat.Draw("hist same")

h_dat.SetLineColor(1)
h_sig.SetLineColor(ROOT.kBlue + 1)

canvas.Draw()

ROOT.gStyle.SetOptStat(0)
leg = ROOT.TLegend(0.4, 0.65, 0.88, 0.88)
leg.SetFillStyle(0)
leg.SetBorderSize(0)
leg.SetTextSize(0.03)
leg.AddEntry(h_dat, "Background", "l")
leg.AddEntry(h_sig, "Signal", "l")
leg.Draw()

canvas.Print(f"combine/{outputFile}.pdf)","Title:Signal Region");

fout = ROOT.TFile(f"combine/{outputFile}.root","recreate")
fout.cd()
h_dat.Write()
h_sig.Write()
fout.Close()