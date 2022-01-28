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
import ROOT
import sys
import uproot
# https://pypi.org/project/uproot-tree-utils/
from uproot_tree_utils import clone_tree
from utils.plotter import Hist
import vector

### ------------------------------------------------------------------------------------
## Implement command line parser

print(".. parsing command line arguments")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--cfg',    dest='cfg',    help='config file', required=True)
parser.add_argument('--signal', dest='signal', help='signal file', required=True)
parser.add_argument('--data' ,  dest='data',   help='data file',   required=True)
# parser.add_argument('--output', dest='output', help='output file', required=True)

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Implement config parser

print(".. parsing config file")

config = ConfigParser()
config.optionxform = str
config.read(args.cfg)

treename = config['file']['tree']

maxSR = float(config['mass']['maxSR'])
maxVR = float(config['mass']['maxVR'])
maxCR = float(config['mass']['maxCR'])
if maxCR == -1: maxCR = 9999

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

def get_regions(filename, treename, is_signal):
    tree = uproot.open(f"{filename}:{treename}")

    HX_m = tree['HX_m'].array()
    HY1_m = tree['HY1_m'].array()
    HY2_m = tree['HY2_m'].array()

    HX_b1_btag  = tree['HX_b1_DeepJet'].array()
    HX_b2_btag  = tree['HX_b2_DeepJet'].array()
    HY1_b1_btag = tree['HY1_b1_DeepJet'].array()
    HY1_b2_btag = tree['HY1_b2_DeepJet'].array()
    HY2_b1_btag = tree['HY2_b1_DeepJet'].array()
    HY2_b2_btag = tree['HY2_b2_DeepJet'].array()

    btagavg = (HX_b1_btag + HX_b2_btag + HY1_b1_btag + HY1_b2_btag + HY2_b1_btag + HY2_b2_btag)/6

    low_btag_mask  = btagavg < score
    high_btag_mask = btagavg >= score

    SR_mask = (abs(HX_m - mH) <= maxSR) & (abs(HY1_m - mH) <= maxSR) & (abs(HY2_m - mH) <= maxSR)

    SR_hs_mask = SR_mask & high_btag_mask
    if is_signal: return tree, SR_hs_mask

    VR_mask = (abs(HX_m - mH) <= maxVR) & (abs(HY1_m - mH) <= maxVR) & (abs(HY2_m - mH) <= maxVR) & (abs(HX_m - mH) > maxSR) & (abs(HY1_m - mH) > maxSR) & (abs(HY2_m - mH) > maxSR)
    CR_mask = (abs(HX_m - mH) <= maxCR) & (abs(HY1_m - mH) <= maxCR) & (abs(HY2_m - mH) <= maxCR) & (abs(HX_m - mH) > maxVR) & (abs(HY1_m - mH) > maxVR) & (abs(HY2_m - mH) > maxVR)

    CR_ls_mask = CR_mask & low_btag_mask
    CR_hs_mask = CR_mask & high_btag_mask

    VR_ls_mask = VR_mask & low_btag_mask
    VR_hs_mask = VR_mask & high_btag_mask

    SR_ls_mask = SR_mask & low_btag_mask

    return tree, CR_ls_mask, CR_hs_mask, VR_ls_mask, VR_hs_mask, SR_ls_mask

sigtree, sig_SR_mask = get_regions(args.signal, treename, is_signal=True)
tree, CR_ls_mask, CR_hs_mask, VR_ls_mask, VR_hs_mask, SR_ls_mask = get_regions(args.data, treename, is_signal=False)

### ------------------------------------------------------------------------------------
## train BDT

print(".. preparing inputs to train BDT")

from utils.analysis import build_p4

HX_b1 = build_p4(
    tree['HX_b1_pt'].array(),
    tree['HX_b1_eta'].array(),
    tree['HX_b1_phi'].array(),
    tree['HX_b1_m'].array()
)
HX_b2 = build_p4(
    tree['HX_b2_pt'].array(),
    tree['HX_b2_eta'].array(),
    tree['HX_b2_phi'].array(),
    tree['HX_b2_m'].array()
)
HY1_b1 = build_p4(
    tree['HY1_b1_pt'].array(),
    tree['HY1_b1_eta'].array(),
    tree['HY1_b1_phi'].array(),
    tree['HY1_b1_m'].array()
)
HY1_b2 = build_p4(
    tree['HY1_b2_pt'].array(),
    tree['HY1_b2_eta'].array(),
    tree['HY1_b2_phi'].array(),
    tree['HY1_b2_m'].array()
)
HY2_b1 = build_p4(
    tree['HY2_b1_pt'].array(),
    tree['HY2_b1_eta'].array(),
    tree['HY2_b1_phi'].array(),
    tree['HY2_b1_m'].array()
)
HY2_b2 = build_p4(
    tree['HY2_b2_pt'].array(),
    tree['HY2_b2_eta'].array(),
    tree['HY2_b2_phi'].array(),
    tree['HY2_b2_m'].array()
)

HX_dr = HX_b1.deltaR(HX_b2)
HY1_dr = HY1_b1.deltaR(HY1_b2)
HY2_dr = HY2_b1.deltaR(HY2_b2)

vars = {'HX_dr':HX_dr, 'HY1_dr':HY1_dr, 'HY2_dr':HY2_dr}

def create_dict(mask):
    features = {}
    for var in variables:
        if var in tree.keys(): features[var] = tree[var].array()[mask]
        else: features[var] = vars[var][mask]
    return features

df_cr_ls = DataFrame(create_dict(CR_ls_mask))
df_cr_hs = DataFrame(create_dict(CR_hs_mask))

df_vr_ls = DataFrame(create_dict(VR_ls_mask))
df_vr_hs = DataFrame(create_dict(VR_hs_mask))

df_sr_ls = DataFrame(create_dict(SR_ls_mask))

TF = sum(CR_hs_mask)/sum(CR_ls_mask)
print("TF",TF)

# Train BDT on CR data
# Use low-score CR to estimate high-score CR

ls_weights = np.ones(ak.sum(CR_ls_mask))*TF
hs_weights = np.ones(ak.sum([CR_hs_mask]))

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

print(".. predicting weights in validation region")
weights_pred = reweighter.predict_weights(df_vr_ls,np.ones(ak.sum(VR_ls_mask))*TF,lambda x: np.mean(x, axis=0))

nbins = 60
mBins = np.linspace(0,2000,nbins)

fig, ax = plt.subplots()
n_dat_VRls, _ = np.histogram(tree['X_m'].array(library='np')[VR_ls_mask], bins=mBins)
print(len(tree['X_m'].array(library='np')[VR_ls_mask]))
print(len(weights_pred))
n_dat_VRls_transformed, e = Hist(tree['X_m'].array(library='np')[VR_ls_mask], weights=weights_pred, bins=mBins, ax=ax, label='Estimation')
n_dat_VRhs, e = Hist(tree['X_m'].array(library='np')[VR_hs_mask], bins=mBins, ax=ax, label='Target')
ax.set_xlabel(r"$m_X$ [GeV]")
ax.set_ylabel("Events")
ax.set_title("BDT Estimation of Data Yield in Validation Region")
fig.savefig("combine/data_BDT_VR.pdf", bbox_inches='tight')

n_sig_SRhs, _ = np.histogram(sigtree['X_m'].array(library='np')[sig_SR_mask], bins=mBins)

### ------------------------------------------------------------------------------------
## add branches and prepare to save

weights_pred = reweighter.predict_weights(df_sr_ls,np.ones(ak.sum(SR_ls_mask))*TF,lambda x: np.mean(x, axis=0))
n_dat_SRls, _ = np.histogram(tree['X_m'].array(library='np')[SR_ls_mask], bins=mBins)
print(len(tree['X_m'].array(library='np')[SR_ls_mask]))
print(len(weights_pred))
n_dat_SRls_transformed, e = Hist(tree['X_m'].array(library='np')[SR_ls_mask], weights=weights_pred, bins=mBins, ax=ax, label='Estimation')

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

# canvas.Print(f"plots/{sigTree.mXmY}_SR.pdf)","Title:Signal Region");
canvas.Print(f"combine/root_hist.pdf)","Title:Signal Region");

# fout = ROOT.TFile("mass_info/{sigTree.mXmY}_mX.root","recreate")
fout = ROOT.TFile("combine/shapes.root","recreate")
fout.cd()
h_dat.Write()
h_sig.Write()
fout.Close()

# print(".. appending region masks to tree")

# new_branches = {'CR_ls':CR_ls_mask, 'CR_hs':CR_hs_mask,'VR_ls':VR_ls_mask, 'VR_hs':VR_hs_mask, 'SR_ls':SR_ls_mask, 'SR_hs':SR_hs_mask}

# print(".. adding new branches")

# newtree = {}
# for k,v in itertools.chain(tree.items(),new_branches.items()):
#     try: newtree[k] = v.array()
#     except: newtree[k] = v

# NormWeightTree = {}
# for k,v in nwtree.items():
#     NormWeightTree[k] = v.array()

# print(".. saving to output")
# with uproot.recreate(args.output) as file:
#     file[treename] = newtree
#     file['NormWeightTree'] = NormWeightTree
#     print(file[treename].show())