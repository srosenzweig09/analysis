print("[INFO] .. starting program")

from argparse import ArgumentParser
from array import array
import awkward as ak
from configparser import ConfigParser
from hep_ml import reweight
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import re
import ROOT
import sys
from utils.analysis import Signal
from utils.plotter import Hist

### ------------------------------------------------------------------------------------
## Implement command line parser

print(".. parsing command line arguments")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--cfg',    dest='cfg',    help='config file', required=True)
parser.add_argument('--rectangular',    dest='rectangular',    help='', action='store_true', default=False)
parser.add_argument('--spherical',    dest='spherical',    help='', action='store_true', default=False)
parser.add_argument('--dHHH',    dest='dHHH',    help='', action='store_true', default=False)
parser.add_argument('--mH',    dest='mH',    help='', action='store_true', default=False)

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

minMX = int(config['plot']['minMX'])
maxMX = int(config['plot']['maxMX'])
nbins = int(config['plot']['nbins'])
mBins = np.linspace(minMX,maxMX,nbins)

if args.rectangular: region_type = 'rect'
elif args.spherical: region_type = 'sphere'

if args.dHHH: pairing = 'dHHH_pairs'
elif args.mH: pairing = 'mH_pairs'
pairing_type = pairing.split('_')[0]

indir = f"root://cmseos.fnal.gov/{base}/{pairing}/"

sigFileName = f"{indir}NMSSM/{signal}"
sigTree = Signal(sigFileName)

datFileName = f"{indir}{data}"
datTree = Signal(datFileName)

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
    sigTree.rectangular_region(maxSR, maxVR, maxCR)
    datTree.rectangular_region(maxSR, maxVR, maxCR)
    dat_mX_VRls = datTree.np('X_m')[datTree.VRls_mask]
    dat_mX_VRhs = datTree.X_m[datTree.VRhs_mask]
    dat_mX_SRls = datTree.np('X_m')[datTree.SRls_mask]
    sig_mX_SRhs = sigTree.np('X_m')[sigTree.SRhs_mask]
elif args.spherical:
    ARcenter =float(config['spherical']['ARcenter'])
    VRcenter =float(config['spherical']['VRcenter'])
    rInner   =float(config['spherical']['rInner'])
    rOuter   =float(config['spherical']['rOuter'])
    sigTree.spherical_region(rInner, rOuter, ARcenter, VRcenter)
    datTree.spherical_region(rInner, rOuter, ARcenter, VRcenter)
    dat_mX_VRls = datTree.np('X_m')[datTree.V_SRls_mask]
    dat_mX_VRhs = datTree.np('X_m')[datTree.V_SRhs_mask]
    dat_mX_SRls = datTree.np('X_m')[datTree.A_SRls_mask]
    sig_mX_SRhs = sigTree.np('X_m')[sigTree.A_SRhs_mask]
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
## train bdt and predict weights

print(".. predicting weights in validation region")
VR_weights, SR_weights = datTree.bdt_process(region_type, config)

fig, ax = plt.subplots()
n_estimate, _ = Hist(dat_mX_VRls, weights=VR_weights, bins=mBins, ax=ax, label='Estimation')
n_target, _ = Hist(dat_mX_VRhs, bins=mBins, ax=ax, label='Target')
ax.set_xlabel(r"$m_X$ [GeV]")
ax.set_ylabel("Events")
ax.set_title("BDT Estimation of Data Yield in Validation Region")
fig.savefig(f"combine/{outputFile}_validation.pdf", bbox_inches='tight')

### ------------------------------------------------------------------------------------
## signal region estimate

n_sig_SRhs, _ = np.histogram(sig_mX_SRhs, bins=mBins)
n_sig_SRhs = n_sig_SRhs * sigTree.scale

n_dat_SRhs_estimate, e = Hist(dat_mX_SRls, weights=SR_weights, bins=mBins, ax=ax, label='Estimation')

print("N(data,SR,est) =",n_dat_SRhs_estimate.sum())

canvas = ROOT.TCanvas('c1','c1', 600, 600)
canvas.SetFrameLineWidth(3)

h_dat = ROOT.TH1D("h_dat",";m_{X} [GeV];Events",len(n_dat_SRhs_estimate),array('d',list(e)))
h_sig = ROOT.TH1D("h_sig",";m_{X} [GeV];Events",len(n_sig_SRhs),array('d',list(e)))

for i,(bin_sig,bin_dat) in enumerate(zip(n_sig_SRhs, n_dat_SRhs_estimate)):
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

