from utils.analysis.data import DataCard
from utils.analysis.signal import Data

import os
from array import array
import ROOT
ROOT.gROOT.SetBatch(True)

from utils.analysis.gnn import model_path
model_dir = model_path.split('/')[-3]
fsave = f'combine/feynnet/{model_dir}'

data = Data(fdata, config='config/bdt_params.cfg')
data.spherical_region(which_region)
data.train()

fig, axs, n_4b_model = data.sr_hist()
axs[0].set_title(data.sample)

fig, ax, n_4b_vsr = data.vr_hist()

nbins = len(data.mBins)

# fout = ROOT.TFile(f"plots/gnn/data_asr_model.root","recreate")
filename = f"{fsave}/data.root"
fout = ROOT.TFile(filename,"recreate")
fout.cd()

canvas = ROOT.TCanvas('c1','c1', 600, 600)
canvas.SetFrameLineWidth(3)
canvas.Draw()

h_title = f"data"
ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins-1,array('d',list(data.mBins)))
for i,(val) in enumerate(n_4b_model):
    ROOT_hist.SetBinContent(i+1, val)

ROOT_hist.Draw("hist")
ROOT_hist.Write()
fout.Close()
# fout.Save()
# ROOT.gStyle.SetOptStat(0)

datacard = {
    'crtf' : round(data.crtf,2),
    'vr_stat_prec' : round(data.vr_stat_prec,2),
    'vr_yield_val' : round(data.vr_yield_val,2)
}