from utils.analysis.datacard import DataCard
from utils.analysis.signal import Data
from utils.plotter import Hist
import numpy as np
import awkward as ak
import os, sys
from configparser import ConfigParser
import matplotlib.pyplot as plt

from array import array
import ROOT
ROOT.gROOT.SetBatch(True)

from utils.analysis.feyn import model_name
fsave = f'combine/feynnet/{model_name}'
HCsave = f'/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/datacards/feynnet/{model_name}'


def writeHist(h_title, n):
    ROOT_hist = ROOT.TH1D(h_title,";m_{X} [GeV];Events",nbins-1,array('d',list(data.mBins)))
    for i,(val) in enumerate(n):
        ROOT_hist.SetBinContent(i+1, val)

    ROOT_hist.Draw("hist")
    ROOT_hist.Write()

def read_cfg(cfg):
    config = ConfigParser()
    config.optionxform = str
    config.read(cfg)
    return config

fdata = "/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/JetHT_Data_UL/ntuple.root"

data = Data(fdata)
data.spherical_region()
data.train()
data.pull_plots()

n_nom_acr = sum(data.acr_mask)

fig, axs, n_nominal = data.sr_hist()
axs[0].set_title(data.sample)

nbins = len(data.mBins)

# fout = ROOT.TFile(f"plots/gnn/data_asr_model.root","recreate")
filename = f"{fsave}/model.root"
fout = ROOT.TFile(filename,"recreate")
fout.cd()

canvas = ROOT.TCanvas('c1','c1', 600, 600)
canvas.SetFrameLineWidth(3)
canvas.Draw()

h_title = f"model"
writeHist(h_title, n_nominal)

# ----------------------------------------------
# Modify CR

# Increase CR
config = read_cfg("config/bdt_params.cfg")
config['spherical']['CRedge'] = '40'
cr_up = Data('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/JetHT_Data_UL/ntuple.root', cfg=config)
cr_up.spherical_region()
n_cr_up_acr = sum(cr_up.acr_mask)
cr_up.train()
fig, ax, n_cr_up = cr_up.sr_hist()

# Decrease CR
config = read_cfg("config/bdt_params.cfg")
config['spherical']['CRedge'] = '60'
cr_down = Data('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/JetHT_Data_UL/ntuple.root', cfg=config)
cr_down.spherical_region()
n_cr_down_acr = sum(cr_down.acr_mask)
cr_down.train()
fig, ax, n_cr_down = cr_down.sr_hist()

# Decrease CR
config = read_cfg("config/bdt_params.cfg")
config['spherical']['CRedge'] = '45'
cr_upper_down5 = Data('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/JetHT_Data_UL/ntuple.root', cfg=config)
cr_upper_down5.spherical_region()
n_cr_upper_down5_acr = sum(cr_upper_down5.acr_mask)
cr_upper_down5.train()
fig, ax, n_cr_upper_down5 = cr_upper_down5.sr_hist()

# Decrease CR
config = read_cfg("config/bdt_params.cfg")
config['spherical']['SRedge'] = '30'
cr_lower = Data('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/JetHT_Data_UL/ntuple.root', cfg=config)
cr_lower.spherical_region()
n_cr_lower_acr = sum(cr_lower.acr_mask)
cr_lower.train()
asr_mask = cr_lower.deltaM <= 25
asr_ls_mask = asr_mask & cr_lower.ls_mask
df = cr_lower.get_df(asr_ls_mask, cr_lower.variables)
initial_weights = np.ones(ak.sum(asr_ls_mask))*cr_lower.AR_TF
weights = cr_lower.reweighter.predict_weights(df,initial_weights,lambda x: np.mean(x, axis=0))
n_cr_shift = Hist(cr_lower.X.m[asr_ls_mask], weights=weights, bins=cr_lower.mBins, ax=axs[0], label='asr', density=False)

print(n_nom_acr, n_cr_up_acr, n_cr_down_acr, n_cr_lower_acr, n_cr_upper_down5_acr)

fig, ax = plt.subplots()
bins = data.mBins
x = (bins[1:] + bins[:-1])/2
n = Hist(x, weights=n_nominal, bins=bins, ax=ax, label='Nominal (25 <= CR < 50)')
n = Hist(x, weights=n_cr_up, bins=bins, ax=ax, label='CR Outer Up (25 <= CR < 60)')
n = Hist(x, weights=n_cr_upper_down5, bins=bins, ax=ax, label='CR Outer Up (25 <= CR < 45)')
n = Hist(x, weights=n_cr_down, bins=bins, ax=ax, label='CR Outer Down (25 <= CR < 40)')
n = Hist(x, weights=n_cr_shift, bins=bins, ax=ax, label='CR Inner Up (30 <= CR < 50)')
ax.set_xlabel(r"$M_X$ [GeV]")
ax.set_ylabel("Events")
fig.savefig("plots/6_background_modeling/cr_shift.pdf")

diff1 = abs(n_cr_up-n_nominal)
diff2 = abs(n_cr_down-n_nominal)
diff3 = abs(n_cr_shift-n_nominal)
diff4 = abs(n_cr_upper_down5-n_nominal)
diff = np.maximum(diff1, diff2)
diff = np.maximum(diff, diff3)
diff = np.maximum(diff, diff4)

x = (data.mBins[1:] + data.mBins[:-1])/2
n_up = Hist(x, weights=n_nominal+diff, bins=data.mBins, ax=axs[0])
n_down = Hist(x, weights=n_nominal-diff, bins=data.mBins, ax=axs[0])

writeHist("model_CRShiftUp", n_up)
writeHist("model_CRShiftDown", n_down)

# ----------------------------------------------
# Modify avg b tag score threshold

# Increase hs btag region
config = read_cfg("config/bdt_params.cfg")
config['score']['threshold'] = '0.6'
btag_up = Data('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/JetHT_Data_UL/ntuple.root', cfg=config)
btag_up.spherical_region()
btag_up.train()
ls_mask = btag_up.btag_avg < 0.65 # ls
hs_mask = btag_up.btag_avg >= 0.65 # hs
asr_ls_mask = btag_up.asr_mask & ls_mask
acr_ls_mask = btag_up.acr_mask & ls_mask
acr_hs_mask = btag_up.acr_mask & hs_mask
TF = sum(acr_hs_mask)/sum(acr_ls_mask)
df = btag_up.get_df(asr_ls_mask, btag_up.variables)
initial_weights = np.ones(ak.sum(asr_ls_mask))*TF
weights = btag_up.reweighter.predict_weights(df,initial_weights,lambda x: np.mean(x, axis=0))
n_btag_up = Hist(btag_up.X.m[asr_ls_mask], weights=weights, bins=btag_up.mBins, ax=axs[0], label='asr', density=False)
fig, ax, N_btag_up = btag_up.sr_hist()
# writeHist("data_avgbtagUp", n_4b_model)

# Decrease hs btag region
config = read_cfg("config/bdt_params.cfg")
config['score']['threshold'] = '0.55'
btag_down = Data('/eos/uscms/store/user/srosenzw/sixb/ntuples/Summer2018UL/maxbtag_4b/JetHT_Data_UL/ntuple.root', cfg=config)
btag_down.spherical_region()
btag_down.train()
ls_mask = btag_down.btag_avg < 0.65 # ls
hs_mask = btag_down.btag_avg >= 0.65 # hs
asr_ls_mask = btag_down.asr_mask & ls_mask
acr_ls_mask = btag_down.acr_mask & ls_mask
acr_hs_mask = btag_down.acr_mask & hs_mask
TF = sum(acr_hs_mask)/sum(acr_ls_mask)
initial_weights = np.ones(ak.sum(asr_ls_mask))*TF
df = btag_down.get_df(asr_ls_mask, btag_down.variables)
weights = btag_down.reweighter.predict_weights(df,initial_weights,lambda x: np.mean(x, axis=0))
n_btag_down = Hist(btag_down.X.m[asr_ls_mask], weights=weights, bins=btag_down.mBins, ax=axs[0], label='asr', density=False)
fig, ax, N_btag_down = btag_down.sr_hist()
# fig, ax, n_4b_model = btag_down.sr_hist()
# writeHist("data_avgbtagDown", n_4b_model)

fig, ax = plt.subplots()
bins = data.mBins
x = (bins[1:] + bins[:-1])/2
n = Hist(x, weights=n_nominal, bins=bins, ax=ax, label='Nominal (Avg BTag > 0.65)')
n = Hist(x, weights=n_btag_up, bins=bins, ax=ax, label='Avg BTag Shift (Avg BTag > 0.6)')
n = Hist(x, weights=n_btag_down, bins=bins, ax=ax, label='Avg BTag Shift (Avg BTag > 0.55)')
ax.set_xlabel(r"$M_X$ [GeV]")
ax.set_ylabel("Events")
fig.savefig("plots/6_background_modeling/btag_shift.pdf")

fig, ax = plt.subplots()
bins = data.mBins
x = (bins[1:] + bins[:-1])/2
n = Hist(x, weights=n_nominal, bins=bins, ax=ax, label='Nominal (Avg BTag > 0.65)')
n = Hist(x, weights=N_btag_up, bins=bins, ax=ax, label='Avg BTag Shift (Avg BTag > 0.6)')
n = Hist(x, weights=N_btag_down, bins=bins, ax=ax, label='Avg BTag Shift (Avg BTag > 0.55)')
ax.set_xlabel(r"$M_X$ [GeV]")
ax.set_ylabel("Events")
fig.savefig("plots/6_background_modeling/btag_shift_uncorrected.pdf")

diff1 = abs(n_btag_up-n_nominal)
diff2 = abs(n_btag_down-n_nominal)
diff = np.maximum(diff1, diff2)

n_up = Hist(x, weights=n_nominal+diff, bins=data.mBins, ax=axs[0])
n_down = Hist(x, weights=n_nominal-diff, bins=data.mBins, ax=axs[0])

writeHist("model_AvgBTagUp", n_up)
writeHist("model_AvgBTagDown", n_down)

fout.Close()
# fout.Save()
# ROOT.gStyle.SetOptStat(0)

datacard = {
    'crtf' : round(data.crtf,2),
    'vr_stat_prec' : round(data.vr_stat_prec,2),
    'vr_yield_val' : round(data.vr_yield_val,2),
    'norm' : round(data.norm_err,2),
}

import json
with open(f"{fsave}/model.json", "w") as f:
    json.dump(datacard, f)