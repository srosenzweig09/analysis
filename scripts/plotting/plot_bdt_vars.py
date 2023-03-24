from configparser import ConfigParser
import matplotlib.pyplot as plt
from utils.plotter import Hist
from utils.analysis import Signal
import numpy as np
from utils.useCMSstyle import *
plt.style.use(lotsa_plots)
from matplotlib.gridspec import GridSpec
import sys

nbins = 30
var_bins = {
   'pt6bsum' : np.linspace(200, 1000, nbins),
   'dR6bmin' : np.linspace(0, 2, nbins),
   'dEta6bmax' : np.linspace(0.4, 2, nbins),
   'HX_pt' : np.linspace(0, 600, nbins),
   'HY1_pt' : np.linspace(0, 400, nbins),
   'HY2_pt' : np.linspace(0, 400, nbins),
   'HX_dr' : np.linspace(0, 4, nbins),
   'HY1_dr' : np.linspace(0, 5, nbins),
   'HY2_dr' : np.linspace(0, 5, nbins),
   'HX_m' : np.linspace(135, 215, nbins),
   'HY1_m' : np.linspace(135, 215, nbins),
   'HY2_m' : np.linspace(135, 215, nbins),
   'HX_HY1_dEta' : np.linspace(0, 6, nbins),
   'HY1_HY2_dEta' : np.linspace(0, 6, nbins),
   'HY2_HX_dEta' : np.linspace(0, 6, nbins),
   'HX_HY1_dPhi' : np.linspace(0, 3.2, nbins),
   'HY1_HY2_dPhi' : np.linspace(0, 3.2, nbins),
   'HY2_HX_dPhi' : np.linspace(0, 3.2, nbins),
   'HX_costheta' : np.linspace(0, 1, nbins),
   'HY1_costheta' : np.linspace(0,1 , nbins),
   'HY2_costheta' : np.linspace(0, 1, nbins),
   'Y_HX_dR' : np.linspace(0, 8, nbins),
   'HY1_HY2_dR' : np.linspace(0, 8, nbins),
   'HX_HY1_dR' : np.linspace(0, 8, nbins),
   'HX_HY2_dR' : np.linspace(0, 8, nbins),
   'X_m' : np.linspace(400,1500,nbins)
}

cfg = 'config/sphereConfig_bias.cfg'

config = ConfigParser()
config.optionxform = str
config.read(cfg)

base = config['file']['base']
data = config['file']['data']

indir = f"root://cmseos.fnal.gov/{base}"
datFileName = f"{indir}{data}"
datTree = Signal(datFileName)
region_type = 'sphere'

variables = config['BDT']['variables']
variables = variables.split(', ')

datTree.spherical_region(config)
datTree.bdt_process(region_type, config)

fontsize=16
height_ratios = np.tile([4,1],4)
legend_loc = [1, 1, 1, 1, 1, 1, 8, 8, 8, 1, 1, 2, 8, 2, 1, 1, 2, 2, 2, 1]

title = 'Validation Signal Region'

fig = plt.figure(constrained_layout=True, figsize=(30,20))
gs = GridSpec(8, 5, figure=fig, height_ratios=height_ratios)

fig.suptitle(t=f"{title}", fontsize=20)

i, j, k = 0, -2, 0
# if 'X_m' in variables: variables = variables[:-1]
variables.append('X_m')
for var in variables:
   col = i % 5
   if i % 5 == 0: j += 2
   plot_row = j
   ratio_row = j + 1

   ax1 = fig.add_subplot(gs[plot_row, col])
   ax2 = fig.add_subplot(gs[ratio_row, col], sharex=ax1)
   ax1.tick_params(axis='x', labelbottom=False)

   target = abs(datTree.get(var, 'np'))[datTree.V_SRhs_mask]
   original = abs(datTree.get(var, 'np'))[datTree.V_SRls_mask]
   ratio = datTree.V_CRhs_mask.sum() / datTree.V_CRls_mask.sum()
   # norm = datTree.V_CRhs_mask.sum() / datTree.V_CR_weights.sum()
   norm = 1

   if min(target) < min(original): b_min = min(target)
   else: b_min = min(original)
   if max(target) > max(original): b_max = max(target)
   else: b_max = max(original)
   if b_max >= 1000: b_max = b_max / 2

   bins = np.linspace(b_min,b_max,20)
   x = (bins[:-1] + bins[1:]) / 2

   n_target, e = np.histogram(target, bins=bins)
   n_target = Hist(x, weights=n_target, bins=bins, ax=ax1, label=f'Target')
   # n_scaled, e = np.histogram(original, bins=bins)
   # n_unweighted = Hist(original, weights=datTree.V_CR_weights*norm, bins=bins, ax=ax1, label=f'Model')
   n_unweighted = Hist(original, weights=ratio, bins=bins, ax=ax1, label=f'Constant Scale Factor')

   # n_weighted   = Hist(original, weights=datTree.CR_weights, bins=bins, ax=ax1, label=f'Weighted', )
   ax1.tick_params(axis='x', labelbottom=False)
   ax1.tick_params(axis='both', labelsize=fontsize)
   ax1.yaxis.offsetText.set_fontsize(fontsize)
   ax1.legend(fontsize=fontsize-4, loc=legend_loc[i])
   # ax1.legend(fontsize=fontsize-4, loc=(0.7,0.9), bbox_transform=ax2.transAxes, frameon=True, fancybox=True, framealpha=1)
   ax1.set_ylabel('Events', fontsize=fontsize)

   ratio_unweighted = n_target / n_unweighted
   # ratio_weighted   = n_target / n_weighted

   x = (bins[:-1] + bins[1:]) / 2
   # r_unweighted = Hist(x, bins=bins, weights=ratio_unweighted, ax=ax2, color='C1')
   Hist(x, weights=ratio_unweighted, bins=bins, color='k', ax=ax2)
   # ax2.plot(x, ratio_unweighted, color='C1')
   # r_weighted = Hist(x, bins=bins, weights=ratio_weighted, ax=ax2, color='C2')
   # ax2.plot(x, ratio_weighted, color='C2')
   ax2.plot(x, [1]*len(x), color='grey', linestyle='--')

   ax2.set_ylabel('Ratio', fontsize=fontsize)
   ax2.set_xlabel(var, fontsize=fontsize)
   ax2.tick_params(axis='both', labelsize=fontsize)
   ax2.yaxis.get_offset_text().set_fontsize(fontsize)
   ax2.set_ylim(0.5,1.5)
# _ = Hist(datTree.X_m[datTree.V_CRls_mask], weights=np.linspace(2,7,sum(datTree.V_CRls_mask)), bins=bins, ax=ax, , label='Wonky')

   i += 1

fig.savefig('plots/var_compare_tf_vsr.pdf', bbox_inches='tight')