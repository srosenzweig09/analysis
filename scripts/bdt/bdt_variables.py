from configparser import ConfigParser
import matplotlib as mpl
mpl.rcParams['pgf.texsystem'] = 'pdflatex'
import matplotlib.pyplot as plt
from utils.plotter import Ratio, Hist
from utils.analysis import Signal
import numpy as np
from matplotlib.backends.backend_pgf import PdfPages
from pandas import DataFrame, Series
from colorama import Fore, Style
# from os import mkdir
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
   'X_m' : np.linspace(400, 1500, nbins)
}

var_labels = {
   'pt6bsum' : r"$\sum_{jets} p_T$ [GeV]",
   'dR6bmin' : r"${min}(\Delta R_{bb})$",
   'dEta6bmax' : r"${max}(\Delta\eta_{bb})$",
   'HX_pt' : r"$H_X \; p_T$ [GeV]",
   'HY1_pt' : r"$H_1 \; p_T$ [GeV]",
   'HY2_pt' : r"$H_2 \; p_T$ [GeV]",
   'HX_dr' : r"$H_X \; \Delta R_{bb}$",
   'HY1_dr' : r"$H_1 \; \Delta R_{bb}$",
   'HY2_dr' : r"$H_2 \; \Delta R_{bb}$",
   'HX_m' : r"$H_X \; m$ [GeV]",
   'HY1_m' : r"$H_1 \; m$ [GeV]",
   'HY2_m' : r"$H_2 \; m$ [GeV]",
   'HX_HY1_dEta' : r"$\Delta\eta(H_X, H_1)$",
   'HY1_HY2_dEta' : r"$\Delta\eta(H_1, H_2)$",
   'HY2_HX_dEta' : r"$\Delta\eta(H_2, H_X)$",
   'HX_HY1_dPhi' : r"$\Delta\phi(H_X, H_1)$",
   'HY1_HY2_dPhi' : r"$\Delta\phi(H_1, H_2)$",
   'HY2_HX_dPhi' : r"$\Delta\phi(H_2, H_X)$",
   'HX_costheta' : r"$\cos(\theta_{HX})$",
   'HY1_costheta' : r"$\cos(\theta_{H1})$",
   'HY2_costheta' : r"$\cos(\theta_{H2})$",
   'Y_HX_dR' : r"$\Delta R({Y}, H_X)$",
   'HY1_HY2_dR' : r"$\Delta R(H_1, H_2)$",
   'HX_HY1_dR' : r"$\Delta R(H_X, H_1)$",
   'HX_HY2_dR' : r"$\Delta R(H_X, H_2)$",
   'X_m' : r"$m_X$ [GeV]",
}

def producePulls(datTree, variables, pdf=False, **kwargs):
   V_SRhs_mask = datTree.V_SRhs_mask
   V_SRls_mask = datTree.V_SRls_mask
   V_CRhs_mask = datTree.V_CRhs_mask
   V_CRls_mask = datTree.V_CRls_mask

   pull_vars = variables.copy()
   pull_vars.append('X_m')

   pulls = []
   for var in pull_vars:
      fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,8), gridspec_kw={'height_ratios':[4,1]})
      original = datTree.np(var)[V_SRls_mask]
      target = datTree.np(var)[V_SRhs_mask]
      norm = sum(V_CRhs_mask)/sum(datTree.V_CR_weights)
      norm = 1

      bins = var_bins[var]
      n_target, n_model, n_pull, pull = Ratio([target, original],  bins=bins, axs=axs, labels=['Target','Model'], xlabel=var, weights=[None, datTree.V_SR_weights*norm], pull=True)

      if pdf:
         pdf.savefig()
      pulls.append(pull)
      plt.close()

   pulls = np.asarray(pulls)

   return pulls


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

pulls = producePulls(datTree, variables)
print(f"No variable removed: {np.around(pulls.mean(),3)}")
# print(f"pt6bsum removed: {np.around(pulls.mean(),3)}")
# print(pulls)


# var_pulls = {}
# var_avg = []
# for i,var in enumerate(variables, start=1):
#    # if i>1: break
#    # if var != 'HY1_pt': continue
#    print(f"\n[INFO] {Fore.GREEN}{i}/{len(variables)} dropping variable {var}{Style.RESET_ALL}\n")
#    tvars = variables.copy()
#    tvars.remove(var)
#    tvars = ", ".join(tvars) # updated training variables
#    config['BDT']['variables'] = tvars
#    datTree.spherical_region(config)
#    datTree.bdt_process(region_type, config)

#    # try: mkdir(f"plots/{var}")
#    # except: pass

#    pdf_name = f"plots/{var}_pull.pdf"
#    with PdfPages(pdf_name) as pdf:
#       pulls = producePulls(datTree, variables, pdf)
#    var_pulls[var] = np.around(pulls, 3)
#    var_avg.append(np.around(pulls, 3).mean())

#    x = np.arange(len(variables)+1)
#    fig, ax = plt.subplots()
#    n = Hist(x, weights=pulls, bins=np.arange(len(variables)+2), ax=ax)
#    ax.text(0.5, 0.1, f"avg pull = {round(n.mean(),3)}", fontsize=16, transform=ax.transAxes, ha='center')
#    ax.set_title(f"{var} removed from training", fontsize=20)
#    ax.set_ylabel("Average Pull per Variable")
#    ax.set_xticks(ticks=x, labels=variables+['X_m'], rotation=-45, fontsize=14, ha='left')
#    fig.savefig(f"plots/{var}_pulls.pdf")
#    plt.close()

# print(var_avg)
# print(len(var_avg))
# df = Series(var_avg, index=variables, name='Pull')
# df = df.sort_values(ascending=False)
# print(df.to_latex())
