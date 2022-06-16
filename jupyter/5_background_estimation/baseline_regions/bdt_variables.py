from configparser import ConfigParser
import matplotlib as mpl
mpl.rcParams['pgf.texsystem'] = 'pdflatex'
import matplotlib.pyplot as plt
from utils.plotter import Ratio, Hist
from utils.analysis import Signal
import numpy as np
from matplotlib.backends.backend_pgf import PdfPages
from pandas import DataFrame
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
   'HY2_costheta' : np.linspace(0, 1, nbins)
}

def producePulls(datTree, variables, pdf):
   V_SRhs_mask = datTree.V_SRhs_mask
   V_SRls_mask = datTree.V_SRls_mask


   pulls = []
   for var in variables:
      fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,8), gridspec_kw={'height_ratios':[4,1]})
      original = datTree.np(var)[V_SRls_mask]
      target = datTree.np(var)[V_SRhs_mask]
      norm = len(target)/sum(datTree.V_SR_weights)

      bins = var_bins[var]

      pull = Ratio([target, original],  bins=bins, axs=axs, labels=['Target','Model'], xlabel=var, weights=[None, datTree.V_SR_weights*norm], pull=True)

      # pdf.savefig()
      pulls.append(pull)
      plt.close()

   return np.array(pulls)


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

var_pulls = {}
var_avg = {}
for i,var in enumerate(variables):
   if i>1: break
   print(f"\n[INFO] {Fore.GREEN}{i}/{len(variables)} dropping variable {var}{Style.RESET_ALL}\n")
   tvars = variables.copy()
   tvars.remove(var)
   tvars = ", ".join(tvars) # updated training variables
   config['BDT']['variables'] = tvars
   datTree.spherical_region(config)
   datTree.bdt_process(region_type, config)

   # try: mkdir(f"plots/{var}")
   # except: pass

   pdf_name = f"plots/{var}_pull.pdf"
   with PdfPages(pdf_name) as pdf:
      pulls = producePulls(datTree, variables, pdf)
   var_pulls[var] = pulls
   var_avg[var] = pulls.mean()

df = DataFrame.from_dict(var_avg, orient='index')
print(df)

# df = DataFrame.from_dict(var_pulls, orient='index', columns=variables)
# print(df)
# # df.to_latex('plots/pull_table.tex')
# print(df.to_latex(columns=variables[:7], col_space=10, float_format="%.3f"))
# print(df.to_latex(columns=variables[7:14], col_space=10, float_format="%.3f"))
# print(df.to_latex(columns=variables[14:], col_space=10, float_format="%.3f"))
   # x = np.arange(len(variables))
   # fig, ax = plt.subplots()
   # n = Hist(x, weights=pulls, bins=np.arange(len(variables)+1), ax=ax)
   # ax.text(0.5, 0.1, f"avg pull = {round(n.mean(),3)}", fontsize=16, transform=ax.transAxes, ha='center')
   # ax.set_title(f"{var} removed from training", fontsize=20)
   # ax.set_ylabel("Pull")
   # ax.set_xticks(ticks=x, labels=variables, rotation=-45, fontsize=14, ha='left')
   # fig.savefig(f"plots/{var}_pulls.pdf")
   # plt.close()
