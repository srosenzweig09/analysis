"""
pull = (n_target - n_model) / std(n_target - n_model)
1. fit pull distribution to a Gaussian
2. 
"""

from argparse import ArgumentParser
from configparser import ConfigParser
import matplotlib as mpl
mpl.rcParams['pgf.texsystem'] = 'pdflatex'
import matplotlib.pyplot as plt
from utils.plotter import Hist, Ratio
from utils.analysis import Signal
import numpy as np
from matplotlib.backends.backend_pgf import PdfPages
from pandas import DataFrame, Series
# from colorama import Fore, Style
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
# from os import mkdir
import math
import sys
from matplotlib.patches import Rectangle
from scipy.stats import chisquare
# from utils.useCMSstyle import *

parser = ArgumentParser()
parser.add_argument('--VCR', dest='VCR', action='store_true', default=0)
parser.add_argument('--unweighted', dest='unweighted', action='store_true', default=False)
parser.add_argument('--bins', dest='bins', type=int, default=False)
args = parser.parse_args()

nbins = 41
if args.bins: nbins = args.bins
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
   'X_m' : np.linspace(400, 1500, nbins),
   'HX_phi' : np.linspace(-np.pi, np.pi, nbins),
   'HY1_phi' : np.linspace(-np.pi, np.pi, nbins),
   'HY2_phi' : np.linspace(-np.pi, np.pi, nbins),
}

def gauss(x, H, A, x0, sigma):
   return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def kernel_func(X, h, xi):
   A = 1/(h*np.sqrt(2*np.pi))
   B = -0.5*(X-xi)**2/h**2
   return A*np.exp(B)

def producePulls(datTree, variables, pdf=False, region='VSR', **kwargs):
   if region == 'VSR':
      hs_mask = datTree.V_SRhs_mask
      ls_mask = datTree.V_SRls_mask
      weights = datTree.V_SR_weights
   elif region == 'VCR':
      hs_mask = datTree.V_CRhs_mask
      ls_mask = datTree.V_CRls_mask
      weights = datTree.V_CR_weights
   elif region == 'VSR_unweighted':
      hs_mask = datTree.V_SRhs_mask
      ls_mask = datTree.V_SRls_mask
      weights = np.ones_like(ls_mask)[ls_mask]

   pull_vars = variables.copy()
   # pull_vars.append('X_m')
   pull_vars = pull_vars + ['X_m', 'HX_phi', 'HY1_phi', 'HY2_phi']

   means = []
   sigmas = []
   ChiSq = []
   for var in pull_vars:
      print(var)

      original = datTree.np(var)[ls_mask]
      target = datTree.np(var)[hs_mask]

      # bins = np.linspace(200, 1000, 60)
      bmin = target.min()
      bmax = target.max()
      bins = np.linspace(bmin, bmax, nbins)
      x = (bins[:-1] + bins[1:]) / 2

      n_target, e = np.histogram(target, bins=bins)
      bmax = e[:-1][n_target > 0.05*n_target.max()][-1]
      bmin = e[:-1][n_target > 0.05*n_target.max()][0]
      # if var == 'X_m': bmin = 375
      bins = np.linspace(bmin, bmax, nbins)
      x = (bins[:-1] + bins[1:]) / 2
      
      n_target, e = np.histogram(target, bins=bins)

      weights = weights * n_target.sum() / weights[(original >= bmin) & (original < bmax)].sum()
      
      n_model, e  = np.histogram(original, bins=bins, weights=weights)
      # Xsq, p = chisquare(n_target, n_model)
      # print(Xsq)
      # if np.isinf(Xsq): Xsq = 999
      # Xsq, p = int(Xsq), round(p, 3)

      fig = plt.figure(figsize=(30,10))
      GS = GridSpec(1, 3, figure=fig, width_ratios=[4,5,5])
      gs1 = GridSpecFromSubplotSpec(2, 1, subplot_spec=GS[0], height_ratios=[3,1])
      gs2 = GridSpecFromSubplotSpec(1, 1, subplot_spec=GS[1])
      gs3 = GridSpecFromSubplotSpec(1, 1, subplot_spec=GS[2])

      ax1t = fig.add_subplot(gs1[0])
      ax1b = fig.add_subplot(gs1[1])
      ax2 = fig.add_subplot(gs2[0])
      ax3 = fig.add_subplot(gs3[0])

      # ax1t.set_title(f"Xsq statistic = {Xsq}, p-value = {p}")

      n_model = Hist(x, weights=n_model, bins=bins, ax=ax1t, zorder=9)
      _ = ax1t.plot([x,x],[n_target+np.sqrt(n_target), n_target-np.sqrt(n_target)], color='k', zorder=10)      
      scatter = ax1t.scatter(x, n_target, s=10, c='k', zorder=10)
      handles = [Rectangle([0,0],1,1,color='C0', fill=False, lw=2), Rectangle([0,0],1,1,color='C0', alpha=0.2)]
      labels=[f'Bkg Model ({int(n_model.sum())})', 'Bkg Uncertainty']
      handles.insert(0,scatter)
      labels.insert(0, f'Observed Data ({int(n_target.sum())})')
      ax1t.legend(handles=handles, labels=labels)

      bin_error_model = np.array(())
      for X,n,l in zip(x, n_model, bins):
         width = X - l
         r = X + width
         est_mask = (original >= X - width) & (original < X + width)
         est_err = np.sqrt(np.sum(weights[est_mask]**2))
         bin_error_model = np.append(bin_error_model, np.sum(weights[est_mask]**2))
         ax1t.fill_between([l, r], n-est_err, n+est_err, color='C0', alpha=0.2, zorder=0)

         # est_err_hi = np.where(n == 0, 1 + est_err, (n + est_err) / n)
         # est_err_lo =  np.where(n == 0, 1 - est_err, (n - est_err) / n)
         # ax1b.fill_between([l, r], est_err_lo, est_err_hi, color='C0', alpha=0.2)

      ax1t.axes.set_xticklabels([])
      ax1t.set_ylabel('Events')

      ratio = np.nan_to_num(n_model / n_target, 1)
      ratio = np.where(ratio > 10**5, 1, ratio)
      ratio = np.where(ratio == 0, 1, ratio)
      error = np.sqrt(bin_error_model/n_model**2 + 1/n_target)

      ax1b.plot([bins[0], bins[-1]], [1, 1], color='grey', linestyle='--')
      n_ratio, e = np.histogram(x, weights=ratio, bins=bins)
      _ = ax1b.scatter(x, n_ratio, s=10, c='k')
      ax1b.plot([x,x],[n_ratio+error,n_ratio-error], color='k')
      ax1b.set_xlabel(var)
      ax1b.set_ylabel('Obs/Exp')
      ax1b.set_ylim(0.5,1.5)

      bin_width = 0.5

      # RATIO plots on MIDDLE axis

      rBins = np.arange(0.5,1.50001,0.1)
      rX = (rBins[:-1] + rBins[1:]) / 2
      n_ratio = Hist(ratio, bins=rBins, ax=ax2, label='Ratio', align='mid', zorder=10)
      err_ratio = np.sqrt(n_ratio)
      # for n,err,xval in zip(n_ratio, err_ratio, rX):
         # ax2.plot([xval,xval],[n+err,n-err], color='C0')
      ax2.set_xlabel('Ratio')
      ax2.set_xlim(0,2)

      mean = (rX*n_ratio).sum()/n_ratio.sum()
      sigma = np.sqrt(sum(n_ratio * (rX-mean) ** 2) / sum(n_ratio))

      X = np.linspace(0.5,1.5,200)
      params, covar = curve_fit(gauss, rX, n_ratio, p0=[min(n_ratio), max(n_ratio), mean, sigma])
      H, A, x0, s = params
      std = np.around(np.sqrt(np.diag(covar)),3)

      y_r = gauss(rX, H, A, x0, s)

      # Xsq_r, p_r = chisquare(n_ratio[n_ratio > 0]/n_ratio.sum(),y_r[n_ratio > 0]/y_r[n_ratio > 0].sum())
      # print(Xsq_r)
      # ax2.scatter(rX, n_ratio)
      # ax2.scatter(rX, y_r)
      # print(n_ratio - y_r)
      # print((n_ratio - y_r)**2)
      # print((n_ratio - y_r)**2 / y_r)
      # print(((n_ratio/n_ratio.sum() - y_r/y_r.sum())**2 / y_r/y_r.sum()).sum())
      # sys.exit()

      # print("Ratio Chi Sq")
      # print(Xsq_r, p_r)
      # print()
      # Xsq_r, p_r = round(Xsq_r, 3), round(p_r, 3)
      # ax2.set_title(f"Xsq statistic = {Xsq_r}, p-value = {p_r}")
      y = gauss(X, H, A, x0, s)
      ax2.plot(X, y, label='Gaussian Fit')
      ax2.legend(loc=2)
      box = {
         'boxstyle' :'round',
         'fc' : 'white'
      }
      ax2.text(.99,.99,f"mean = {round(x0,3)}+-{std[2]}\nstd = {round(s,3)}+-{std[3]}", transform=ax2.transAxes, bbox=box, fontsize=18, va='top', ha='right')

      # PULL plots on RIGHT axis

      bins = np.arange(-3,3.0001,bin_width)
      diff = n_model - n_target
      e_diff = np.sqrt(n_target + bin_error_model)
      pull = diff / e_diff
      n_pull, e = np.histogram(pull, bins=bins)
      pBins = np.arange(-4,4.0001,bin_width)
      pX = (pBins[:-1] + pBins[1:]) / 2
      N_pull = Hist(pull, bins=pBins, ax=ax3, align='mid', label='Pull')
      err_pull = np.sqrt(N_pull)
      # for n,err,xval in zip(N_pull, err_pull, pX):
         # ax3.plot([xval,xval],[n+err,n-err], color='C0')

      x = (bins[:-1] + bins[1:]) / 2
      X = np.linspace(-2,2,200)
      
      ax3.set_xlabel('Pull')
      ax3.set_ylabel('Bins')

      mean = (x*n_pull).sum()/n_pull.sum()
      sigma = np.sqrt(sum(n_pull * (x - mean) ** 2) / sum(n_pull))

      params, covar = curve_fit(gauss, x, n_pull, p0=[min(n_pull), max(n_pull), mean, sigma])
      H, A, x0, s = params
      std = np.around(np.sqrt(np.diag(covar)),3)

      fit_pull = gauss(pX, H, A, x0, s)
      # Xsq_p, p_p = chisquare(N_pull/N_pull.sum(), fit_pull/fit_pull.sum())
      # ax3.scatter(pX, N_pull)
      # ax3.scatter(pX, fit_pull)
      # plt.show()
      # sys.exit()
      # print("Pull Chi Sq")
      # print(Xsq_p, p_p)
      # print()
      # Xsq_p, p_p = round(Xsq_p, 3), round(p_p, 3)
      # ax3.set_title(f"Xsq statistic = {Xsq_p}, p-value = {p_p}")

      y = gauss(X, H, A, x0, s)
      ax3.plot(X, y, label='Gaussian Fit')
      box = {
         'boxstyle' :'round',
         'fc' : 'white'
      }
      ax3.text(.99,.99,f"mean = {round(x0,3)}+-{std[2]}\nstd = {round(s,3)}+-{std[3]}", transform=ax3.transAxes, bbox=box, fontsize=18, va='top', ha='right')

      ax3.legend(loc=2)
      plt.tight_layout()

      # plt.show()
      # sys.exit()

      if pdf:
         print(".. saving fig\n")
         pdf.savefig(fig)
      plt.close()

      # sys.exit()

      means.append(round(x0,3))
      sigmas.append(round(s,3))
      # ChiSq.append([Xsq, p])
      ChiSq.append([0, 0])
      # break

   return means, sigmas, ChiSq


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
print(variables,"\n")

datTree.spherical_region(config)
datTree.bdt_process(region_type, config)

filename = 'all_vars_pull'
region = 'VSR'
if args.VCR:
   filename = 'all_vars_pull_VCR'
   region = 'VCR'
elif args.unweighted:
   filename = 'all_vars_pull_unweighted'
   region = 'VSR_unweighted'
pdf_name = f"plots/6_background_modeling/pull_plots/{filename}.pdf"
with PdfPages(pdf_name) as pdf:
   means, sigmas, X = producePulls(datTree, variables, pdf, region=region)#, region='VSR_unweighted')


extra_vars = ['X_m', 'HX_phi', 'HY1_phi', 'HY2_phi']
mean_df = Series(means, index=variables+extra_vars)
sigma_df = Series(sigmas, index=variables+extra_vars)
X = np.row_stack(X)
X_df = DataFrame(X, index=variables+extra_vars, columns=['Xsq', 'p'])

print(mean_df)
print(sigma_df)
print(X_df)

# mean_dict = {}
# sigma_dict = {}
# for i,var in enumerate(variables, start=1):
#    # if i>0: break
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

#    pdf_name = f"plots/all_vars_pull.pdf"
#    with PdfPages(pdf_name) as pdf:
#       pulls = producePulls(datTree, variables, pdf)
#       means, sigmas = producePulls(datTree, variables)
#       mean_dict[var] = means
#       sigma_dict[var] = sigmas
      
#       # var_avg.append([x0, s])


#       # x = np.arange(len(variables)+1)
#       # fig, ax = plt.subplots()
#       # n = Hist(x, weights=pulls, bins=np.arange(len(variables)+2), ax=ax)
#       # ax.text(0.5, 0.1, f"avg pull = {round(n.mean(),3)}", fontsize=16, transform=ax.transAxes, ha='center')
#       # ax.set_title(f"{var} removed from training", fontsize=20)
#       # ax.set_ylabel("Average Pull per Variable")
#       # ax.set_xticks(ticks=x, labels=variables+['X_m'], rotation=-45, fontsize=14, ha='left')
#       # fig.savefig(f"plots/{var}_pulls.pdf")
#       # plt.close()

# # print(mean_dict)
# # print(sigma_dict)

# mean_df = DataFrame.from_dict(mean_dict, orient='index', columns=variables+['X_m'])
# sigma_df = DataFrame.from_dict(sigma_dict, orient='index', columns=variables+['X_m'])
# print(mean_df)
# print(sigma_df)
# # print(df.to_latex())
# mean_df.to_csv("var_pull_means.csv")
# sigma_df.to_csv("var_pull_sigmas.csv")