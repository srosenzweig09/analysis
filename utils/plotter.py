"""
Author: Suzanne Rosenzweig

This script works as a wrapper for 1D and 2D histograms.
"""

from . import *
from .useCMSstyle import *
plt.style.use(CMS)
from .varUtils import *

from awkward import flatten
from awkward.highlevel import Array
import matplotlib.colors as colors
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# import matplotlib as mpl
# mpl.rcParams['axes.formatter.limits'] = (-1,4)

from matplotlib.ticker import ScalarFormatter
class OOMFormatter(ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

# for informational purposes because I find myself looking this up often lol
legend_loc = {
    'best': 0,
    'upper right': 1,
    'upper left': 2,
    'lower left': 3,
    'lower right': 4,
    'right': 5,
    'center left': 6,
    'center right': 7,
    'lower center': 8,
    'upper center': 9,
    'center': 10}

def fig_ax_ratio():
   """Returns fig, axs for ratio plot."""
   return plt.subplots(nrows=2, ncols=1, figsize=(8,8), gridspec_kw={'height_ratios':[4,1]})

def latexTitle(mx,my):
   #  ind = -2
   #  if 'output' in descriptor: ind = -3
   #  mass_point = descriptor.split("/")[ind]
   #  mX = mass_point.split('_')[ind-1]
   #  mY = mass_point.split('_')[ind+1]
    return r"$M_X=$ " + str(mx) + r" GeV, $M_Y=$ " + str(my) + " GeV"


def change_cmap_bkg_to_white(colormap, arr=False, n=256):
    """Changes lowest value of colormap to white."""
    import matplotlib.cm as cm
    tmp_colors = cm.get_cmap(colormap, n)
    newcolors = tmp_colors(np.linspace(0, 1, n))
    # Define colors by [Red, Green, Blue, Alpha]
    white = np.array([1, 1, 1, 1])
    newcolors[0, :] = white  # Only change bins with 0 entries.
    newcmp = colors.ListedColormap(newcolors)
    if arr: return newcolors
    return newcmp

r_cmap = change_cmap_bkg_to_white('rainbow')
r_arr = change_cmap_bkg_to_white('rainbow', arr=True)

def Hist2d(x, y, bins, log=False, density=False, **kwargs):
   if 'ax' in kwargs.keys():
      ax = kwargs['ax']
      kwargs.pop('ax')
   else:
      fig, ax = plt.subplots(figsize=(12, 10))
      
   if 'cmap' in kwargs.keys():
      cmap = kwargs['cmap']
      kwargs.pop('cmap')
   else: cmap = r_cmap

   try: x = ak.flatten(x)
   except: pass
   try: y = ak.flatten(y)
   except: pass

   if isinstance(x, Array): x = x.to_numpy()
   if isinstance(y, Array): y = y.to_numpy()


   if log:
      n, xe, ye, im = ax.hist2d(
         x, y, bins=bins, norm=colors.LogNorm(), cmap=cmap, **kwargs)
   elif density:
      n, xe, ye, im = ax.hist2d(
         x, y, bins=bins, cmap=cmap, density=True)
         # x, y, bins=bins, density=True, cmap=cmap)
   else:
      n, xe, ye, im = ax.hist2d(x, y, bins=bins, cmap=cmap, **kwargs)

   if 'fig' in kwargs.keys():
      fig = kwargs['fig']
      fig.colorbar(im)
      fig.set_size_inches(10, 8)

   return n, xe, ye, im


def norm_hist(arr, bins=100):
    n, b = np.histogram(arr, bins=bins)
    x = (b[:-1] + b[1:]) / 2
    return n/n.max(), b, x


plot_dict = {
    'histtype': 'step',
    'align': 'mid',
    'linewidth': 2
}


def Hist(x, scale=1, legend_loc='best', weights=None, density=False, ax=None, patches=False, exp=0, dec=2, total=False, **kwargs):
    """
    This function is a wrapper for matplotlib.pyplot.hist that allows me to generate histograms quickly and consistently.
    It also helps deal with background trees, which are typically given as lists of samples.
    """

    bins = kwargs['bins']
    x_arr = x_bins(bins)

    # convert array to numpy if it is an awkward array
    if isinstance(x, Array): 
      try: x = flatten(x).to_numpy()
      except: x = x.to_numpy()

    if isinstance(x, list):
      for arr in x:
         if isinstance(x, Array): 
            try: x = flatten(x).to_numpy()
            except: x = x.to_numpy()

    if isinstance(weights, Array): weights = weights.to_numpy() 

    if ax is None: fig, ax = plt.subplots()
    
    if weights is None: 
      weights = np.ones_like(x)
    
    if isinstance(weights, float): 
      weights = np.ones_like(x) * weights

    # set default values for histogramming
    for k, v in plot_dict.items():
        if k not in kwargs:
            kwargs[k] = v

    if density:
      n, _ = np.histogram(x, bins=bins, weights=weights)
      n = np.where(np.isinf(n), 0, n)
      n = np.nan_to_num(n)
      n, _, im = ax.hist(x_arr, weights=n/n.sum(), **kwargs)
      ax.yaxis.set_major_formatter(OOMFormatter(exp, f"%2.{dec}f"))
   #   ax.ticklabel_format(axis='y', style='sci', scilimits=(0,4))
      if total: return n, n.sum()
      return n

    if scale != 1:
      n, _, im = ax.hist(x_arr, weights=weights*scale, **kwargs)
    if np.array_equal(weights, np.ones_like(x_arr)):
      n, _, im = ax.hist(x, **kwargs)
    else:
      n, _, im = ax.hist(x, weights=weights, **kwargs)

    if 'label' in kwargs.keys():
        ax.legend()
    
    if patches: return n, im
    if total: return n, n.sum()
    return n

def Ratio(data, bins, labels, xlabel, axs=None, weights=[None, None], density=False, ratio_ylabel='Ratio', broken=False, pull=False, data_norm=False, norm=None, total=False):

   data1, data2 = data
   label1, label2 = labels

   if isinstance(weights[0], float): weights1 = weights[0] * np.ones_like(data1)
   if isinstance(weights[1], float): weights2 = weights[1] * np.ones_like(data2)
   
   if weights[0] is None: weights1 = np.ones_like(data1)
   else: weights1 = weights[0]

   if weights[1] is None: weights2 = np.ones_like(data2)
   else: weights2 = weights[1]
   
   if axs is None: fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
   ax = axs[0]
   if norm is not None: 
      norm = np.nan_to_num(norm)
      # print("norm",norm)
      n_num, edges = np.histogram(data1, bins=bins, weights=weights1)
      x = x_bins(edges)
      if total: n_num, total = Hist(x, bins=bins, ax=ax, weights=n_num*norm, density=density, total=True)
      else: n_num = Hist(x, bins=bins, ax=ax, weights=n_num*norm, density=density)
   else:
      if total: n_num, total = Hist(data1, bins=bins, ax=ax, weights=weights1, density=density, total=True)
      else: n_num = Hist(data1, bins=bins, ax=ax, weights=weights1, density=density)
   # print(n_num)
   for i,edge in enumerate(bins[:-1]):
      try: weights = np.sqrt(np.square(weights1[(data1 >= edge) & (data1 < bins[i+1])]).sum())
      except: weights = np.sqrt(ak.sum(np.square(weights1[(data1 >= edge) & (data1 < bins[i+1])])))
      # print(weights, n_num[i])
      ax.fill_between([edge, bins[i+1]], n_num[i]-weights, n_num[i]+weights, color='C0', alpha=0.25)

   if broken:
      n_den,e = np.histogram(data2, bins=bins, weights=weights2)
      for i,(edge,val) in enumerate(zip(e[:-1], n_den)):
         ax.plot([edge, e[i+1]],[val,val], color='C1')
   else:
      if 'Data' in label2: 
         n_den, e = np.histogram(data2, bins=bins, weights=weights2)
         n_err = np.sqrt(n_den)
         if data_norm: 
            n_den = n_den / n_den.sum()
            n_err = np.zeros_like(n_den)
         # ax.plot(x_bins(bins), n_den, 'o', color='k')#, lw=0)
         ax.errorbar(x_bins(bins), n_den, n_err, marker='o', color='k', ls='none')
      else: n_den = Hist(data2, bins=bins, ax=ax, weights=weights2, density=density)

   handle1 = Line2D([0], [0], color='C0', lw=2, label=label1)
   handle2 = Line2D([0], [0], color='C1', lw=2, label=label2)
   if 'Data' in label2: handle2 = Line2D([], [], color='k', marker='o', markersize=6, lw=0, label=label2)
   ax.legend(handles=[handle1, handle2])

   ax = axs[1]
   x = (bins[1:] + bins[:-1])/2
   if pull:
      axs[0].set_ylabel('Events')
      diff = n_num - n_den
      # print(diff)
      pull = diff / np.std(diff)
      # std = np.sqrt(1/n_num + 1/n_den)
      # std = np.nan_to_num(std, nan=1)
      # print(std)
      # pull = diff / std
      # print(pull)
      # diff = diff / std
      n_pull = Hist(x, weights=pull, bins=bins, ax=ax)
      # for xval,n,s in zip(x, n_pull, std):
         # ax.plot([xval,xval], [n,s], color='k')
      pull = np.mean(np.abs(n_pull))
      ax.set_ylabel('Bin Difference', ha='center', fontsize=18)
      axs[0].text(x=0.9, y=0.5, s=f"pull = {round(np.abs(diff).mean(),1)}", transform=axs[0].transAxes, fontsize=16, ha='right')
   else:
      n_ratio = n_num / n_den
      n_ratio = n_den / n_num
      n_ratio = np.where(np.isnan(n_ratio), 0, n_ratio)
      n_ratio = np.where(np.isposinf(n_ratio), 0, n_ratio)
      # n_uncert_up = np.where(n_num != 0, (n_num + n_err) / n_num, 0)
      # n_uncert_down = np.where(n_num != 0, (n_num - n_err) / n_num, 0)

      # n_data_err_up = np.where(n_den != 0, (n_den + n_err)/n_den, 0)
      # n_data_err_down = np.where(n_den != 0, (n_den - n_err)/n_den, 0)
      # for i,n_nominal in enumerate(n_num):
      #    axs[1].fill_between([bins[i],bins[i+1]], n_uncert_down[i], n_uncert_up[i], color='C0', alpha=0.25)
      #    if n_ratio[i] < 1: y = -(1-n_ratio[i])
      #    else: y = n_ratio[i] - 1
      #    axs[1].plot([x[i], x[i]], [n_data_err_down[i]+y, n_data_err_up[i]+y], color='k', lw=2)
      ax.plot(x, np.ones_like(x), '--', color='gray')
      ax.scatter(x, n_ratio, color='k')
      ax.set_ylabel(ratio_ylabel)
      ax.set_ylim(0, 2)

   ax.set_xlabel(xlabel)
   
   if total: return n_num, n_den, n_ratio, total
   if pull: return n_num, n_den, n_pull, pull
   return n_num, n_den, n_ratio


def getRatio(numer, denom):
   ratio = numer / denom
   ratio = np.where(denom < 1e-6, 0, ratio)
   return ratio

def DataRatio(data, axs=None, fsave=None):
   if axs is None: fig, axs = plt.subplots(nrows=2,  gridspec_kw={'height_ratios':[4,1]})

   n_model_SR_hs = Hist(data.X_m[data.asr_ls_mask], weights=data.asr_weights, bins=data.mBins, ax=axs[0], label='asr', density=False, color='C2')
   weights2 = np.histogram(data.X_m[data.asr_ls_mask], weights=data.asr_weights**2, bins=data.mBins)[0]
   data.error = np.sqrt(weights2)

   sumw2 = []
   err = []
   for i,n_nominal in enumerate(n_model_SR_hs):#, model_uncertainty_up, model_uncertainty_down)):
      low_x = data.X_m[data.asr_ls_mask] > data.mBins[i]
      high_x = data.X_m[data.asr_ls_mask] <= data.mBins[i+1]
      weights = np.sum(data.asr_weights[low_x & high_x]**2)
      sumw2.append(weights)
      weights = np.sqrt(weights)
      err.append(weights)
      model_uncertainty_up = n_nominal + weights
      model_uncertainty_down = n_nominal - weights

      axs[0].fill_between([data.mBins[i], data.mBins[i+1]], model_uncertainty_down, model_uncertainty_up, color='C2', alpha=0.25)
      
      ratio_up = np.nan_to_num(model_uncertainty_up / n_nominal)
      ratio_down = np.nan_to_num(model_uncertainty_down / n_nominal)
      # print(ratio_down)
      # print(i, i+1, ratio_down, ratio_up)
      axs[1].fill_between([data.mBins[i], data.mBins[i+1]], ratio_down, ratio_up, color='C2', alpha=0.25)

   data.sumw2 = np.array((sumw2))
   data.err = np.array((err))

   model_nominal = Line2D([0], [0], color='C2', lw=2, label='Bkg Model')
   handles = [model_nominal, Patch(facecolor='C2', alpha=0.25, label='Bkg Uncertainty')]
   
   axs[0].legend(handles=handles)

   axs[0].set_ylabel('Events')
   axs[1].set_ylabel('Uncertainty')

   axs[1].plot([data.mBins[0], data.mBins[-1]], [1,1], color='gray', linestyle='--')
   axs[1].set_xlabel(r"$M_X$ [GeV]")

   if fsave is not None: fig.savefig(fsave)
   
   return axs, n_model_SR_hs

def NewRatio(numer, denom, bins, labels, axs=None, weights=[None, None], density=False, **kwargs):
   if axs is None:
      fig = plt.figure()
      gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[4, 1])
      ax1 = fig.add_subplot(gs[0])
      ax2 = fig.add_subplot(gs[1], sharex=ax1)
   elif isinstance(axs, tuple): ax1, ax2 = axs 

   x = (bins[1:] + bins[:-1])/2
   n_numer = np.histogram(numer, bins=bins, weights=weights[0])[0]
   n_denom = np.histogram(denom, bins=bins, weights=weights[1])[0]

   if density:
      n_numer = n_numer / n_numer.sum()
      n_denom = n_denom / n_denom.sum()
      n_numer = Hist(x, bins=bins, ax=ax1, weights=numer, label=labels[0])
      n_denom = Hist(x, bins=bins, ax=ax1, weights=denom, label=labels[1])
   else:
      n_numer = Hist(numer, bins=bins, ax=ax1, weights=weights[0], label=labels[0])
      n_denom = Hist(denom, bins=bins, ax=ax1, weights=weights[1], label=labels[1])

   ratio = getRatio(n_numer, n_denom)
   ax2.plot(x, np.ones_like(x), '--', color='gray')
   n_ratio = Hist(x, ax=ax2, weights=ratio, bins=bins)
   ax2.set_ylim(0,2)
   
   return axs
      



def RatioWithError(data, bins, labels, xlabel, axs=None, weights=[None, None], density=False, ratio_ylabel='Ratio', broken=False, pull=False, data_norm=False, norm=None, total=False):
   from matplotlib.lines import Line2D

   color1, color2 = None, None
   data1, data2 = data
   label1, label2 = labels


class Model:
  def __init__(self, h_sig, h_bkg, h_data=None):
    if isinstance(h_bkg, Stack): h_bkg = h_bkg.get_histo()

    self.h_sig = h_sig
    self.h_bkg = h_bkg
    self.h_data = h_bkg if h_data is None else h_data

    self.w = pyhf.simplemodels.uncorrelated_background(
      signal=h_sig.histo.tolist(), bkg=h_bkg.histo.tolist(), bkg_uncertainty=h_bkg.error.tolist()
    )
    self.data = self.h_data.histo.tolist()+self.w.config.auxdata

  def upperlimit(self, poi=np.linspace(0,5,11), level=0.05):
    self.h_sig.stats.obs_limit, self.h_sig.stats.exp_limits = pyhf.infer.intervals.upperlimit(
        self.data, self.w, poi, level=level,
    )



def model_ratio(target, prediction, weights, bins, ax_top, ax_bottom, lbf=False):
   x = (bins[:-1] + bins[1:]) / 2
   ax_bottom.plot([bins[0], bins[-1]], [1,1], '--', color='gray')
   ax_top.set_ylabel('Events')
   ax_bottom.set_ylabel('Data/Model')
   ax_bottom.set_xlabel(r'$M_X$ [GeV]')

   # plot distributions
   n_target = np.histogram(target, bins=bins)[0]
   n_pred = np.histogram(prediction, bins=bins, weights=weights)[0]

   ax_top.hist(x, weights=n_pred, bins=bins, histtype='step',lw=2,align='mid', color='C1', label='VSR Prediction')
   ax_top.scatter(x, n_target, color='black', label='VSR Data')
   
   ratio = np.nan_to_num(n_target / n_pred, nan=1)
   ax_bottom.scatter(x, ratio, color='k')
   ax_bottom.set_ylim(0,2)
   
   err_target = np.sqrt(n_target)
   # print(err)

   sumw2, err = [], []
   for i,(n_w,n,e) in enumerate(zip(n_pred, n_target, err_target)):
      low_x = prediction > bins[i]
      high_x = prediction <= bins[i+1]
      w = np.sum(weights[low_x & high_x]**2)
      sumw2.append(w)
      w = np.sqrt(w)
      err.append(w)

      model_uncertainty_up   = n_w + w
      model_uncertainty_down = n_w - w
      ratio_up = np.nan_to_num(model_uncertainty_up / n_w)
      ratio_down = np.nan_to_num(model_uncertainty_down / n_w)

      ax_top.fill_between([bins[i], bins[i+1]], model_uncertainty_down, model_uncertainty_up, color='C1', alpha=0.25)
      # The blue error bars in the ratio plot represent only the error in the model prediction
      ax_bottom.fill_between([bins[i], bins[i+1]], ratio_down, ratio_up, color='C1', alpha=0.25)

      target_uncert_up = n + e
      target_uncert_down = n - e
      ax_top.plot([x[i],x[i]],[target_uncert_down, target_uncert_up], color='k')

      ratio_err_up = target_uncert_up / n_w
      ratio_err_down = target_uncert_down / n_w
      ax_bottom.plot([x[i],x[i]],[ratio_err_up, ratio_err_down], color='k')


   if lbf:
      a, b = np.polyfit(x, ratio, 1)
      lbf = a * x + b
      ax_bottom.plot(x, lbf, ':', color='k')
   # print(err)

   return n_pred, n_target, ratio, sumw2

def gauss(x, H, A, x0, sigma):
   return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def plot_residuals(ratio, ax):
   rBins = np.arange(-0.5,0.50001,0.1)
   rBins = np.linspace(-0.5,0.5,12)
   rX = (rBins[:-1] + rBins[1:]) / 2
   h_ratio = Hist(1-ratio, bins=rBins, ax=ax, label='Ratio', align='mid', zorder=10)
   ax.set_xlabel('1 - Ratio')
   ax.set_ylabel('Bins')
   ax.set_xlim(-1,1)

   mean = (rX*h_ratio).sum()/h_ratio.sum()
   sigma = np.sqrt(sum(h_ratio * (rX-mean) ** 2) / sum(h_ratio))

   # X = np.linspace(0.5,1.5,200)
   X = np.linspace(-0.5,0.5,200)
   from scipy.optimize import curve_fit
   params, covar = curve_fit(gauss, rX, h_ratio, p0=[min(h_ratio), max(h_ratio), mean, sigma])
   H, A, x0, s = params
   std = np.around(np.sqrt(np.diag(covar)),3)

   y = gauss(X, H, A, x0, s)
   ax.plot(X, y, label='Gaussian Fit')
   ax.legend(loc=2)
   box = {
      'boxstyle' :'round',
      'fc' : 'white'
   }
   ax.text(.99,.99,f"mean = {round(x0,3)}+-{std[2]}\nstd = {round(s,3)}+-{std[3]}", transform=ax.transAxes, bbox=box, fontsize=18, va='top', ha='right')

   return 1+round(x0,3)

def plot_pulls(n_model, n_target, ax, err):
   bin_width = 0.5
   bins = np.arange(-3,3.0001,bin_width)
   diff = n_model - n_target
   e_diff = np.sqrt(n_target + err)
   pull = diff / e_diff
   n_pull, e = np.histogram(pull, bins=bins)
   pBins = np.arange(-4,4.0001,bin_width)
   pX = (pBins[:-1] + pBins[1:]) / 2
   N_pull = Hist(pull, bins=pBins, ax=ax, align='mid', label='Pull')

   x = (bins[:-1] + bins[1:]) / 2
   X = np.linspace(-2,2,200)
   
   ax.set_xlabel('Pull')
   ax.set_ylabel('Bins')

   mean = (x*n_pull).sum()/n_pull.sum()
   sigma = np.sqrt(sum(n_pull * (x - mean) ** 2) / sum(n_pull))

   from scipy.optimize import curve_fit
   params, covar = curve_fit(gauss, x, n_pull, p0=[min(n_pull), max(n_pull), mean, sigma])
   H, A, x0, s = params
   std = np.around(np.sqrt(np.diag(covar)),3)

   y = gauss(X, H, A, x0, s)
   ax.plot(X, y, label='Gaussian Fit')
   box = {
      'boxstyle' :'round',
      'fc' : 'white'
   }
   ax.text(.99,.99,f"mean = {round(x0,3)}+-{std[2]}\nstd = {round(s,3)}+-{std[3]}", transform=ax.transAxes, bbox=box, fontsize=18, va='top', ha='right')

   ax.legend(loc=2)