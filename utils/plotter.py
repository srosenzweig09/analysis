"""
Author: Suzanne Rosenzweig

This script works as a wrapper for 1D and 2D histograms.
"""

# from tkinter import Y
from . import *
from .useCMSstyle import *
from .varUtils import *

from awkward import flatten
from awkward.highlevel import Array
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np

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

file_location = '/uscms/home/srosenzw/nobackup/workarea/higgs/sixb_analysis/CMSSW_10_2_18/src/sixb/plots/'

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

def latexTitle(descriptor):
    ind = -2
    if 'output' in descriptor: ind = -3
    mass_point = descriptor.split("/")[ind]
    mX = mass_point.split('_')[ind-1]
    mY = mass_point.split('_')[ind+1]
    return r"$M_X=$ " + mX + r" GeV, $M_Y=$ " + mY + " GeV"


def change_cmap_bkg_to_white(colormap, arr=False, n=256):
    """Changes lowest value of colormap to white."""
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
   else:
      fig, ax = plt.subplots(figsize=(10, 8))
      
   if 'cmap' in kwargs.keys():
      cmap = kwargs['cmap']
      kwargs.pop('cmap')
   else: cmap = r_cmap

   if isinstance(x, Array): x = x.to_numpy()
   if isinstance(y, Array): y = y.to_numpy()

   if log:
      n, xe, ye, im = ax.hist2d(
         x, y, bins=bins, norm=colors.LogNorm(), cmap=cmap)
   elif density:
      n, xe, ye, im = ax.hist2d(
         x, y, bins=bins, cmap=cmap)
         # x, y, bins=bins, density=True, cmap=cmap)
   else:
      n, xe, ye, im = ax.hist2d(x, y, bins=bins, cmap=cmap)

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


def Hist(x, scale=1, legend_loc='best', weights=False, density=True, ax=None, patches=False, qcd=False, exp=0, dec=2, **kwargs):
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
    if not weights: 
      weights = np.ones_like(x)
   #  else: density = False
    if isinstance(weights, float): 
      weights = np.ones_like(x) * weights
      density = False

    # set default values for histogramming
    for k, v in plot_dict.items():
        if k not in kwargs:
            kwargs[k] = v

    if qcd:
        # this handles background events, which are provided as a list of arrays
        n = np.zeros_like(x_arr)
        for bkg_kin, scale in zip(x, scale):
            n_temp, e = np.histogram(bkg_kin.to_numpy(), bins=bins)
            n += n_temp*scale
            if density: n = n/n.sum()
        n, _, im = ax.hist(x=x_bins(bins), weights=n, **kwargs)
        return n
    
    if density:
        n, _ = np.histogram(x, weights=weights, bins=bins)
        n, _, im = ax.hist(x_arr, weights=n/n.sum(), **kwargs)
        ax.yaxis.set_major_formatter(OOMFormatter(exp, f"%2.{dec}f"))
      #   ax.ticklabel_format(axis='y', style='sci', scilimits=(0,4))
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
    return n

def Ratio(data, bins, labels, xlabel, axs=None, weights=[None, None], density=False, ratio_ylabel='Ratio', broken=False, pull=False):
   
   from matplotlib.lines import Line2D

   data1, data2 = data
   label1, label2 = labels

   if isinstance(weights[0], float):
      weights1 = weights[0] * np.ones_like(data1)
   if isinstance(weights[1], float):
      weights2 = weights[1] * np.ones_like(data2)
   
   if weights[0] is None: 
      weights1 = np.ones_like(data1)
   else:
      weights1 = weights[0]

   if weights[1] is None: 
      weights2 = np.ones_like(data2)
   else:
      weights2 = weights[1]
   
   if axs is None: fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
   ax = axs[0]
   n_num = Hist(data1, bins=bins, ax=ax, weights=weights1, density=density)
   if broken:
      n_den,e = np.histogram(data2, bins=bins, weights=weights2)
      for i,(edge,val) in enumerate(zip(e[:-1], n_den)):
         ax.plot([edge, e[i+1]],[val,val], color='C1')
   else:
      n_den = Hist(data2, bins=bins, ax=ax, weights=weights2, density=density)

   line1 = Line2D([0], [0], color='C0', lw=2, label=label1)
   line2 = Line2D([0], [0], color='C1', lw=2, label=label2)
   ax.legend(handles=[line1, line2])

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
      n_ratio = np.where(np.isnan(n_ratio), 0, n_ratio)
      n_ratio = np.where(np.isposinf(n_ratio), 0, n_ratio)
      # print(n_ratio)
      if one: ax.plot(x, np.ones_like(x), '--', color='gray')
      n_ratio = Hist(x, weights=n_ratio, bins=bins, ax=ax)
      ax.set_ylabel(ratio_ylabel)
      ax.set_ylim(0.5, 1.5)

   ax.set_xlabel(xlabel)
   
   if pull: return n_num, n_den, n_pull, pull
   return n_num, n_den

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



