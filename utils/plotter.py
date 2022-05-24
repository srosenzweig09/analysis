"""
Author: Suzanne Rosenzweig

This script works as a wrapper for 1D and 2D histograms.
"""

from tkinter import Y
from . import *
from .useCMSstyle import *
from .varUtils import *

from awkward.highlevel import Array
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np

import matplotlib as mpl
mpl.rcParams['axes.formatter.limits'] = (-1,4)

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


def latexTitle(descriptor):
    mass_point = descriptor.split("/")[-2]
    mX = mass_point.split('_')[-3]
    mY = mass_point.split('_')[-1]
    return r"$m_X=$ " + mX + r" GeV, $m_Y=$ " + mY + " GeV"


def change_cmap_bkg_to_white(colormap, n=256):
    """Changes lowest value of colormap to white."""
    tmp_colors = cm.get_cmap(colormap, n)
    newcolors = tmp_colors(np.linspace(0, 1, n))
    # Define colors by [Red, Green, Blue, Alpha]
    white = np.array([1, 1, 1, 1])
    newcolors[0, :] = white  # Only change bins with 0 entries.
    newcmp = colors.ListedColormap(newcolors)
    return newcmp


def Hist2d(x, y, bins, log=False, density=False, **kwargs):
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
        kwargs.pop('cmap')
    else: cmap = change_cmap_bkg_to_white('rainbow')

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
        n, xe, ye, im = ax.hist2d(
            x, y, bins=bins, cmap=cmap)

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


def Hist(x, scale=1, legend_loc='best', weights=None, density=False, ax=None, patches=False, CMS=False, **kwargs):
    """
    This function is a wrapper for matplotlib.pyplot.hist that allows me to generate histograms quickly and consistently.
    It also helps deal with background trees, which are typically given as lists of samples.
    """

    if CMS: plt.style.use(CMS)

    bins = kwargs['bins']
    x_arr = x_bins(bins)

    # convert array to numpy if it is an awkward array
    if isinstance(x, Array): x = x.to_numpy()
    if isinstance(weights, Array): weights = weights.to_numpy()
    if ax is None: fig, ax = plt.subplots()
    if weights is None: weights = np.ones_like(x)

    # set default values for histogramming
    for k, v in plot_dict.items():
        if k not in kwargs:
            kwargs[k] = v

    if isinstance(x, list):
        # this handles background events, which are provided as a list of arrays
        n = np.zeros_like(x_arr)
        for bkg_kin, scale in zip(x, scale):
            n_temp, e = np.histogram(bkg_kin.to_numpy(), bins=bins)
            n += n_temp*scale
            if density: n = n/n.sum()
        n, _, im = ax.hist(x=x_bins(bins), weights=n, **kwargs)
        return n
    
    if density:
        if weights is None: weights = np.ones_like(x)
        n, _ = np.histogram(x, weights=weights, bins=bins)
        n, _, im = ax.hist(x_arr, weights=n/n.sum(), **kwargs)
        ax.yaxis.set_major_formatter(OOMFormatter(-2, "%2.0f"))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,4))
        ax.legend()
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

def Ratio(data, bins, labels, xlabel, axs=None, weights=[None, None], one=True, density=False, ratio_ylabel='Ratio'):
   data1, data2 = data
   label1, label2 = labels

   if weights[0] is None and weights[1] is not None: 
      weights1 = np.ones_like(data1)
      weights2 = weights[1]
   if weights[1] is None and weights[0] is not None: 
      weights1 = weights[0]
      weights2 = np.ones_like(data2)
   if weights[0] is None and weights[1] is None: 
      weights1 = np.ones_like(data1)
      weights2 = np.ones_like(data2)
   if isinstance(weights[0], float):
      weights1 = weights[0] * np.ones_like(data1)
   if isinstance(weights[1], float):
      weights2 = weights[1] * np.ones_like(data2)
   
   if axs is None: fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[4,1]})
   ax = axs[0]
   n_num = Hist(data1, bins=bins, ax=ax, label=label1, weights=weights1, density=density)
   n_den = Hist(data2, bins=bins, ax=ax, label=label2, weights=weights2, density=density)

   ax = axs[1]
   x = (bins[1:] + bins[:-1])/2
   n_ratio = n_num / n_den
   n_ratio = np.where(np.isnan(n_ratio), 0, n_ratio)
   n_ratio = np.where(np.isposinf(n_ratio), 0, n_ratio)
   # print(n_ratio)
   dist = np.abs(n_ratio - 1)
   dist = dist.sum()
   ax.set_xlabel(xlabel)
   print(dist)
   if one: ax.plot(x, np.ones_like(x), '--', color='gray')
   n_ratio = Hist(x, weights=n_ratio, bins=bins, ax=ax)
   ax.set_ylabel(ratio_ylabel)
   ax.set_ylim(0.75, 1.25)
   
   if axs is None: return fig, axs, dist
   else: return n_num, n_den, dist