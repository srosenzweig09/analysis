"""
Author: Suzanne Rosenzweig

This script works as a wrapper for 1D and 2D histograms.
"""

from . import *
from .useCMSstyle import *
from .varUtils import *

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
        fig, ax = plt.subplots(figsize=(10, 6))
        
    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
        kwargs.pop('cmap')
    else: cmap = change_cmap_bkg_to_white('rainbow')

    if not isinstance(x, np.ndarray):
        x = x.to_numpy()
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()

    if log:
        n, xe, ye, im = ax.hist2d(
            x, y, bins=bins, norm=colors.LogNorm(), cmap=cmap)
    elif density:
        n, xe, ye, im = ax.hist2d(
            x, y, bins=bins, norm=colors.Normalize(), cmap=cmap, vmin=0, vmax=1)
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


def Hist(x, scale=1, centers=False, **kwargs):
    """
    This function is a wrapper for matplotlib.pyplot.hist that allows me to generate histograms quickly and consistently.
    It also helps deal with background trees, which are typically given as lists of samples.
    """
    # if (scale == 1) & scale_warn: print("Setting scale=1. Was this intentional?")

    # make fig, ax if not provided as arguments
    if 'ax' not in kwargs:
        fig, ax = plt.subplots()
        return_ax = True
    else:
        ax = kwargs['ax']
        kwargs.pop('ax')
        return_ax = False

    # set default values for histogramming
    for k, v in plot_dict.items():
        if k not in kwargs:
            kwargs[k] = v

    bins = kwargs['bins']

    if scale == 0: x_arr = x_bins(bins)
    else: 
        x_arr = x
        scale = 1

    if isinstance(x, list):
        # this handles background events, which are provided as a list of arrays
        n = np.zeros_like(bins[:-1])
        for bkg_kin, scale in zip(x, scale):
            n_temp, e = np.histogram(bkg_kin.to_numpy(), bins=bins)
            n += n_temp*scale
        n, edges, im = ax.hist(x=x_bins(bins), weights=n, **kwargs)
    else:
        if 'density' in kwargs.keys():
            if kwargs['density'] == 1:
                kwargs.pop('density')
                try:
                    n, e = np.histogram(x, bins)
                    n = n * scale
                except:
                    n, e = np.histogram(x.to_numpy(), bins)
                    n = n * scale
                    if 'weights' in kwargs.keys():
                        kwargs.pop('weights')
                n, edges, im = ax.hist(x=x_arr, weights=n/n.sum(), **kwargs)
                ax.yaxis.set_major_formatter(OOMFormatter(-2, "%2.0f"))
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,4))
        else:
            if 'weights' in kwargs.keys() and 'evt_weights' not in kwargs.keys():
                weights = kwargs['weights']
                kwargs.pop('weights')
                n, edges, im = ax.hist(x, weights=weights*scale, **kwargs)
            elif 'weights' not in kwargs.keys() and 'evt_weights' not in kwargs.keys():
                # try:
                    # n, e = np.histogram(x, bins)
                # except:
                    # n, e = np.histogram(x.to_numpy(), bins)
                n, edges, im = ax.hist(x_arr, **kwargs)
            else:
                weights = kwargs['evt_weights']
                kwargs.pop('evt_weights')
                print(weights)
                n, edges, im = ax.hist(x, weights=weights, **kwargs)
                print(n)
    if 'label' in kwargs:
        ax.legend()
    if return_ax:
        if centers:
            return ax, n, edges, x_arr
        else:
            return ax, n, edges
    else:
        if centers:
            return n, edges, x_arr
        else:
            return n, edges
