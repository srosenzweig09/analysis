"""
I would like my plots to be consistent so I've saved what I can in the rcparams file but there are some things I cannot change. This script will help me keep those changes consistent.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np

def change_cmap_bkg_to_white(colormap, n=256):
    """The lowest value of colormaps is not often white by default, which can help idenfity empty bins.
    This function will make the lowest value (typically zero) white."""
    
    tmp_colors = cm.get_cmap(colormap, n)
    newcolors = tmp_colors(np.linspace(0, 1, n))
    white = np.array([1, 1, 1, 1])    # White background (Red, Green, Blue, Alpha).
    newcolors[0, :] = white    # Only change bins with 0 entries.
    newcmp = colors.ListedColormap(newcolors)
    
    return newcmp

class Plotter:
    def __init__():
        pass

def hist(x, bins=100, label=None, weights=None, color=None, density=False, stacked=False, histtype='step', alpha=1.0, ax=False, xlim=None, ylim=None, xlabel=None, ylabel=None):
    try: bins = np.linspace(x.min(), x.max(), bins)
    except: pass
    if ~ax: fig, ax = plt.subplots()
    if weights is not None: x = (bins[1:] + bins[:-1])/2
    try: bins = np.linspace(x.min(), x.max(), bins)
    except: pass
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    N = sum(x)
    if weights is not None: N = sum(weights)
    textstr = f'Entries = {N}'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(0.8, 1.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
    return ax.hist(x, bins=bins, histtype=histtype, align='mid', label=label, weights=weights,  color=color, density=density, stacked=stacked, alpha=alpha)

def hist2d(ax, x, y, xbins=100, ybins=100, norm=LogNorm()):
    cmap = change_cmap_bkg_to_white('rainbow')
    return ax.hist2d(x, y, bins=(xbins, ybins), norm=LogNorm(), cmap=cmap)

def norm_hist(arr, bins=100):
    n, b = np.histogram(arr, bins=bins)
    x = (b[:-1] + b[1:]) / 2
    
    return n/n.max(), b, x