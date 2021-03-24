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

def hist(ax, x, bins=100, label=None, weights=None, color=None, density=False):
    return ax.hist(x, bins=bins, histtype='step', align='mid', label=label, weights=weights,  color=color, density=density)

def hist2d(ax, x, y, xbins=100, ybins=100, norm=LogNorm()):
    cmap = change_cmap_bkg_to_white('rainbow')
    return ax.hist2d(x, y, bins=(xbins, ybins), norm=LogNorm(), cmap=cmap)