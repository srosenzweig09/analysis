"""
I would like my plots to be consistent so I've saved what I can in the rcparams file but there are some things I cannot change. This script will help me keep those changes consistent.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from kinematics import change_cmap_bkg_to_white

def hist(ax, x, bins=100, label=None):
    return ax.hist(x, bins=bins, histtype='step', align='mid', label=label)

def hist2d(ax, x, y, xbins=100, ybins=100):
    cmap = change_cmap_bkg_to_white('rainbow')
    return ax.hist2d(x, y, bins=(xbins, ybins), norm=LogNorm(), cmap=cmap)