"""This script will be a collection of calculations I find myself performing often."""

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm

def change_cmap_bkg_to_white(colormap, n=256):
    """
    The lowest value of colormaps is not often white by default, which can help identify empty bins.
    This function will make the lowest value (typically zero) white.
    """
    
    tmp_colors = cm.get_cmap(colormap, n)
    newcolors = tmp_colors(np.linspace(0, 1, n))
    white = np.array([1, 1, 1, 1])    # White background (Red, Green, Blue, Alpha).
    newcolors[0, :] = white    # Only change bins with 0 entries.
    newcmp = colors.ListedColormap(newcolors)
    
    return newcmp