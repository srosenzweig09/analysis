"""This script will be a collection of calculations I find myself performing often."""

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm

def calculate_deltaR(eta1, eta2, phi1, phi2):
    deltaEta = eta1 - eta2
    deltaPhi = phi1 - phi2
    # Add and subtract 2pi to values below and above -pi and pi, respectively.
    # This limits the range of deltaPhi to (-pi, pi).
    deltaPhi = np.where(deltaPhi < -np.pi, deltaPhi + 2*np.pi, deltaPhi)
    deltaPhi = np.where(deltaPhi > +np.pi, deltaPhi - 2*np.pi, deltaPhi)
    deltaR = np.sqrt(deltaEta**2 + deltaPhi**2)
    
    return deltaR


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