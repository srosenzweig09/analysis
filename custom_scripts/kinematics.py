import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm


z_ME2 = 808 # cm

def calcStar(eta_gen, phi_gen, vx, vy, vz):
    """Calculate etastar and phistar for a gen particle."""
    
    r = (z_ME2 - abs(vz))/abs(np.sinh(eta_gen))
    xStar = vx + r * np.cos(phi_gen)
    yStar = vy + r * np.sin(phi_gen)
    rStar = np.sqrt(xStar*xStar + yStar*yStar)
    
    etaStar_gen = np.arcsinh(z_ME2/rStar) * (eta_gen/abs(eta_gen))

    phiStar_gen = []
    for i,evt in enumerate(phi_gen):
        temp = evt.copy()
        temp = np.where(xStar[i] >= 0, np.arctan(yStar[i]/xStar[i]), temp)
        temp = np.where(np.logical_and(xStar[i] < 0, yStar[i] >= 0), np.pi + np.arctan(yStar[i]/xStar[i]), temp)
        temp = np.where(np.logical_and(xStar[i] < 0, yStar[i] < 0), np.arctan(yStar[i]/xStar[i]) - np.pi, temp)
        phiStar_gen.append(temp)
    phiStar_gen = awkward.fromiter(phiStar_gen)    
    
    return etaStar_gen, phiStar_gen

def calc_d0(pt, phi, vx, vy, q, B=3.811):
    R = -pt/(q*0.003*B) # [cm]
    xc = vx - R*np.sin(phi)
    yc = vy + R*np.cos(phi)
    d0 = R - np.sign(R)*np.sqrt(xc**2 + yc**2)
    return d0

def calcDeltaR(eta1, eta2, phi1, phi2):
    deltaEta = eta1 - eta2
    deltaPhi = phi1 - phi2
    # Add and subtract 2np.pi to values below and above -np.pi and np.pi, respectively.
    # This limits the range of deltaPhi to (-np.pi, np.pi).
    deltaPhi = np.where(deltaPhi < -np.pi, deltaPhi + 2*np.pi, deltaPhi)
    deltaPhi = np.where(deltaPhi > +np.pi, deltaPhi - 2*np.pi, deltaPhi)
    deltaR = np.sqrt(deltaEta**2 + deltaPhi**2)
    
    return deltaR

def change_cmap_bkg_to_white(colormap, n=256):
    """
    The lowest value of colormaps is not often white by default, which can help identify empty bins.
    This function will make the lowest value (tynp.pically zero) white.
    """
    
    tmp_colors = cm.get_cmap(colormap, n)
    newcolors = tmp_colors(np.linspace(0, 1, n))
    white = np.array([1, 1, 1, 1])    # White background (Red, Green, Blue, Alpha).
    newcolors[0, :] = white    # Only change bins with 0 entries.
    newcmp = colors.ListedColormap(newcolors)
    
    return newcmp
