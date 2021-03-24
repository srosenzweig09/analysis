import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
import awkward0
import math

z_ME2 = 808 # cm

def calcStar(eta_gen, phi_gen, vx, vy, vz):
    """Calculate etastar and phistar for a gen particle."""
    
    if eta_gen.ndim > 1:
        is2darray = True
    else:
        is2darray = False

    r = (z_ME2 - abs(vz))/abs(np.sinh(eta_gen))
    xStar = vx + r * np.cos(phi_gen)
    yStar = vy + r * np.sin(phi_gen)
    rStar = np.sqrt(xStar*xStar + yStar*yStar)
    
    etaStar_gen = np.arcsinh(z_ME2/rStar) * (eta_gen/abs(eta_gen))
    phiStar_gen = []

    for x,y in zip(xStar, yStar):
        if x >= 0:
            phiStar_gen.append(np.arctan(y/x))
        elif y >= 0 and x < 0:
            phiStar_gen.append(np.pi + np.arctan(y/x))
        elif y <= 0 and x < 0:
            phiStar_gen.append(np.arctan(y/x) - np.pi)
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

def convert_emtf_pt(local_pt, is2darray=True, flatten=False):
    if is2darray:
        global_pt = []
        for i,evt in enumerate(local_pt):
            temp = local_pt[i].copy()
            temp = temp*0.5
            global_pt.append(temp)
    else:
        global_pt  = local_pt*0.5

    global_pt  = awkward0.fromiter(global_pt)

    if flatten:
        global_pt  = global_pt.flatten()

    return global_pt


def convert_emtf(local_pt,  local_eta, local_phi, is2darray=True, flatten=False):
    """Convert BDT-assigned EMTF variables."""

    if is2darray:
        global_pt = []
        for i,evt in enumerate(local_pt):
            temp = local_pt[i].copy()
            temp = temp*0.5
            global_pt.append(temp)

        global_eta = []
        for i,evt in enumerate(local_eta):
            temp = local_eta[i].copy()
            temp = temp*0.010875
            global_eta.append(temp)

        global_phi = []
        for i,evt in enumerate(local_phi):
            temp = local_phi[i] / 576 *2*np.pi - np.pi*15/180
            temp = np.where(temp > np.pi, temp-2*np.pi, temp)
            global_phi.append(temp)

    else:
        global_pt  = local_pt*0.5
        global_eta = local_eta*0.010875
        global_phi = local_phi / 576 *2*np.pi - np.pi*15/180
        global_phi = np.where(global_phi > np.pi, global_phi-2*np.pi, global_phi)

    global_pt  = awkward0.fromiter(global_pt)
    global_eta = awkward0.fromiter(global_eta)
    global_phi = awkward0.fromiter(global_phi)

    if flatten:
        global_pt  = global_pt.flatten()
        global_eta = global_eta.flatten()
        global_phi = global_phi.flatten()

    return global_pt, global_eta, global_phi