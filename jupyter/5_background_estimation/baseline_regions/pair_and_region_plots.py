import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import sys

from utils.fileUtils import FileLocations
from utils.analysis import Signal
from utils.plotter import Hist2d

# -------------------------------------------------------------------
# Open and load files
print(".. opening and loading files")

Sum18UL_dHHH = FileLocations('Summer2018UL','dHHH_pairs',False)
Sum18UL_mH   = FileLocations('Summer2018UL','mH_pairs',  False)

data_dHHH = Sum18UL_dHHH.data
data_mH   = Sum18UL_mH.data

mx, my = 700, 400
MX700_MY400_dHHH = Sum18UL_dHHH.get_NMSSM(mx,my)
MX700_MY400_mH   = Sum18UL_mH.get_NMSSM(mx,my)

print(".. data files")
print(data_dHHH)
print(data_mH)
data_dHHH_tree = Signal(data_dHHH)
data_mH_tree = Signal(data_mH)

print(".. signal files")
print(MX700_MY400_dHHH)
print(MX700_MY400_mH)
sig_dHHH_tree = Signal(MX700_MY400_dHHH)
sig_mH_tree = Signal(MX700_MY400_mH)

# -------------------------------------------------------------------
# Define regions
print(".. defining regions")

SR_edge = 25
VR_edge = 60

score_edge = 0.66

mHbins = np.linspace(0,500,100)
score_bins = np.linspace(0.35,1.01,100)

# -------------------------------------------------------------------
# Define functions

def rectangularRegions(tree):

    avg_btag = tree.btag_avg
    DeltaM = tree.DeltaM

    hi_score_mask = avg_btag >= score_edge
    lo_score_mask = avg_btag < score_edge

    SR = ak.all(DeltaM <= SR_edge, axis=1) # SR
    VR = ak.all(DeltaM > SR_edge, axis=1) & ak.all(DeltaM <= VR_edge, axis=1) # VR
    CR = ak.all(DeltaM > VR_edge, axis=1) # CR

    CR_ls = CR & lo_score_mask
    CR_hs = CR & hi_score_mask
    VR_ls = VR & lo_score_mask
    VR_hs = VR & hi_score_mask
    SR_ls = SR & lo_score_mask
    SR_hs = SR & hi_score_mask

    regionDict = {
        'SR':{'all':SR, 'ls':SR_ls, 'hs':SR_hs},
        'VR':{'all':VR, 'ls':VR_ls, 'hs':VR_hs},
        'CR':{'all':CR, 'ls':CR_ls, 'hs':CR_hs}
    }

    return regionDict

def sphericalRegions(tree):

    avg_btag = tree.btag_avg
    hi_score_mask = avg_btag >= score_edge
    lo_score_mask = avg_btag < score_edge

    # Analysis Region
    DeltaM_A = tree.DeltaM

    AR_rad = DeltaM_A * DeltaM_A
    AR_rad = AR_rad.sum(axis=1)
    AR_rad = np.sqrt(AR_rad)

    AR = AR_rad <= 50 # VR
    AR_SR = AR_rad <= SR_edge # SR
    AR_CR = (AR_rad > SR_edge) & (AR_rad <= 50) # VR

    AR_CR_hi = AR_CR & hi_score_mask
    AR_CR_lo = AR_CR & lo_score_mask
    AR_SR_hi = AR_SR & hi_score_mask
    AR_SR_lo = AR_SR & lo_score_mask

    # Validation Region
    DeltaM_V = tree.DeltaM_V

    VR_rad = DeltaM_V * DeltaM_V
    VR_rad = VR_rad.sum(axis=1)
    VR_rad = np.sqrt(VR_rad)

    VR = VR_rad <= 50 # VR
    VR_SR = VR_rad <= SR_edge # SR
    VR_CR = (VR_rad > SR_edge) & (VR_rad <= 50) # VR

    VR_CR_hi = VR_CR & hi_score_mask
    VR_CR_lo = VR_CR & lo_score_mask
    VR_SR_hi = VR_SR & hi_score_mask
    VR_SR_lo = VR_SR & lo_score_mask

    regionDict = {
        'AR':{ 
            'all':AR,
            'CR':{'all':AR_CR, 'ls':AR_CR_lo, 'hs':AR_CR_hi},
            'SR':{'all':AR_SR, 'ls':AR_SR_lo, 'hs':AR_SR_hi}
            },
        'VR':{
            'all':VR, 
            'CR':{'all':VR_CR, 'ls':VR_CR_lo, 'hs':VR_CR_hi},
            'SR':{'all':VR_SR, 'ls':VR_SR_lo, 'hs':VR_SR_hi}
            },
    }

    return regionDict

def plotRegions(tree, regionType, pairing, isSignal=False):

    # mCand = tree.mCand
    m1, m2, m3 = tree.HX_m, tree.HY1_m, tree.HY2_m
    mCand = [m1, m2, m3]
    mCand2 = [m2, m3, m1]

    avg_btag = tree.btag_avg
    hi_score_mask = avg_btag >= score_edge
    lo_score_mask = avg_btag < score_edge

    if regionType == 'rect':
        print(".. evaluating RECTANGULAR regions")
        regionDict = rectangularRegions(tree)
        print(f"CR    = {ak.sum(regionDict['CR']['all'])}")
        print(f"CR ls = {ak.sum(regionDict['CR']['ls'])}")
        print(f"CR hs = {ak.sum(regionDict['CR']['hs'])}")
        print(f"VR    = {ak.sum(regionDict['VR']['all'])}")
        print(f"VR ls = {ak.sum(regionDict['VR']['ls'])}")
        print(f"VR hs = {ak.sum(regionDict['VR']['hs'])}")

        if isSignal: 
            print(f"SR    = {ak.sum(regionDict['SR']['all'])}")
            print(f"SR ls = {ak.sum(regionDict['SR']['ls'])}")
            print(f"SR hs = {ak.sum(regionDict['SR']['hs'])}")
            mask = np.asarray(np.ones_like(m1), dtype=bool)
            title = 'Signal Events'
            saveas = f'NMSSM_{my}_GeV_{my}_GeV'
        else: 
            mask = ~regionDict['SR']['all']
            title = 'Data Events (Blinded)'
            saveas = f'data'
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30,7))

        for i,(mass,ax) in enumerate(zip(mCand, axs.flatten())):
            n,xe,ye,im = Hist2d(mass[mask], avg_btag[mask], bins=(mHbins, score_bins), ax=ax)
            fig.colorbar(im, ax=ax)
            ax.set_ylabel('Average b-tag score')
            ax.set_xlabel(f'$m_{i+1}$ [GeV]')
        fig.suptitle(title)
        fig.savefig(f'{saveas}_{regionType}_{pairing}_mass_score.pdf', bbox_inches='tight')

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30,7))
        for i,(mass,mass2,ax) in enumerate(zip(mCand, mCand2, axs.flatten())):
            n,xe,ye,im = Hist2d(mass[mask], mass2[mask], bins=(mHbins, mHbins), ax=ax)
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(f'$m_{i+1}$ [GeV]')
            ax.set_ylabel(f'$m_{(i+2) % 3}$ [GeV]')
        fig.suptitle(title)
        fig.savefig(f'{saveas}_{regionType}_{pairing}_mass_scatter.pdf', bbox_inches='tight')

    elif regionType == 'sphere':
        print(".. evaluating SPHERICAL regions")
        regionDict = sphericalRegions(tree)
        print(regionDict.keys())
        print(f"VR       = {ak.sum(regionDict['VR']['all'])}")
        print(f"VR CR ls = {ak.sum(regionDict['VR']['CR']['ls'])}")
        print(f"VR CR hs = {ak.sum(regionDict['VR']['CR']['hs'])}")
        print(f"VR SR ls = {ak.sum(regionDict['VR']['SR']['ls'])}")
        print(f"VR SR hs = {ak.sum(regionDict['VR']['SR']['hs'])}")

        if isSignal:
            print(f"AR       = {ak.sum(regionDict['AR']['all'])}")
            print(f"AR CR ls = {ak.sum(regionDict['AR']['CR']['ls'])}")
            print(f"AR CR hs = {ak.sum(regionDict['AR']['CR']['hs'])}")
            print(f"AR SR ls = {ak.sum(regionDict['AR']['SR']['ls'])}")
            print(f"AR SR hs = {ak.sum(regionDict['AR']['SR']['hs'])}")
            mask = np.asarray(np.ones_like(m1), dtype=bool)
            title = 'Signal Events'
            saveas = f'NMSSM_{my}_GeV_{my}_GeV'
        else: 
            mask = ~regionDict['AR']['all']
            title = 'Data Events (Blinded)'
            saveas = f'data'

        hs_mask = mask & hi_score_mask
        ls_mask = mask & lo_score_mask
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30,7))
        for i,(mass,mass2,ax) in enumerate(zip(mCand, mCand2, axs.flatten())):
            n,xe,ye,im = Hist2d(mass[hs_mask], mass2[hs_mask], bins=(mHbins, mHbins), ax=ax)
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(f'$m_{i+1}$ [GeV]')
            ax.set_ylabel(f'$m_{(i+2) % 3}$ [GeV]')
            draw_circle = plt.Circle((125, 125), SR_edge, fill=False, color='k', linewidth=1.5)
            ax.add_artist(draw_circle)
            draw_circle = plt.Circle((125, 125), VR_edge, fill=False, color='k', linewidth=1.5)
            ax.add_artist(draw_circle)
            draw_circle = plt.Circle((185, 185), SR_edge, fill=False, color='k', linewidth=1.5)
            ax.add_artist(draw_circle)
            draw_circle = plt.Circle((185, 185), VR_edge, fill=False, color='k', linewidth=1.5)
            ax.add_artist(draw_circle)
        fig.suptitle(title + ', High Avg btag')
        fig.savefig(f'{saveas}_{regionType}_{pairing}_mass_scatter_hs.pdf', bbox_inches='tight')

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30,7))
        for i,(mass,mass2,ax) in enumerate(zip(mCand, mCand2, axs.flatten())):
            n,xe,ye,im = Hist2d(mass[ls_mask], mass2[ls_mask], bins=(mHbins, mHbins), ax=ax)
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(f'$m_{i+1}$ [GeV]')
            ax.set_ylabel(f'$m_{(i+2) % 3}$ [GeV]')
            draw_circle = plt.Circle((125, 125), SR_edge, fill=False, color='k', linewidth=1.5)
            ax.add_artist(draw_circle)
            draw_circle = plt.Circle((125, 125), VR_edge, fill=False, color='k', linewidth=1.5)
            ax.add_artist(draw_circle)
            draw_circle = plt.Circle((185, 185), SR_edge, fill=False, color='k', linewidth=1.5)
            ax.add_artist(draw_circle)
            draw_circle = plt.Circle((185, 185), VR_edge, fill=False, color='k', linewidth=1.5)
            ax.add_artist(draw_circle)
        fig.suptitle(title + ', Low Avg btag')
        fig.savefig(f'{saveas}_{regionType}_{pairing}_mass_scatter_ls.pdf', bbox_inches='tight')

    return

# -------------------------------------------------------------------
# Loop through trees and generate plots

# for data, signal in zip([data_dHHH_tree, data_mH_tree], [sig_dHHH_tree, sig_mH_tree]):

# print(".. plotting rectangular regions")

# print("  .. plotting signal")
# print("  .. dHHH")
# plotRegions(sig_dHHH_tree, 'rect', 'dHHH', True)
# print("  .. mH")
# plotRegions(sig_mH_tree, 'rect', 'mH', True)

# print("  .. plotting data")
# print("  .. dHHH")
# plotRegions(data_dHHH_tree, 'rect', 'dHHH')
# print("  .. mH")
# plotRegions(data_mH_tree, 'rect', 'mH')

print(".. plotting spherical regions")

print("  .. plotting signal")
print("  .. dHHH")
plotRegions(sig_dHHH_tree, 'sphere', 'dHHH', True)
print("  .. mH")
plotRegions(sig_mH_tree, 'sphere', 'mH', True)

print("  .. plotting data")
print("  .. dHHH")
plotRegions(data_dHHH_tree, 'sphere', 'dHHH')
print("  .. mH")
plotRegions(data_mH_tree, 'sphere', 'mH')

