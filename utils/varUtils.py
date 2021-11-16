"""
Author: Suzanne Rosenzweig
"""

import numpy as np

## control regions
m_diff = 60
NN6j = 0.8
NN3d = 0.8

## b tag working points (2018)
from .cutConfig import jet_btagWP

tight_b = jet_btagWP[3]
medium_b = jet_btagWP[2]
loose_b = jet_btagWP[1]

# bins for histograms
score_bins = np.linspace(0,1,101)
mass_bins = np.linspace(0,350,101)

def x_bins(bins):
    return (bins[:-1] + bins[1:])/2

