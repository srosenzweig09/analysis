"""
This will be a class that collects the six bs from the signal event and returns their p4s.
"""

import itertools
import numpy as np
import random
import vector

from colors import CYAN, W
from logger import info
from myuproot import open_up
from tqdm import tqdm

def get_6jet_p4(filename=None):
    
    if filename:
        reco_filename = filename
    else:
        reco_filename = f'signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_reco_preselections.root'
        
    info(f"Opening ROOT file {CYAN}{reco_filename}{W} with columns")
    tree, table, nptab =  open_up(reco_filename, 'sixBtree')
    nevents = len(table)
    
    jet_idx = nptab['jet_idx']
    jet_pt = nptab['jet_pt']
    jet_eta = nptab['jet_eta']
    jet_phi = nptab['jet_phi']
#     jet_m = nptab['jet_m']
#     jet_btag = nptab['jet_btag']
    
    jet_p4 = []
    
    for evt in tqdm(range(5000)):
        # Make a mask for jets matched to signal bs
        signal_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj > -1]
        if len(jet_pt[evt][signal_mask]) < 7: continue
        print(jet_idx[evt][signal_mask])
        # Skip any events with duplicate matches (for now)
        if len(np.unique(jet_idx[evt][signal_mask])) < len(jet_idx[evt][signal_mask]): continue
        print(signal_mask)
        # Skip and events with less than 6 matching signal bs (for now)
        
        # Mask the background jets
        background_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj == -1]
        # Cound the background jets
        n_nonHiggs = len(jet_pt[evt][background_mask])
        # Choose a random background jet
        random_nonHiggs = random.choice(np.arange(0,n_nonHiggs))
        
        print(signal_mask)
        
        sixb_pt = jet_pt[evt][signal_mask]
        random_pt = jet_pt[evt][background_mask][random_nonHiggs]
        # print(sixb_pt)
        # print(random_pt)
        pt_combos = list(itertools.combinations(np.append(sixb_pt, random_pt), 6))
        
        sixb_eta = jet_eta[evt][signal_mask]
        random_eta = jet_eta[evt][background_mask][random_nonHiggs]
        eta_combos = list(itertools.combinations(np.append(sixb_eta, random_eta), 6))
        
        sixb_phi = jet_phi[evt][signal_mask]
        random_phi = jet_phi[evt][background_mask][random_nonHiggs]
        phi_combos = list(itertools.combinations(np.append(sixb_phi, random_phi), 6))
    
        jet_p4.append(vector.obj(
                        pt=pt_combos, 
                        eta=eta_combos, 
                        phi=phi_combos, 
                        mass=np.repeat(4e-9, 6)))
        
    return jet_p4

def get_sixb_p4(filename=None):
    
    if filename:
        reco_filename = filename
    else:
        reco_filename = f'signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_reco_preselections.root'
        
    info(f"Opening ROOT file {CYAN}{reco_filename}{W} with columns")
    tree, table, nptab =  open_up(reco_filename, 'sixBtree')
    nevents = len(table)

    HX_b1  = {'pt': nptab[f'HX_b1_recojet_ptRegressed' ],
            'eta':nptab[f'HX_b1_recojet_eta'],
            'phi':nptab[f'HX_b1_recojet_phi'],
            'm':  nptab[f'HX_b1_recojet_m'  ]}
    HX_b2  = {'pt': nptab[f'HX_b2_recojet_ptRegressed' ],
            'eta':nptab[f'HX_b2_recojet_eta'],
            'phi':nptab[f'HX_b2_recojet_phi'],
            'm':  nptab[f'HX_b2_recojet_m'  ]}
    HY1_b1 = {'pt': nptab[f'HY1_b1_recojet_ptRegressed'],
            'eta':nptab[f'HY1_b1_recojet_eta'],
            'phi':nptab[f'HY1_b1_recojet_phi'],
            'm':  nptab[f'HY1_b1_recojet_m' ]}
    HY1_b2 = {'pt': nptab[f'HY1_b2_recojet_ptRegressed'],
            'eta':nptab[f'HY1_b2_recojet_eta'],
            'phi':nptab[f'HY1_b2_recojet_phi'],
            'm':  nptab[f'HY1_b2_recojet_m' ]}
    HY2_b1 = {'pt': nptab[f'HY2_b1_recojet_ptRegressed'],
            'eta':nptab[f'HY2_b1_recojet_eta'],
            'phi':nptab[f'HY2_b1_recojet_phi'],
            'm':  nptab[f'HY2_b1_recojet_m' ]}
    HY2_b2 = {'pt': nptab[f'HY2_b2_recojet_ptRegressed'],
            'eta':nptab[f'HY2_b2_recojet_eta'],
            'phi':nptab[f'HY2_b2_recojet_phi'],
            'm':  nptab[f'HY2_b2_recojet_m' ]}

    part_dict = {0:HX_b1, 1:HX_b2, 2:HY1_b1, 3:HY1_b2, 4:HY2_b1, 5:HY2_b2}
    part_name = {0:'HX_b1', 1:'HX_b2', 2:'HY1_b1', 3:'HY1_b2', 4:'HY2_b1', 5:'HY2_b2'}
    pair_dict = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4} # Used later to verify that non-Higgs
                                            # pair candidates are truly non-Higgs pairs

    nonHiggs_labels = np.array((
    'X b1, Y1 b1',
    'X b1, Y1 b2',
    'X b1, Y2 b1',
    'X b1, Y2 b2',
    'X b2, Y1 b1',
    'X b2, Y1 b2',
    'X b2, Y2 b1',
    'X b2, Y2 b2',
    'Y1 b1, Y2 b1',
    'Y1 b1, Y2 b2',
    'Y1 b2, Y2 b1',
    'Y1 b2, Y2 b2'))

    signal_b_p4 = []
    for i in range(6):
        signal_b_p4.append(vector.obj(
            pt=part_dict[i]['pt'], 
            eta=part_dict[i]['eta'], 
            phi=part_dict[i]['phi'], 
            mass=np.repeat(4e-9, nevents)))
        
    return signal_b_p4