"""
This will be a class that collects the six bs from the signal event and returns their p4s.
"""

import awkward1 as ak
import itertools
import numpy as np
import random
import vector

from colors import CYAN, W
from logger import info
from myuproot import open_up
from tqdm import tqdm

def get_6jet_p4(filename=None):

    vector.register_awkward()
    
    if filename:
        reco_filename = filename
    else:
        reco_filename = f'signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_reco_preselections.root'
        # reco_filename = 'signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_accstudies_500k_May2021.root'
        
    info(f"Opening ROOT file {CYAN}{reco_filename}{W} with columns")
    tree, table, nptab =  open_up(reco_filename, 'sixBtree')
    nevents = len(table)
    
    jet_idx = nptab['jet_idx']
    jet_pt = nptab['jet_pt']
    jet_eta = nptab['jet_eta']
    jet_phi = nptab['jet_phi']
#     jet_m = nptab['jet_m']
#     jet_btag = nptab['jet_btag']
    
    signal_p4 = []
    background_p4 = []
    
    pass_count = 0
    for evt in tqdm(range(nevents)):
        # if pass_count == 2120: break
        # Make a mask for jets matched to signal bs
        signal_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj > -1]
        if len(jet_pt[evt][signal_mask]) < 6: continue
        # Skip any events with duplicate matches (for now)
        if len(np.unique(jet_idx[evt][signal_mask])) < len(jet_idx[evt][signal_mask]): continue
        # Skip and events with less than 6 matching signal bs (for now)
        
        # Mask the background jets
        background_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj == -1]
        # Cound the background jets
        n_nonHiggs = len(jet_pt[evt][background_mask])
        if (n_nonHiggs < 1): continue
        # Choose a random background jet
        random_nonHiggs = random.choice(np.arange(0,n_nonHiggs))
        random_signalb_to_replace = random.choice(np.arange(0,6))
        
        sixb_pt = jet_pt[evt][signal_mask]
        random_pt = jet_pt[evt][background_mask][random_nonHiggs]
        non_sixb_pt = sixb_pt.copy()
        non_sixb_pt[random_signalb_to_replace] = random_pt
        
        sixb_eta = jet_eta[evt][signal_mask]
        random_eta = jet_eta[evt][background_mask][random_nonHiggs]
        non_sixb_eta = sixb_eta.copy()
        non_sixb_eta[random_signalb_to_replace] = random_eta
        
        sixb_phi = jet_phi[evt][signal_mask]
        random_phi = jet_phi[evt][background_mask][random_nonHiggs]
        non_sixb_phi = sixb_phi.copy()
        non_sixb_phi[random_signalb_to_replace] = random_phi
        
        sixb_pt = np.sort(sixb_pt)
        sixb_eta = sixb_eta[np.argsort(sixb_pt)]
        sixb_phi = sixb_phi[np.argsort(sixb_pt)]
        
        non_sixb_pt = np.sort(non_sixb_pt)
        non_sixb_eta = non_sixb_eta[np.argsort(non_sixb_pt)]
        non_sixb_phi = non_sixb_phi[np.argsort(non_sixb_pt)]

        signal = []
        for pt, eta, phi in zip(sixb_pt, sixb_eta, sixb_phi): 
            signal.append({
                        "pt":pt, 
                        "eta":eta, 
                        "phi":phi, 
                        "mass":4})
        background = []
        for pt, eta, phi in zip(non_sixb_pt, non_sixb_eta, non_sixb_phi):
            background.append({
                        "pt":pt, 
                        "eta":eta, 
                        "phi":phi, 
                        "mass":4})

        signal_p4.append(ak.Array(signal, with_name="MomentumObject4D"))
        background_p4.append(ak.Array(background, with_name="MomentumObject4D"))

        signal_p4.append(signal)
        background_p4.append(background)
        
        pass_count += 1

        
    return signal_p4, background_p4

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
            mass=np.repeat(4, nevents)))
        
    return signal_b_p4

def get_background_p4(filename=None):
    
    if filename:
        reco_filename = filename
    else:
        reco_filename = f'signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_reco_preselections.root'
        # reco_filename = 'signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_accstudies_500k_May2021.root'
        
    info(f"Opening ROOT file {CYAN}{reco_filename}{W} with columns")
    tree, table, nptab =  open_up(reco_filename, 'sixBtree')
    nevents = len(table)
    
    jet_idx = nptab['jet_idx']
    jet_pt = nptab['jet_pt']
    jet_eta = nptab['jet_eta']
    jet_phi = nptab['jet_phi']
#     jet_m = nptab['jet_m']
#     jet_btag = nptab['jet_btag']
    
    signal_p4 = []
    background_p4 = []
    
    pass_count = 0
    for evt in tqdm(range(nevents)):
        # Make a mask for jets matched to signal bs
        signal_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj > -1]
        # if len(jet_pt[evt][signal_mask]) < 6: continue
        # Skip any events with duplicate matches (for now)
        # if len(np.unique(jet_idx[evt][signal_mask])) < len(jet_idx[evt][signal_mask]): continue
        # Skip and events with less than 6 matching signal bs (for now)
        
        # Mask the background jets
        background_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj == -1]
        # Cound the background jets
        n_nonHiggs = len(jet_pt[evt][background_mask])
        if (n_nonHiggs < 6): continue
        pass_count += 1
        
        bkgd_pt = jet_pt[evt][background_mask][:6]
        bkgd_eta = jet_eta[evt][background_mask][:6]
        bkgd_phi = jet_phi[evt][background_mask][:6]
        # Choose a random background jet
        
        bkgd_pt = np.sort(bkgd_pt)
        bkgd_eta = bkgd_eta[np.argsort(bkgd_pt)]
        bkgd_phi = bkgd_phi[np.argsort(bkgd_pt)]

        background_p4.append(vector.obj(
                        pt=bkgd_pt, 
                        eta=bkgd_eta, 
                        phi=bkgd_phi, 
                        mass=np.repeat(4, 6)))
        
    print(pass_count)
        
    return background_p4