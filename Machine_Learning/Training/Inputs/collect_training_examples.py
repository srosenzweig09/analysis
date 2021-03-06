# This script is used to construct the inputs used to train the neural network (NN).
# ***Intended to replace combinations.py!***

import numpy as np
import uproot3_methods
from random import sample
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from uproot_open import get_uproot_Table
from logger import info
from kinematics import calcDeltaR



### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--MX'    , dest = 'MX'    , help = 'Mass of X resonance' , default = 700        )
parser.add_argument('--MY'    , dest = 'MY'    , help = 'Mass of Y resonance' , default = 400        )
parser.add_argument('--type'  , dest = 'type'  , help = 'parton or reco'      , required = True      )
parser.add_argument('--no_presel', dest = 'presel', help = 'apply preselections?', action = 'store_false',
                    default = True)

args = parser.parse_args()


### ------------------------------------------------------------------------------------
## Implement command line parser

def get_nonHiggs(keep):

    indices = sample(top_hat, 1)[0] # Choose three random non-Higgs pair indices

    if indices in keep:
        while indices in keep:
            indices = sample(top_hat, 1)[0]
    ind_sep = [int(indices[0]), int(indices[1])] # Easiest way to use the indices later
    ind_1, ind_2 = ind_sep[0], ind_sep[1]

    assert(pair_dict[ind_1] != ind_2) # Verify not a Higgs pair
    assert(ind_1 != ind_2) # Verify not a self-pairing

    input_1 = np.array((part_dict[ind_1]['pt'][ievt], part_dict[ind_1]['eta'][ievt], part_dict[ind_1]['phi'][ievt]))
    input_2 = np.array((part_dict[ind_2]['pt'][ievt], part_dict[ind_2]['eta'][ievt], part_dict[ind_2]['phi'][ievt]))
    input_dR = calcDeltaR(part_dict[ind_1]['eta'][ievt], part_dict[ind_2]['eta'][ievt], part_dict[ind_1]['phi'][ievt], part_dict[ind_2]['phi'][ievt])
    features = np.concatenate((input_1, input_2))
    features = np.append(features, input_dR)
    return features, indices


### ------------------------------------------------------------------------------------
## Load signal events

MX = args.MX
MY = args.MY

reco_filename = f'NanoAOD/NMSSM_XYH_YToHH_6b_MX_{MX}_MY_{MY}_accstudies.root'
table =  get_uproot_Table(reco_filename, 'sixBtree')
nevents = table._length()

info(f"Opening ROOT file {reco_filename} with columns\n{table.columns}")

### ------------------------------------------------------------------------------------
## Prepare bs for pairing

if args.type == 'parton':
    tag = ''
    type = 'Gen'
if args.type == 'reco':
    tag = '_recojet'
    type = 'Reco'

HX_b1  = {'pt':table[f'gen_HX_b1{tag}_pt' ], 'eta':table[f'gen_HX_b1{tag}_eta' ], 
          'phi':table[f'gen_HX_b1{tag}_phi' ], 'm':table[f'gen_HX_b1{tag}_m' ]}
HX_b2  = {'pt':table[f'gen_HX_b2{tag}_pt' ], 'eta':table[f'gen_HX_b2{tag}_eta' ],
          'phi':table[f'gen_HX_b2{tag}_phi' ], 'm':table[f'gen_HX_b2{tag}_m' ]}
HY1_b1 = {'pt':table[f'gen_HY1_b1{tag}_pt'], 'eta':table[f'gen_HY1_b1{tag}_eta'],
          'phi':table[f'gen_HY1_b1{tag}_phi'], 'm':table[f'gen_HY1_b1{tag}_m']}
HY1_b2 = {'pt':table[f'gen_HY1_b2{tag}_pt'], 'eta':table[f'gen_HY1_b2{tag}_eta'], 
          'phi':table[f'gen_HY1_b2{tag}_phi'], 'm':table[f'gen_HY1_b2{tag}_m']}
HY2_b1 = {'pt':table[f'gen_HY2_b1{tag}_pt'], 'eta':table[f'gen_HY2_b1{tag}_eta'],
          'phi':table[f'gen_HY2_b1{tag}_phi'], 'm':table[f'gen_HY2_b1{tag}_m']}
HY2_b2 = {'pt':table[f'gen_HY2_b2{tag}_pt'], 'eta':table[f'gen_HY2_b2{tag}_eta'], 
          'phi':table[f'gen_HY2_b2{tag}_phi'], 'm':table[f'gen_HY2_b2{tag}_m']}

part_dict = {0:HX_b1, 1:HX_b2, 2:HY1_b1, 3:HY1_b2, 4:HY2_b1, 5:HY2_b2}
pair_dict = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4} # Used later to verify that non-Higgs
                                           # pair candidates are truly non-Higgs pairs

params = ['pt1', 'eta1', 'phi1', 'pt2', 'eta2', 'phi2', 'DeltaR_12']
nfeatures = len(params)

# top_hat contains non-Higgs pair indices (Higgs pairs are 01, 23, and 45).
top_hat = ['02', '03', '04', '05', '12', '13', '14', '15', '24', '25', '34', '35']

random_selection = np.array(())

nevt = len(table[f'gen_HX_b1{tag}_pt'])
print("File contains",nevt,"events.")

evt_indices = np.arange(nevt)
test_size = 0.20
val_size = 0.125
evt_train, evt_test = train_test_split(evt_indices, test_size=test_size)
evt_train, evt_val = train_test_split(evt_train, test_size=val_size)

### ------------------------------------------------------------------------------------
## Loop through events and build arrays of features

if args.presel:
    pt_mask = ((table[f'gen_HX_b1{tag}_pt' ] > 20) &
               (table[f'gen_HX_b2{tag}_pt' ] > 20) &
               (table[f'gen_HY1_b1{tag}_pt'] > 20) &
               (table[f'gen_HY1_b2{tag}_pt'] > 20) &
               (table[f'gen_HY2_b1{tag}_pt'] > 20) &
               (table[f'gen_HY2_b2{tag}_pt'] > 20))

    eta_mask = ((np.abs(table[f'gen_HX_b1{tag}_eta' ]) < 2.4) &
                (np.abs(table[f'gen_HX_b2{tag}_eta' ]) < 2.4) &
                (np.abs(table[f'gen_HY1_b1{tag}_eta']) < 2.4) &
                (np.abs(table[f'gen_HY1_b2{tag}_eta']) < 2.4) &
                (np.abs(table[f'gen_HY2_b1{tag}_eta']) < 2.4) &
                (np.abs(table[f'gen_HY2_b2{tag}_eta']) < 2.4))

    evt_mask = pt_mask & eta_mask
    print(np.sum(evt_mask*1))


x_train = np.array(())
y_train = np.array(())
m_train = np.array(())

x_test = np.array(())
y_test = np.array(())
m_test = np.array(())

x_val = np.array(())
y_val = np.array(())
m_val = np.array(())

# extra_bkgd_x = np.array(())
# extra_bkgd_mjj = np.array(())
# extra_bkgd_y = np.array(())

count = 0

for ievt in range(nevt):
   # Loop over events, select the three Higgs pairs, select three random non-Higgs pairs
    if ievt % 10000 == 0: print("Processing evt {} / {}".format(ievt,nevt))
    if args.presel:
        if not evt_mask[ievt]: continue

    # TLorentzArray requires an input of arrays (not scalars) so I have to calculate the
    # p4 for all 6 bs together
    # Calculate invariant mass of Higgs pairs

    # Prepare inputs for Higgs pairs
    HX_b1_input = np.array((HX_b1['pt'][ievt], HX_b1['eta'][ievt], HX_b1['phi'][ievt]))
    HX_b2_input = np.array((HX_b2['pt'][ievt], HX_b2['eta'][ievt], HX_b2['phi'][ievt]))
    HX_dR = calcDeltaR(HX_b1['eta'][ievt], HX_b2['eta'][ievt], HX_b1['phi'][ievt], HX_b2['phi'][ievt])
    HX_input = np.concatenate((HX_b1_input, HX_b2_input))
    HX_input = np.append(HX_input, HX_dR)

    HY1_b1_input = np.array((HY1_b1['pt'][ievt], HY1_b1['eta'][ievt], HY1_b1['phi'][ievt]))
    HY1_b2_input = np.array((HY1_b2['pt'][ievt], HY1_b2['eta'][ievt], HY1_b2['phi'][ievt]))
    HY1_dR = calcDeltaR(HY1_b1['eta'][ievt], HY1_b2['eta'][ievt], HY1_b1['phi'][ievt], HY1_b2['phi'][ievt])
    HY1_input = np.concatenate((HY1_b1_input, HY1_b2_input))
    HY1_input = np.append(HY1_input, HY1_dR)
    
    HY2_b1_input = np.array((HY2_b1['pt'][ievt], HY2_b1['eta'][ievt], HY2_b1['phi'][ievt]))
    HY2_b2_input = np.array((HY2_b2['pt'][ievt], HY2_b2['eta'][ievt], HY2_b2['phi'][ievt]))
    HY2_dR = calcDeltaR(HY2_b1['eta'][ievt], HY2_b2['eta'][ievt], HY2_b1['phi'][ievt], HY2_b2['phi'][ievt])
    HY2_input = np.concatenate((HY2_b1_input, HY2_b2_input))
    HY2_input = np.append(HY2_input, HY2_dR)

    H_input = np.concatenate((HX_input, HY1_input, HY2_input))
    H_in = H_input.reshape(3, nfeatures)

    b1 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(H_in[:,0], H_in[:,1], H_in[:,2], np.zeros_like(H_in[:,2]))
    b2 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(H_in[:,3], H_in[:,4], H_in[:,5], np.zeros_like(H_in[:,5]))
    b1_b2 = b1 + b2
    mbb = b1_b2.mass

    # Preparing inputs for non-Higgs pairs
    keepsies = []
    nonH_kins = np.array(())
    for _ in np.arange(3):
        features, indices = get_nonHiggs(keepsies)
        keepsies.append(indices)
        nonH_kins = np.append(nonH_kins, features)

    nfeatures = len(features)
    nonH_in = nonH_kins.reshape(3, nfeatures)
    random_selection = np.append(random_selection, np.ravel(keepsies))
    
    non_Higgs_bs_0 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(nonH_in[:,0], nonH_in[:,1], nonH_in[:,2], np.zeros_like(nonH_in[:,2]))
    non_Higgs_bs_1 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(nonH_in[:,3], nonH_in[:,4], nonH_in[:,5], np.zeros_like(nonH_in[:,5]))
    non_Higgs = non_Higgs_bs_0 + non_Higgs_bs_1
    m_nonH = non_Higgs.mass

    y = np.concatenate((np.repeat(1, 3), np.repeat(0, 3)))
    
    if ievt in evt_train:
        m_train = np.concatenate((m_train, mbb, m_nonH))
        # Append 3 true booleans and 3 false booleans, corresponding to the three Higgs pairs (1) and three non-Higgs pairs (0)
        y_train = np.append(y_train, y)
        # Stacking Higgs pair inputs, each with shape (6,1), with randomly chosen non-Higgs pair inputs
        x_train = np.append(x_train, np.concatenate((H_input, nonH_kins)))

    elif ievt in evt_test:
        m_test = np.concatenate((m_test, mbb, m_nonH))
        # Append 3 true booleans and 3 false booleans, corresponding to the three Higgs pairs (1) and three non-Higgs pairs (0)
        y_test = np.append(y_test, y)
        # Stacking Higgs pair inputs, each with shape (6,1), with randomly chosen non-Higgs pair inputs
        x_test = np.append(x_test, np.concatenate((H_input, nonH_kins)))

    elif ievt in evt_val:
        m_val = np.concatenate((m_val, mbb, m_nonH))
        # Append 3 true booleans and 3 false booleans, corresponding to the three Higgs pairs (1) and three non-Higgs pairs (0)
        y_val = np.append(y_val, y)
        # Stacking Higgs pair inputs, each with shape (6,1), with randomly chosen non-Higgs pair inputs
        x_val = np.append(x_val, np.concatenate((H_input, nonH_kins)))


x_train = x_train.reshape(evt_train, nfeatures)
x_test = x_test.reshape(evt_test, nfeatures)
x_val = x_val.reshape(evt_val, nfeatures)
# extra_bkgd_x = extra_bkgd_x.reshape(int(len(extra_bkgd_x)/len(HX_input)), len(HX_input))

# np.savez(f"{type}_Inputs/nn_input_MX{args.MX}_MY{args.MY}_class", x=x,  y=y,  mjj=mjj, extra_bkgd_x=extra_bkgd_x, extra_bkgd_mjj=extra_bkgd_mjj, extra_bkgd_y = extra_bkgd_y, params=params, random_selection=random_selection)

# np.savez(f"{type}_Inputs/nn_input_MX{args.MX}_MY{args.MY}_class", x=x,  y=y,  mjj=mjj, params=params, random_selection=random_selection)

np.savez(f"{type}_Inputs/nn_input_MX{args.MX}_MY{args.MY}_class", x_train=x_train,  x_test=x_test, x_val=x_val,  y_train=y_train, y_test=y_test, y_val=y_val,  m_test=m_test, train=evt_train, val=evt_val, test=evt_test, params=params, random_selection=random_selection)