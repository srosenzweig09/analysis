# This script is used to construct the inputs used to train the neural network (NN).
# ***Intended to replace combinations.py!***

import numpy as np
import uproot3_methods
from random import sample
from argparse import ArgumentParser

from uproot_open import get_uproot_Table
from logger import info
from kinematics import calcDeltaR

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--MX'       , dest = 'MX'       , help = 'Mass of X resonance'                      ,  required = True               )
parser.add_argument('--MY'       , dest = 'MY'       , help = 'Mass of Y resonance'                      ,  required = True               )
# parser.add_argument('--tag'       , dest = 'tag'       , help = 'production tag'                      ,  required = True               )
# parser.add_argument('--nlayers'   , dest = 'nlayers'   , help = 'number of hidden layers'             ,  required = False , type = int )

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Load signal events

MX = args.MX
MY = args.MY

reco_filename = f'/eos/user/s/srosenzw/SWAN_projects/sixB/Analysis_6b/Data_Preparation/NMSSM_XYH_YToHH_6b_MX_{MX}_MY_{MY}_accstudies.root'
table =  get_uproot_Table(reco_filename, 'sixBtree')
nevents = table._length()

info(f"Opening ROOT file {reco_filename} with columns\n{table.columns}")


# # filename = '/eos/user/s/srosenzw/SWAN_projects/sixB/Signal_Exploration/Mass_Pair_ROOT_files/X_YH_HHH_6b_MX700_MY400.root'

# with open('sim_files.txt') as f:
#     filename = f.readlines()
# filename = filename.split('*')
# filename = filename[0] + MX + filename[1] + MY  + filename[2]

# print(filename)

# print()
# print("Creating NN training inputs for:",filename)
# f = uproot3.open(filename)
# tree = f['sixbntuplizer/sixBtree']
# branches = tree.arrays(namedecode='utf-8')
# table = awk.Table(branches)

### ------------------------------------------------------------------------------------
## Prepare bs for pairing

HX_b1 = {'pt':table['gen_HX_b1_recojet_pt'], 'eta':table['gen_HX_b1_recojet_eta'], 'phi':table['gen_HX_b1_recojet_phi'], 'm':table['gen_HX_b1_recojet_m']}
HX_b2 = {'pt':table['gen_HX_b2_pt'], 'eta':table['gen_HX_b2_eta'], 'phi':table['gen_HX_b2_phi'], 'm':table['gen_HX_b2_m']}
HY1_b1 = {'pt':table['gen_HY1_b1_pt'], 'eta':table['gen_HY1_b1_eta'], 'phi':table['gen_HY1_b1_phi'], 'm':table['gen_HY1_b1_m']}
HY1_b2 = {'pt':table['gen_HY1_b2_pt'], 'eta':table['gen_HY1_b2_eta'], 'phi':table['gen_HY1_b2_phi'], 'm':table['gen_HY1_b2_m']}
HY2_b1 = {'pt':table['gen_HY2_b1_pt'], 'eta':table['gen_HY2_b1_eta'], 'phi':table['gen_HY2_b1_phi'], 'm':table['gen_HY2_b1_m']}
HY2_b2 = {'pt':table['gen_HY2_b2_pt'], 'eta':table['gen_HY2_b2_eta'], 'phi':table['gen_HY2_b2_phi'], 'm':table['gen_HY2_b2_m']}
part_dict = {0:HX_b1, 1:HX_b2, 2:HY1_b1, 3:HY1_b2, 4:HY2_b1, 5:HY2_b2}
pair_dict = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4} # Used later to verify that non-Higgs pair candidates are truly non-Higgs pairs

params = ['pt1', 'eta1', 'phi1', 'pt2', 'eta2', 'phi2', 'pt1*pt2', 'DeltaR_12']

# b1 kinematics for HX, HY1, and HY2
b1_pt_arr  = [part_dict[i]['pt']  for i in range(0,6,2)]
b1_eta_arr = [part_dict[i]['eta'] for i in range(0,6,2)]
b1_phi_arr = [part_dict[i]['phi'] for i in range(0,6,2)]
b1_m_arr   = [part_dict[i]['m']   for i in range(0,6,2)]

# b2 kinematics for HX, HY1, and HY2
b2_pt_arr  = [part_dict[i]['pt']  for i in range(1,6,2)]
b2_eta_arr = [part_dict[i]['eta'] for i in range(1,6,2)]
b2_phi_arr = [part_dict[i]['phi'] for i in range(1,6,2)]
b2_m_arr   = [part_dict[i]['m']   for i in range(1,6,2)]

# top_hat contains non-Higgs pair indices (Higgs pairs are 01, 23, and 45).
top_hat = ['02', '03', '04', '05', '12', '13', '14', '15', '24', '25', '34', '35']

random_selection = np.array(())

nevt = len(table['gen_HX_b1_recojet_pt'])
print("File contains",nevt,"events.")

### ------------------------------------------------------------------------------------
## Loop through events and build arrays of features

x = np.array(())
y = np.array(())
mjj = np.array(())
extra_bkgd_x = np.array(())
extra_bkgd_mjj = np.array(())
extra_bkgd_y = np.array(())

count = 0

for ievt in range(nevt):
   # Loop over events, select the three Higgs pairs, select three random non-Higgs pairs
    if ievt % 10000 == 0: print("Processing evt {} / {}".format(ievt,nevt))

    # TLorentzArray requires an input of arrays (not scalars) so I have to calculate the
    # p4 for all 6 bs together
    # Calculate invariant mass of Higgs pairs
    
    pt1 = np.array((b1_pt_arr[0][ievt], b1_pt_arr[1][ievt], b1_pt_arr[2][ievt]))
    eta1 = np.array((b1_eta_arr[0][ievt], b1_eta_arr[1][ievt], b1_eta_arr[2][ievt]))
    phi1 = np.array((b1_phi_arr[0][ievt], b1_phi_arr[1][ievt], b1_phi_arr[2][ievt]))
    m1 = np.array((b1_m_arr[0][ievt], b1_m_arr[1][ievt], b1_m_arr[2][ievt]))
    
    pt2 = np.array((b2_pt_arr[0][ievt], b2_pt_arr[1][ievt], b2_pt_arr[2][ievt]))
    eta2 = np.array((b2_eta_arr[0][ievt], b2_eta_arr[1][ievt], b2_eta_arr[2][ievt]))
    phi2 = np.array((b2_phi_arr[0][ievt], b2_phi_arr[1][ievt], b2_phi_arr[2][ievt]))
    m2 = np.array((b2_m_arr[0][ievt], b2_m_arr[1][ievt], b2_m_arr[2][ievt]))
    
    # Cuts applied here
    pt_mask = np.concatenate((pt1 > 20,  pt2 > 20))
    eta_mask = np.concatenate((np.abs(eta1) < 2.4, np.abs(eta2) < 2.4))
    evt_mask = pt_mask & eta_mask
    if np.sum(evt_mask*1) < 6: continue
    
    count+=1
    
    b1 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(pt1, eta1, phi1, m1)
    b2 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(pt2, eta2, phi2, m2)
    
    b1_b2 = b1 + b2
    mbb = b1_b2.mass

    # Prepare inputs for Higgs pairs
    HX_b1_input = np.array((HX_b1['pt'][ievt], HX_b1['eta'][ievt], HX_b1['phi'][ievt]))
    HX_b2_input = np.array((HX_b2['pt'][ievt], HX_b2['eta'][ievt], HX_b2['phi'][ievt]))
    HX_input = np.concatenate((HX_b1_input, HX_b2_input))
    HX_dR = calcDeltaR(HX_b1['eta'][ievt], HX_b2['eta'][ievt], HX_b1['phi'][ievt], HX_b2['phi'][ievt])
    HX_input = np.append(HX_input, (HX_b1_input[0]*HX_b2_input[0], HX_dR)) # product of b pTs, deltaR

    HY1_b1_input = np.array((HY1_b1['pt'][ievt], HY1_b1['eta'][ievt], HY1_b1['phi'][ievt]))
    HY1_b2_input = np.array((HY1_b2['pt'][ievt], HY1_b2['eta'][ievt], HY1_b2['phi'][ievt]))
    HY1_input = np.concatenate((HY1_b1_input, HY1_b2_input))
    HY1_dR = calcDeltaR(HY1_b1['eta'][ievt], HY1_b2['eta'][ievt], HY1_b1['phi'][ievt], HY1_b2['phi'][ievt])
    HY1_input = np.append(HY1_input, (HY1_b1_input[0]*HY1_b2_input[0], HY1_dR)) # product of b pTs, deltaR
    
    HY2_b1_input = np.array((HY2_b1['pt'][ievt], HY2_b1['eta'][ievt], HY2_b1['phi'][ievt]))
    HY2_b2_input = np.array((HY2_b2['pt'][ievt], HY2_b2['eta'][ievt], HY2_b2['phi'][ievt]))
    HY2_input = np.concatenate((HY2_b1_input, HY2_b2_input))
    HY2_dR = calcDeltaR(HY2_b1['eta'][ievt], HY2_b2['eta'][ievt], HY2_b1['phi'][ievt], HY2_b2['phi'][ievt])
    HY2_input = np.append(HY2_input, (HY2_b1_input[0]*HY2_b2_input[0], HY2_dR)) # product of b pTs, deltaR

    
    # Preparing inputs for non-Higgs pairs
    keepsies = sample(top_hat, 3) # Choose three random non-Higgs pair indices
    keep_arr = [int(keepsies[i][j]) for i in range(3) for j in range(2)] # Easiest way to use the indices later
    random_selection = np.append(random_selection, np.ravel(keepsies))
    
    ind_1_1, ind_1_2 = keep_arr[0], keep_arr[1]
    assert(pair_dict[ind_1_1] != ind_1_2) # Verify not a Higgs pair
    assert(ind_1_1 != ind_1_2) # Verify not a self-pairing
    input_1_1 = np.array((part_dict[ind_1_1]['pt'][ievt], part_dict[ind_1_1]['eta'][ievt], part_dict[ind_1_1]['phi'][ievt]))
    input_1_2 = np.array((part_dict[ind_1_2]['pt'][ievt], part_dict[ind_1_2]['eta'][ievt], part_dict[ind_1_2]['phi'][ievt]))
    input_1 = np.concatenate((input_1_1, input_1_2))
    input_1_dR = calcDeltaR(part_dict[ind_1_1]['eta'][ievt], part_dict[ind_1_2]['eta'][ievt], part_dict[ind_1_1]['phi'][ievt], part_dict[ind_1_2]['phi'][ievt])
    input_1 = np.append(input_1, (input_1_1[0]*input_1_2[0], input_1_dR)) # product of b pTs
    
    ind_2_1, ind_2_2 = keep_arr[2], keep_arr[3]
    assert(pair_dict[ind_2_1] != ind_2_2) # Verify not a Higgs pair
    assert(ind_2_1 != ind_2_2) # Verify not a self-pairing
    input_2_1 = np.array((part_dict[ind_2_1]['pt'][ievt], part_dict[ind_2_1]['eta'][ievt], part_dict[ind_2_1]['phi'][ievt]))
    input_2_2 = np.array((part_dict[ind_2_2]['pt'][ievt], part_dict[ind_2_2]['eta'][ievt], part_dict[ind_2_2]['phi'][ievt]))
    input_2 = np.concatenate((input_2_1, input_2_2))
    input_2_dR = calcDeltaR(part_dict[ind_2_1]['eta'][ievt], part_dict[ind_2_2]['eta'][ievt], part_dict[ind_2_1]['phi'][ievt], part_dict[ind_2_2]['phi'][ievt])
    input_2 = np.append(input_2, (input_2_1[0]*input_2_2[0], input_2_dR)) # product of b pTs
    
    ind_3_1, ind_3_2 = keep_arr[4], keep_arr[5]
    assert(pair_dict[ind_3_1] != ind_3_2) # Verify not a Higgs pair
    assert(ind_3_1 != ind_3_2) # Verify not a self-pairing
    input_3_1 = np.array((part_dict[ind_3_1]['pt'][ievt], part_dict[ind_3_1]['eta'][ievt], part_dict[ind_3_1]['phi'][ievt]))
    input_3_2 = np.array((part_dict[ind_3_2]['pt'][ievt], part_dict[ind_3_2]['eta'][ievt], part_dict[ind_3_2]['phi'][ievt]))
    input_3 = np.concatenate((input_3_1, input_3_2))
    input_3_dR = calcDeltaR(part_dict[ind_3_1]['eta'][ievt], part_dict[ind_3_2]['eta'][ievt], part_dict[ind_3_1]['phi'][ievt], part_dict[ind_3_2]['phi'][ievt])
    input_3 = np.append(input_3, (input_3_1[0]*input_3_2[0], input_3_dR)) # product of b pTs
    
    non_Higgs_b1s = []
    non_Higgs_b2s = []
    for th in top_hat:
        if th in set(keepsies):
            ind_arr = [int(th[0]), int(th[1])]
            ind_0_1, ind_0_2 = ind_arr[0], ind_arr[1]
            assert(pair_dict[ind_0_1] != ind_0_2) # Verify not a Higgs pair
            assert(ind_0_1 != ind_0_2) # Verify not a self-pairing
            input_0_1 = np.array((part_dict[ind_0_1]['pt'][ievt], part_dict[ind_0_1]['eta'][ievt], part_dict[ind_0_1]['phi'][ievt]))
            input_0_2 = np.array((part_dict[ind_0_2]['pt'][ievt], part_dict[ind_0_2]['eta'][ievt], part_dict[ind_0_2]['phi'][ievt]))
            input_0 = np.concatenate((input_0_1, input_0_2))
            input_0_dR = calcDeltaR(part_dict[ind_0_1]['eta'][ievt], part_dict[ind_0_2]['eta'][ievt], part_dict[ind_0_1]['phi'][ievt], part_dict[ind_0_2]['phi'][ievt])
            input_0 = np.append(input_0, (input_0_1[0]*input_0_2[0], input_0_dR)) # product of b pTs
            
            extra_bkgd_x = np.append(extra_bkgd_x, input_0)
            extra_bkgd_y = np.append(extra_bkgd_y, 0)
            
            pt  = np.array((part_dict[ind_0_1]['pt'][ievt],  part_dict[ind_0_2]['pt'][ievt]))
            eta = np.array((part_dict[ind_0_1]['eta'][ievt], part_dict[ind_0_2]['eta'][ievt]))
            phi = np.array((part_dict[ind_0_1]['phi'][ievt], part_dict[ind_0_2]['phi'][ievt]))
            m   = np.array((part_dict[ind_0_1]['m'][ievt],   part_dict[ind_0_2]['m'][ievt]))

            extra_non_Higgs = uproot3_methods.TLorentzVectorArray.from_ptetaphim(pt, eta, phi, m)
            extra_non_Higgs_mjj = (extra_non_Higgs[0] + extra_non_Higgs[1]).mass

    extra_bkgd_mjj = np.append(extra_bkgd_mjj, extra_non_Higgs_mjj)
    
    # Calculate invariant mass for non-Higgs pairs
    pt_arr0  = [part_dict[keep_arr[i]]['pt']  for i in range(0,6,2)]
    eta_arr0 = [part_dict[keep_arr[i]]['eta'] for i in range(0,6,2)]
    phi_arr0 = [part_dict[keep_arr[i]]['phi'] for i in range(0,6,2)]
    m_arr0   = [part_dict[keep_arr[i]]['m']   for i in range(0,6,2)]
    
    pt_arr1  = [part_dict[keep_arr[i]]['pt']  for i in range(1,6,2)]
    eta_arr1 = [part_dict[keep_arr[i]]['eta'] for i in range(1,6,2)]
    phi_arr1 = [part_dict[keep_arr[i]]['phi'] for i in range(1,6,2)]
    m_arr1   = [part_dict[keep_arr[i]]['m']   for i in range(1,6,2)]
    
    pt  = np.array((pt_arr0[0][ievt],  pt_arr0[1][ievt],  pt_arr0[2][ievt]))
    eta = np.array((eta_arr0[0][ievt], eta_arr0[1][ievt], eta_arr0[2][ievt]))
    phi = np.array((phi_arr0[0][ievt], phi_arr0[1][ievt], phi_arr0[2][ievt]))
    m   = np.array((m_arr0[0][ievt],   m_arr0[1][ievt],   m_arr0[2][ievt]))
    
    non_Higgs_bs_0 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(pt, eta, phi, m)
    
    pt  = np.array((pt_arr1[0][ievt],  pt_arr1[1][ievt],  pt_arr1[2][ievt]))
    eta = np.array((eta_arr1[0][ievt], eta_arr1[1][ievt], eta_arr1[2][ievt]))
    phi = np.array((phi_arr1[0][ievt], phi_arr1[1][ievt], phi_arr1[2][ievt]))
    m   = np.array((m_arr1[0][ievt],   m_arr1[1][ievt],   m_arr1[2][ievt]))

    non_Higgs_bs_1 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(pt, eta, phi, m)
    
    non_Higgs = non_Higgs_bs_0 + non_Higgs_bs_1
    m_nonH = non_Higgs.mass
    
    mjj = np.concatenate((mjj, mbb, m_nonH))
    
    # Append 3 true booleans and 3 false booleans, corresponding to the three Higgs pairs (1) and three non-Higgs pairs (0)
    y = np.append(y, np.array((1,1,1,0,0,0)))
    
    # Stacking Higgs pair inputs, each with shape (6,1), with randomly chosen non-Higgs pair inputs
    x = np.append(x, np.concatenate((HX_input, HY1_input, HY2_input, input_1, input_2, input_3)))

#     print("So far so good!")
#     break

# print('mjj',mjj)
# print('x',x)
# print('y',y)

x = x.reshape(int(len(x)/len(HX_input)), len(HX_input))
extra_bkgd_x = extra_bkgd_x.reshape(int(len(extra_bkgd_x)/len(HX_input)), len(HX_input))

print(count, nevt, count/nevt*1.0)

np.savez("Reco_Inputs/nn_input_MX700_MY400_class_reco", x=x,  y=y,  mjj=mjj, extra_bkgd_x=extra_bkgd_x, extra_bkgd_mjj=extra_bkgd_mjj, extra_bkgd_y = extra_bkgd_y, params=params, random_selection=random_selection)
