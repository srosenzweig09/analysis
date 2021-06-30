import os
from keras.models import model_from_json
import numpy as np
import uproot3_methods
from pickle import load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress Keras/TF warnings
from argparse import ArgumentParser

from logger import info
from kinematics import calcDeltaR
from myuproot import get_uproot_Table
from colors import CYAN, W

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--task', dest = 'task', help = 'class or reg'              , required = True )
parser.add_argument('--type', dest = 'type', help = 'reco, parton, smeared'     , required = True )
parser.add_argument('--run' , dest = 'run' , help = 'index of training session' , default = 1     )
parser.add_argument('--MX'  , dest = 'MX'  , help = 'mass of X resonance'       , default = 700   , type = int ) # GeV
parser.add_argument('--MY'  , dest = 'MY'  , help = 'mass of Y resonance'       , default = 400   , type = int ) # GeV

args = parser.parse_args()


### ------------------------------------------------------------------------------------
## Load scaler, as well as model, json and h5

MX = args.MX
MY = args.MY

if args.task == 'class':
    task = 'classifier'
if args.task == 'reg':
    task = 'regressor'

if args.type == 'reco':
    prefix = 'Reco'
    tag = '_recojet'
elif args.type == 'parton':
    prefix = 'Gen'
    tag = ''

### ------------------------------------------------------------------------------------
## Open signal event ROOT folder

model_dir = f'Sessions/{task}/{args.type}/model/'
scaler_file = f'Inputs/{prefix}_Inputs/nn_input_MX700_MY400_{args.task}_scaler.pkl'
examples = np.load(f'Inputs/{prefix}_Inputs/nn_input_MX700_MY400_{args.task}.npz')
test_evts = examples['test']

reco_filename = f'Inputs/NanoAOD/NMSSM_XYH_YToHH_6b_MX_{MX}_MY_{MY}_accstudies.root'
reco_filename2 = f'Inputs/NanoAOD/NMSSM_XYH_YToHH_6b_MX_{MX}_MY_{MY}_accstudies_2.root'
info(f"Opening ROOT file {CYAN}{reco_filename}{W} with columns")
table =  get_uproot_Table(reco_filename, 'sixBtree')
info(f"Opening ROOT file {CYAN}{reco_filename2}{W} with columns")
table2 =  get_uproot_Table(reco_filename2, 'sixBtree')
nevents = table._length() + table2._length()

json_file = open(model_dir + f'model_{args.run}.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_dir + f'model_{args.run}.h5')

scaler = load(open(scaler_file, 'rb'))


HX_b1 = {'pt':np.concatenate((table['gen_HX_b1_pt'], table2['gen_HX_b1_pt'])), 
        'eta':np.concatenate((table['gen_HX_b1_eta'], table2['gen_HX_b1_eta'])), 
        'phi':np.concatenate((table['gen_HX_b1_phi'], table2['gen_HX_b1_phi'])), 
        'm':np.concatenate((table['gen_HX_b1_m'], table2['gen_HX_b1_m']))}
HX_b2 = {'pt':np.concatenate((table['gen_HX_b2_pt'], table2['gen_HX_b2_pt'])), 
        'eta':np.concatenate((table['gen_HX_b2_eta'], table2['gen_HX_b2_eta'])), 
        'phi':np.concatenate((table['gen_HX_b2_phi'], table2['gen_HX_b2_phi'])), 
        'm':np.concatenate((table['gen_HX_b2_m'], table2['gen_HX_b2_m']))}
HY1_b1 = {'pt':np.concatenate((table['gen_HY1_b1_pt'], table2['gen_HY1_b1_pt'])), 
        'eta':np.concatenate((table['gen_HY1_b1_eta'], table2['gen_HY1_b1_eta'])), 
        'phi':np.concatenate((table['gen_HY1_b1_phi'], table2['gen_HY1_b1_phi'])), 
        'm':np.concatenate((table['gen_HY1_b1_m'], table2['gen_HY1_b1_m']))}
HY1_b2 = {'pt':np.concatenate((table['gen_HY1_b2_pt'], table2['gen_HY1_b2_pt'])), 
        'eta':np.concatenate((table['gen_HY1_b2_eta'], table2['gen_HY1_b2_eta'])), 
        'phi':np.concatenate((table['gen_HY1_b2_phi'], table2['gen_HY1_b2_phi'])), 
        'm':np.concatenate((table['gen_HY1_b2_m'], table2['gen_HY1_b2_m']))}
HY2_b1 = {'pt':np.concatenate((table['gen_HY2_b1_pt'], table2['gen_HY2_b1_pt'])), 
        'eta':np.concatenate((table['gen_HY2_b1_eta'], table2['gen_HY2_b1_eta'])), 
        'phi':np.concatenate((table['gen_HY2_b1_phi'], table2['gen_HY2_b1_phi'])), 
        'm':np.concatenate((table['gen_HY2_b1_m'], table2['gen_HY2_b1_m']))}
HY2_b2 = {'pt':np.concatenate((table['gen_HY2_b2_pt'], table2['gen_HY2_b2_pt'])), 
        'eta':np.concatenate((table['gen_HY2_b2_eta'], table2['gen_HY2_b2_eta'])), 
        'phi':np.concatenate((table['gen_HY2_b2_phi'], table2['gen_HY2_b2_phi'])), 
        'm':np.concatenate((table['gen_HY2_b2_m'], table2['gen_HY2_b2_m']))}
part_dict = {0:HX_b1, 1:HX_b2, 2:HY1_b1, 3:HY1_b2, 4:HY2_b1, 5:HY2_b2}



### ------------------------------------------------------------------------------------
## Obtain scores for each pairing (15 possible distinct)

ntest = len(test_evts)
inputs = []
counter = 1
scores = np.array(())

if args.task == 'class':
        
        for i in range(5):
                for j in range(i+1, 6):
                        info(f"Predicting pair {counter}/15")

                        X = np.column_stack((part_dict[i]['pt'], part_dict[i]['eta'], part_dict[i]['phi'], part_dict[j]['pt'], part_dict[j]['eta'], part_dict[j]['phi'], calcDeltaR(part_dict[i]['eta'], part_dict[j]['eta'], part_dict[i]['phi'], part_dict[j]['phi'])))[test_evts]

                        x = scaler.transform(X)
                        model_pred = loaded_model.predict(x)
                        scores = np.append(scores, model_pred[:,1])
                        inputs.append(X)
                        
                        counter+=1

        inputs = np.array((inputs))
        scores = np.transpose(scores.reshape(15, ntest))

else:
        H1_b1 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(HY1_b1['pt'], HY1_b1['eta'], HY1_b1['phi'], HY1_b1['m'])[test_evts]
        H1_b2 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(HY1_b2['pt'], HY1_b2['eta'], HY1_b2['phi'], HY1_b2['m'])[test_evts]

        H1 = H1_b1 + H1_b2

        H2_b1 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(HY2_b1['pt'], HY2_b1['eta'], HY2_b1['phi'], HY2_b1['m'])[test_evts]
        H2_b2 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(HY2_b2['pt'], HY2_b2['eta'], HY2_b2['phi'], HY2_b2['m'])[test_evts]

        H2 = H2_b1 + H2_b2

        HX_b1 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(HX_b1['pt'], HX_b1['eta'], HX_b1['phi'], HX_b1['m'])[test_evts]
        HX_b2 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(HX_b2['pt'], HX_b2['eta'], HX_b2['phi'], HX_b2['m'])[test_evts]

        HX = HX_b1 + HX_b2

        Y = H1 + H2
        y = Y.mass

        dR_Y1Y2 = calcDeltaR(H1.eta, H2.eta, H1.phi, H2.phi)
        dR_Y1X = calcDeltaR(H1.eta, HX.eta, H1.phi, HX.phi)
        dR_XY2 = calcDeltaR(HX.eta, H2.eta, HX.phi, H2.phi)

        X_true = np.column_stack((H1.pt, H1.eta, H1.phi, H2.pt, H2.eta, H2.phi, dR_Y1Y2, H1.mass, H2.mass))
        X_false1 = np.column_stack((H1.pt, H1.eta, H1.phi, HX.pt, HX.eta, HX.phi, dR_Y1X, H1.mass, HX.mass))
        X_false2 = np.column_stack((H2.pt, H2.eta, H2.phi, HX.pt, HX.eta, HX.phi, dR_XY2, HX.mass, H2.mass))

        x = scaler.transform(X_true)
        model_pred = loaded_model.predict(x)
        scores = np.append(scores, model_pred)
        inputs.append(X_true)

        x = scaler.transform(X_false1)
        model_pred = loaded_model.predict(x)
        scores = np.append(scores, model_pred)
        inputs.append(X_false1)

        x = scaler.transform(X_false2)
        model_pred = loaded_model.predict(x)
        scores = np.append(scores, model_pred)
        inputs.append(X_false2)

inputs = np.array((inputs))

print(inputs.shape)
print(scores.shape)

np.savez(f'Evaluations/{task}/scores_{args.run}.npz', scores=scores, X=inputs)
