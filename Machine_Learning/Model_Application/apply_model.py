import os
from keras.models import model_from_json
from kinematics import calcDeltaR
import uproot3
import numpy as np
from awkward0 import Table
from pickle import load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress Keras/TF warnings
from argparse import ArgumentParser

from logger import info

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--tag'       , dest = 'tag'       , help = 'production tag'                      ,  required = True               )
parser.add_argument('--nlayers'   , dest = 'nlayers'   , help = 'number of hidden layers'             ,  required = False , type = int )
parser.add_argument('--run'       , dest = 'run'       , help = 'index of current training session'   ,  required = True , type = int  )
parser.add_argument('--MX'       , dest = 'MX'       , help = 'mass of X resonance'   ,               ,  required = False , type = int , default = 700 ) # GeV
parser.add_argument('--MY'       , dest = 'MY'       , help = 'mass of Y resonance'   ,               ,  required = False , type = int , default = 400 ) # GeV

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Load scaler, as well as model (json and h5)

model_dir = f'../Model_Training/Experiments/layers/layers_{args.nlayers}/{args.tag}/model/'

json_file = open(model_dir + f'model_{args.run}.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_dir + f'model_{args.run}.h5')

scaler = load(open(model_dir + 'scaler.pkl', 'rb'))

### ------------------------------------------------------------------------------------
## Open signal event ROOT folder

filename = '/eos/user/s/srosenzw/SWAN_projects/sixB/Signal_Exploration/Mass_Pair_ROOT_files/X_YH_HHH_6b_MX700_MY400.root'
f = uproot3.open(filename)
tree = f['sixbntuplizer/sixBtree']
branches = tree.arrays(namedecode='utf-8')
table = Table(branches)

HX_b1 = {'pt':table['gen_HX_b1_pt'], 'eta':table['gen_HX_b1_eta'], 'phi':table['gen_HX_b1_phi'], 'm':table['gen_HX_b1_m']}
HX_b2 = {'pt':table['gen_HX_b2_pt'], 'eta':table['gen_HX_b2_eta'], 'phi':table['gen_HX_b2_phi'], 'm':table['gen_HX_b2_m']}
HY1_b1 = {'pt':table['gen_HY1_b1_pt'], 'eta':table['gen_HY1_b1_eta'], 'phi':table['gen_HY1_b1_phi'], 'm':table['gen_HY1_b1_m']}
HY1_b2 = {'pt':table['gen_HY1_b2_pt'], 'eta':table['gen_HY1_b2_eta'], 'phi':table['gen_HY1_b2_phi'], 'm':table['gen_HY1_b2_m']}
HY2_b1 = {'pt':table['gen_HY2_b1_pt'], 'eta':table['gen_HY2_b1_eta'], 'phi':table['gen_HY2_b1_phi'], 'm':table['gen_HY2_b1_m']}
HY2_b2 = {'pt':table['gen_HY2_b2_pt'], 'eta':table['gen_HY2_b2_eta'], 'phi':table['gen_HY2_b2_phi'], 'm':table['gen_HY2_b2_m']}
part_dict = {0:HX_b1, 1:HX_b2, 2:HY1_b1, 3:HY1_b2, 4:HY2_b1, 5:HY2_b2}

### ------------------------------------------------------------------------------------
## Obtain predictions for each pairing (15 possible distinct)

predictions = np.array(())
for i in range(5):
    for j in range(i+1, 6):
        X = np.transpose(np.array((part_dict[i]['pt'], part_dict[i]['eta'], part_dict[i]['phi'], part_dict[j]['pt'], part_dict[j]['eta'], part_dict[j]['phi'], part_dict[i]['pt']*part_dict[j]['pt'], calcDeltaR(part_dict[i]['eta'], part_dict[j]['eta'], part_dict[i]['phi'], part_dict[j]['phi']))))

        x = scaler.transform(X)

        predictions = np.append(predictions, loaded_model.predict(x))

predictions = np.transpose(predictions.reshape(15, 100000))

np.savez(f'signal_predictions_layers{args.nlayers}_{args.tag}.npz', p=predictions)
