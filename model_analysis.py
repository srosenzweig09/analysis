print("Importing argparse")
from argparse import ArgumentParser
print("Importing keras model (from json)")
from keras.models import model_from_json
print("Importing numpy")
import numpy as np

from logger import info

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')
parser.add_argument('--type'   , dest = 'type'   , help = 'reco, parton, smeared'   ,  default = 'reco')
parser.add_argument('--task'   , dest = 'task'   , help = 'class or reg'            ,  default = 'classifier')
parser.add_argument('--run', dest = 'run', help = 'number of models trained',  default = 1    , type = int)

args = parser.parse_args()

## LOAD MODEL
model_dir = f'models/{args.task}/{args.type}/model/'
json_file = open(model_dir + f'model_{args.run}.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_dir + f'model_{args.run}.h5')

## LOAD INPUTS
f = np.load("inputs/Reco_Inputs/nn_input_MX700_MY400_class.npz")
nevents = len(np.concatenate((f['test'], f['train'], f['val'])))

x_test = f['x_test']

ntest = len(f['test'])

scores = model.predict(x_test)
scores = np.around(scores, decimals=3)

## APPLY MODEL TO INPUTS
pairings = []
for i in range(15):
    pairings.append(scores[np.arange(i, ntest, 15),1])

events = []
for i in range(0, ntest*15, 15):
    events.append(scores[i:i+15,1])