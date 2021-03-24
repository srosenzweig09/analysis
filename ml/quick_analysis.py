import numpy as np
from argparse import ArgumentParser
from keras.models import model_from_json
from tensorflow import compat
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR) # suppress Keras/TF warnings
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress Keras/TF warnings
from logger import info, error

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')
parser.add_argument('--task', dest = 'task', help = 'class or reg?'           , default = 'class' )
parser.add_argument('--run' , dest = 'run' , help = 'training sesh index'     , default = 1 )
parser.add_argument('--type', dest = 'type', help = 'parton, smeared, or reco', default = 'reco')

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Load training examples

if args.type == 'reco':
    examples = np.load(f'inputs/Reco_Inputs/nn_input_MX700_MY400_{args.task}.npz')


x_test = examples['x_test']
y_test = examples['y_test']

if args.task == 'class':
    y_test = y_test[:,1]

### ------------------------------------------------------------------------------------
## Load model

if args.task == 'reg':
    task = 'regressor'
elif args.task == 'class':
    task = 'classifier'

model_dir = f'models/{task}/{args.type}/model/'
# load json and create model
model_json_file = open(model_dir + f'model_{args.run}.json', 'r')
model_json = model_json_file.read()
model_json_file.close()
model = model_from_json(model_json)

# load weights into new model
model.load_weights(model_dir + f'model_{args.run}.h5')

### ------------------------------------------------------------------------------------
## Obtain and print scores

scores = model.predict(x_test)
np.savez(f'models/{task}/{args.type}/scores_{args.run}.npz', scores=scores, target=y_test)

print(" "*10 + "Score" + " "*10 + "Target" + " "*18 + "Difference")
print("-"*60)
for i,(s,y) in enumerate(zip(scores, y_test)):
    print(" "*6 + f"{s}" + "  \t" + f"{y}" + "\t" + f"{y-s}")
    if i > 10: break