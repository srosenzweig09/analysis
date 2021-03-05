"""
This program trains a neural network with hyperparameters that are read from a
user-specified configuration file. The user may also provide a hyperparameter
and value to override the configuration file.
"""

import numpy as np
from pandas import DataFrame
from configparser import ConfigParser
from argparse import ArgumentParser
from sys import argv
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2, l1_l2
from keras.constraints import max_norm
from tensorflow import compat
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress Keras/TF warnings
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR) # suppress Keras/TF warnings

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

# Custom libraries and modules
from inputprocessor import InputProcessor
from logger import info, error

print()


### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')
parser.add_argument('--tag'       , dest = 'tag'       , help = 'production tag'                      ,  required = True                               )
parser.add_argument('--type'      , dest = 'type'      , help = 'parton, smeared, or reco'            ,  required = True                               )
parser.add_argument('--run'       , dest = 'run'       , help = 'index of current training session'   ,  required = True       ,  type = int           )
parser.add_argument('--nlayers'   , dest = 'nlayers'   , help = 'number of hidden layers'             ,  default = 4           ,  type = int           )
parser.add_argument('--DeltaR'    , dest = 'DeltaR'    , help = 'include DeltaR in features'          ,  action='store_true'   ,  default = False      )
parser.add_argument('--pTprod'    , dest = 'pTprod'    , help = 'include product of pT in features'   ,  action='store_true'   ,  default = False      )
parser.add_argument('--twoclass'    , dest = 'twoclass'    , help = 'two class classifier'   ,  action='store_true'   ,  default = False      )
parser.add_argument('--outdir'    , dest = 'outdir'    , help = 'output directory'   ,    required = True      )
parser.add_argument('--reg'    , dest = 'reg'    , help = 'regularizer? (l1, l2, l1_l2)'   ,    default = None      )
parser.add_argument('--9feat'    , dest = 'feat9'    , help = '9 features (pt^2)'   ,    action = 'store_true', default=False      )

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Prepare output directories

input_dir = f"{args.outdir}/{args.tag}/"
if args.feat9:
    input_dir = f"{args.outdir}/{args.tag}_9feat/"
model_dir = input_dir + "model/"
eval_dir = input_dir + "evaluation/"

info(f"Evaluating models with {args.tag} hidden layers from location {input_dir}")

if not os.path.exists(input_dir):
    os.makedirs(input_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

### ------------------------------------------------------------------------------------
## Import configuration file

layers = 'layers'
if args.nlayers == 1: layers = 'layer'

assert (args.type == 'parton') or (args.type == 'smeared') or (args.type == 'reco'), "--type must be 'parton', 'smeared', or 'reco'!"

cfg_location = 'Config/'
cfg_name = f'nn_hyperparam_{args.nlayers}{layers}_{args.type}.cfg'

cfg = cfg_location + cfg_name

info(f"Loading configuration file from {cfg}")

# Load the configuration using configparser
config = ConfigParser()
config.optionxform = str
config.read(cfg)

# Hidden hyperparameters
hidden_activations = config['HIDDEN']['HiddenActivations']
nodes              = config['HIDDEN']['Nodes'].split('\n')
nlayers            = len(nodes)

# Output hyperparameters
output_activation  = config['OUTPUT']['OutputActivation']
output_nodes       = int(config['OUTPUT']['OutputNodes'])

# Fitting hyperparameters
optimizer          = config['OPTIMIZER']['OptimizerFunction']
loss_function      = config['LOSS']['LossFunction']
nepochs            = int(config['TRAINING']['NumEpochs'])
batch_size         = int(config['TRAINING']['BatchSize'])
    
inputs_filename    = config['INPUTS']['InputFile']

if args.twoclass:
    c_ind = inputs_filename.find('class')
    inputs_filename = inputs_filename[:c_ind] + 'two_' + inputs_filename[c_ind:]
if args.feat9:
    c_ind = inputs_filename.find('.npz')
    inputs_filename = inputs_filename[:c_ind] + '_9features' + inputs_filename[c_ind:]

info(f"Loading inputs from file: {inputs_filename}")

if args.nlayers:
    nlayers = args.nlayers

### ------------------------------------------------------------------------------------
## 

# Load training examples
inputs = InputProcessor(inputs_filename, include_pt1pt2=args.pTprod, include_DeltaR=args.DeltaR)
inputs.normalize_inputs(run=args.run, model_dir=model_dir)
n_events = np.shape(inputs.X)[0] / 6
indexing = np.arange(0, n_events)

# test_size = 0.20
# val_size = 0.125

# if ~args.twoclass:
inputs.split_input_examples()
x_train, x_test, x_val = inputs.get_x()
y_train, y_test, y_val = inputs.get_y()

param_dim = inputs.param_dim

### ------------------------------------------------------------------------------------
## 

info("Defining the model.")
# Define the keras model
model = Sequential()

# Input layers
model.add(Dense(nodes[0], input_dim=param_dim, activation=hidden_activations))

# # Hidden layers
for i in range(1,nlayers):
    model.add(Dense(int(nodes[i]), activation=hidden_activations, kernel_constraint=max_norm(1.0), kernel_regularizer=l1_l2(), bias_constraint=max_norm(1.0)))

# Output layer
if args.twoclass: 
    output_nodes = 2
    output_activation = 'softmax'
model.add(Dense(output_nodes, activation=output_activation))

# Stop after epoch in which loss no longer decreases but save the best model.
es = EarlyStopping(monitor='loss', restore_best_weights=True)

info("\nCompiling the model.")
model.compile(loss=loss_function, 
              optimizer=optimizer, 
              metrics=['accuracy'])

### ------------------------------------------------------------------------------------
## 

print()
info("Preparing to fit the model!\n")
info(f"Training examples have {param_dim} features.")
info(f"Model has {nlayers} hidden layer(s).")
info(f"Hidden layer(s) have nodes: {nodes}\n")

# fit the keras model on the dataset
history = model.fit(x_train,
                    y_train, 
                    validation_data=(x_val, y_val), 
                    epochs=nepochs, 
                    batch_size=batch_size, 
                    callbacks=[es])

### ------------------------------------------------------------------------------------
## 

# convert the history.history dict to a pandas DataFrame   
hist_df = DataFrame(history.history) 

# Save to json:  
hist_json_file = model_dir + f'history_{args.run}.json' 

with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# Save model to json and weights to h5
model_json = model.to_json()

json_save = model_dir + f"model_{args.run}.json"
h5_save   = json_save.replace(".json", ".h5")

with open(json_save, "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(h5_save)

info(f"Saved model and history to disk in location:"
      f"\n   {json_save}\n   {h5_save}\n   {hist_json_file}")


### ------------------------------------------------------------------------------------
## 



# Make a list of the hyperparameters and print them to the screen.
nn_info_list = [
    f"Training set:                " +
        f"{int(len(x_train)/len(inputs.X)*100):d}%, " +
        f"{int(len(x_train)):d}\n",
    f"Validation set:              " +
        f"{int(len(x_val)/len(inputs.X)*100):d}%, " +
        f"{int(len(x_val)):d}\n",
    f"Testing set:                 " +
        f"{int(len(x_test)/len(inputs.X)*100):d}%, " +
        f"{int(len(x_test)):d}\n",
    f"Class 0 (Non-Higgs  Pair):   " +
        f"{np.sum(inputs.y == 0)/len(inputs.y)*100:.0f}%, " +
        f"{np.sum(inputs.y == 0):d}\n",
    f"Class 1 (Higgs Pair):        " +
        f"{np.sum(inputs.y == 1)/len(inputs.y)*100:.0f}%, " +
        f"{np.sum(inputs.y == 1):d}\n",
    f"Input parameters:            {param_dim},\n" +
        f"                             " + 
        f"{inputs.params}\n",
    f"Optimizer:                   {optimizer}\n",
    f"Loss:                        {loss_function}\n",
    f"Num epochs:                  {nepochs}\n",
    f"Batch size:                  {batch_size}\n",
    f"Num hidden layers:           {nlayers}\n",
    f"Input activation function:   {hidden_activations}\n",
    f"Hidden layer nodes:          {nodes}\n",
    f"Hidden activation functions: {hidden_activations}\n",
    f"Num output nodes:            {output_nodes}\n",
    f"Output activation function:  {output_activation}"]

for info in nn_info_list:
    print(info)


with open(input_dir + 'nn_info.txt', "w") as f:
    for line in nn_info_list:
        f.writelines(line)
