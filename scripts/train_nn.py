"""
This program trains a neural network with hyperparameters that are read from a
user-specified configuration file. The user may also provide a hyperparameter
and value to override the configuration file.
"""

print("Loading libraries. May take a few minutes.")

import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from argparse import ArgumentParser
from configparser import ConfigParser
from keras.models import Sequential
from keras.layers import Activation, Dense, AlphaDropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2, l1_l2
from keras.constraints import max_norm
from keras.utils.generic_utils import get_custom_objects
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import compat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress Keras/TF warnings
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR) # suppress Keras/TF warnings

# Custom libraries and modules
from colors import CYAN, W
from logger import info, error

from utils.analysis import TrainSix, TrainTwo
from utils.models.save import ModelSaver

print("Libraries loaded.")
print()


### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--cfg', dest = 'cfg', help = 'config file' , required = True )
parser.add_argument('--filelist', dest = 'filelist', help = 'tag for reading list of files from .txt file', action = 'store_true', default = False )
parser.add_argument('--task', dest = 'task', help = 'classifier or regressor'  , default = 'classifier' )
parser.add_argument('--njet', dest = 'njet', help = 'how many input jets'      , default = 6 )
parser.add_argument('--tag' , dest = 'tag' , help = 'special tag', default = None)
parser.add_argument('--lr' , dest = 'lr' , help = 'learning rate', default = 0.001)
parser.add_argument('--beta1' , dest = 'beta1' , help = 'exponential decay rate for first moment', default = 0.9)
parser.add_argument('--beta2' , dest = 'beta2' , help = 'beta_2 value', default = 0.999)
parser.add_argument('--epsilon' , dest = 'epsilon' , help = 'epsilon value', default = 1e-07)
parser.add_argument('--dropout', dest = 'dropout', help='implement dropout boolean', action='store_true', default=False)
parser.add_argument('--dijet', dest = 'dijet', help='implement dijet boolean', action='store_true', default=False)

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Import configuration file

cfg = args.cfg

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

optimizer_name = config['OPTIMIZER']['OptimizerFunction']
loss_function      = config['LOSS']['LossFunction']
nepochs            = int(config['TRAINING']['NumEpochs'])
batch_size         = int(config['TRAINING']['BatchSize'])
    
inputs_filename    = config['INPUTS']['InputFile']

nn_type            = config['TYPE']['Type']


#####
if int(args.njet) == 6:
    training = TrainSix(inputs_filename, dijet=bool(args.dijet))
    inputs = training.features
    targets = training.targets
elif int(args.njet) == 2:
    training = TrainTwo(inputs_filename)
    inputs = training.features
    targets = training.targets

print(f"Inputs shape:  {inputs.shape}")
print(f"Targets shape: {targets.shape}")

print(inputs[0,:])
print(targets[0,:])

### ------------------------------------------------------------------------------------
## 

scaler = MinMaxScaler()
scaler.fit(inputs)
x = scaler.transform(inputs)

val_size = 0.10
x_train, x_val, y_train, y_val = train_test_split(x, targets, test_size=val_size)

param_dim = int(np.shape(inputs)[1])

### ------------------------------------------------------------------------------------
## 

# Add the GELU function to Keras
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'gelu': Activation(gelu)})

info("Defining the model.")
# Define the keras model
model = Sequential()

# Input layers
model.add(Dense(nodes[0], input_dim=param_dim, activation=hidden_activations))

# # Hidden layers
for i in range(1,nlayers):
    if 'classifier' in args.task:
        model.add(Dense(int(nodes[i]), activation=hidden_activations, kernel_constraint=max_norm(1.0), kernel_regularizer=l1_l2(), bias_constraint=max_norm(1.0)))
        # model.add(Dense(int(nodes[i]), kernel_initializer='lecun_normal', activation=hidden_activations, kernel_constraint=max_norm(1.0), kernel_regularizer=l1_l2(), bias_constraint=max_norm(1.0)))
        if bool(args.dropout): model.add(AlphaDropout(0.2)) 
    elif args.task == 'regressor':
        model.add(Dense(int(nodes[i]), activation=hidden_activations, kernel_regularizer=l1_l2()))
        # model.add(Dense(int(nodes[i]), kernel_initializer='lecun_normal', activation=hidden_activations, kernel_regularizer=l1_l2()))

# Output layer
model.add(Dense(output_nodes, activation=output_activation))

# Stop after epoch in which loss no longer decreases but save the best model.
es = EarlyStopping(monitor='loss', restore_best_weights=True, patience=10)

if 'classifier' in args.task:
    met = ['accuracy']
elif args.task == 'regressor':
    met = None

# Nadam defaults:
# learning_rate = 0.001
# beta_1 = 0.9
# beta_2 = 0.999
# optimizer = tf.keras.optimizers.Nadam(learning_rate=float(args.lr), beta_1=float(args.beta1), beta_2=float(args.beta2), epsilon=float(args.epsilon), name="Nadam")
optimizer = 'nadam'

info("Compiling the model.")
model.compile(loss=loss_function, 
              optimizer=optimizer, 
              metrics=met)

# Make a list of the hyperparameters and print them to the screen.
nn_info_list = [
    f"Input parameters:            {param_dim}\n",
    f"Optimizer:                   {optimizer}\n",
    f"Learning Rate:               {args.lr}\n",
    f"beta_1:                      {args.beta1}\n",
    f"beta_2:                      {args.beta2}\n",
    f"epsilon:                     {args.epsilon}\n",
    f"Loss:                        {loss_function}\n",
    f"Num epochs:                  {nepochs}\n",
    f"Batch size:                  {batch_size}\n",
    f"Num hidden layers:           {nlayers}\n",
    f"Input activation function:   {hidden_activations}\n",
    f"Hidden layer nodes:          {nodes}\n",
    f"Hidden activation functions: {hidden_activations}\n",
    f"Num output nodes:            {output_nodes}\n",
    f"Output activation function:  {output_activation.capitalize()}"]

for line in nn_info_list:
    print(line)

### ------------------------------------------------------------------------------------
## Fit the model

print()
info("Preparing to fit the model!\n")
info(f"Training with {len(x_train)} examples")

# fit the keras model on the dataset
history = model.fit(x_train,
                    y_train, 
                    validation_data=(x_val, y_val), 
                    epochs=nepochs, 
                    batch_size=batch_size, 
                    callbacks=[es])

model.summary()

### ------------------------------------------------------------------------------------
## Save the model, history, and predictions

# convert the history.history dict to a pandas DataFrame   
hist_df = DataFrame(history.history) 

out_dir = f"{args.njet}jet_{args.task}/models/{args.tag}/"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

info(f"Training sessions will be saved in {out_dir}")

saver = ModelSaver(out_dir, model, hist_df, scaler)

### ------------------------------------------------------------------------------------
## Write model configuration file

cfg_out = ConfigParser()

### FIX THIS TO PROGRAMATICALLY ASSIGN PARAMETER NAMES
cfg_out["model"] = {
    "input_parameters" : param_dim,
    "param1_n6" : "jet_pt",
    "param2_n6" : "jet_eta",
    "param3_n6" : "jet_phi",
    "param4_n6" : "jet_btag",
    "param5_n3" : "boosted_pt",
    "param5_n3" : "delta_R",
    "num_hidden_layers" : nlayers,
    "input_activation_function" : hidden_activations,
    "hidden_layer_nodes" : ",".join(nodes),
    "hidden_activation_function" : hidden_activations,
    "num_output_nodes" : output_nodes,
    "output_activation_function" : output_activation.capitalize()
}

cfg_out["training"] = {
    "optimizer" : optimizer_name,
    "learning_rate" : args.lr,
    "num_epochs" : len(hist_df),
    "beta_1" : args.beta1,
    "beta_2" : args.beta2,
    "epsilon" : args.epsilon,
    "loss_function" : loss_function,
    "batch_size" : batch_size
}

cfg_out["scaler"] = {
    "scale_min" : ",".join(scaler.data_min_.astype('str').tolist()),
    "scale_max" : ",".join(scaler.data_max_.astype('str').tolist())
}

with open(out_dir + 'model.cfg', "w") as conf:
    cfg_out.write(conf)

print("-"*45 + CYAN + " Training ended " + W + "-"*45)