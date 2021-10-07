"""
This program trains a neural network with hyperparameters that are read from a
user-specified configuration file. The user may also provide a hyperparameter
and value to override the configuration file.
"""

import sys

print("Loading libraries. May take a few minutes.")

import numpy as np
import os
import sys
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
# version_utils.should_use_v2(True)
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress Keras/TF warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # suppress Keras/TF warnings

# Custom libraries and modules
from colors import CYAN, W
from logger import info, error

# from utils.analysis import TrainSix, TrainTwo
from utils.train.build import TrainSix
import awkward as ak
import numpy as np
import vector
vector.register_awkward()
from utils.models.save import ModelSaver
from lbn import LBN
lbn = LBN(11, 5, boost_mode=LBN.PRODUCT)

print("Libraries loaded.")

def p4(pt, eta, phi, m):
    return vector.obj(pt=pt, eta=eta, phi=phi, m=m)

filename = 'inputs/NMSSM_XYH_YToHH_6b_MX_700_MY_400_training_set_small_batch.root'
training = TrainSix(filename)
signal = training.correct_mask
cbkgd  = training.incorrect_mask
jet_pt = training.get_t6('jet_pt')
jet_eta = training.get_t6('jet_eta')
jet_phi = training.get_t6('jet_phi')
jet_m = training.get_t6('jet_m')
jet_btag = training.get_t6('jet_btag')
sig_pt  = ak.unzip(ak.combinations(jet_pt[signal], 6))
sig_eta = ak.unzip(ak.combinations(jet_eta[signal], 6))
sig_phi = ak.unzip(ak.combinations(jet_phi[signal], 6))
sig_m   = ak.unzip(ak.combinations(jet_m[signal], 6))
sig_pt = [ak.flatten(pt) for pt in sig_pt]
sig_eta = [ak.flatten(pt) for pt in sig_eta]
sig_phi = [ak.flatten(pt) for pt in sig_phi]
sig_m = [ak.flatten(pt) for pt in sig_m]
sig_p4 = [p4(pt,eta,phi,m) for pt,eta,phi,m in zip(sig_pt,sig_eta,sig_phi,sig_m)]
signal_input = [[p4.E, p4.px, p4.py, p4.pz] for p4 in sig_p4]
signal_input = np.asarray(signal_input)
bkg_pt  = ak.unzip(ak.combinations(jet_pt[cbkgd], 6))
bkg_eta = ak.unzip(ak.combinations(jet_eta[cbkgd], 6))
bkg_phi = ak.unzip(ak.combinations(jet_phi[cbkgd], 6))
bkg_m   = ak.unzip(ak.combinations(jet_m[cbkgd], 6))
bkg_pt = [ak.flatten(pt) for pt in bkg_pt]
bkg_eta = [ak.flatten(pt) for pt in bkg_eta]
bkg_phi = [ak.flatten(pt) for pt in bkg_phi]
bkg_m = [ak.flatten(pt) for pt in bkg_m]
bkg_p4 = [p4(pt,eta,phi,m) for pt,eta,phi,m in zip(bkg_pt,bkg_eta,bkg_phi,bkg_m)]
bkgd_input = [[p4.E, p4.px, p4.py, p4.pz] for p4 in bkg_p4]
bkgd_input = np.asarray(bkgd_input)
sig_btag = jet_btag[signal].to_numpy()
bkg_btag = jet_btag[cbkgd].to_numpy()
bkgd_lbn_features = lbn(np.transpose(bkgd_input, (2,0,1)), features=["E","px","py","pz"])
bkgd_lbn_features = tf.concat([bkgd_lbn_features, bkg_btag], axis=1)
signal_lbn_features = lbn(np.transpose(signal_input, (2,0,1)), features=["E","px","py","pz"])
signal_lbn_features = tf.concat([signal_lbn_features, sig_btag], axis=1)

features = tf.concat([signal_lbn_features, bkgd_lbn_features], axis=0)
targets = np.concatenate((
    np.tile([1,0], (signal_lbn_features.get_shape()[0], 1)),
    np.tile([0,1], (bkgd_lbn_features.get_shape()[0], 1))
))

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--cfg', dest = 'cfg', help = 'config file' , required = True )
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



### ------------------------------------------------------------------------------------
## 

# try: features.numpy()
# except: features.eval()
# finally: sys.exit()

# tf.disable_v2_behavior()
# with sess.as_default():
    # print(type(features.eval()))
# try:
#     from tensorflow.keras.layers.experimental import preprocessing
#     normalizer = preprocessing.Normalization()
#     normalizer.adapt(features)
#     x = normalizer(features)
# except:
features = features.numpy()
scaler = MinMaxScaler()
scaler.fit(features)
x = scaler.transform(features)
    
print(x)
print(targets)


# MinMaxScaler and tensorflow tensors don't play well together
# so preceding lines were ousted for this line
# x = tf.linalg.normalize(features)
# x = tf.keras.utils.normalize(features, axis=-1, order=2)

# norm_layer = tf.keras.layers.LayerNormalization(axis=1)
# x = norm_layer.build(features)
# x = norm_layer(features)

val_size = 0.10
x_train, x_val, y_train, y_val = train_test_split(x, targets, test_size=val_size)
param_dim = int(np.shape(features)[1])

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
        # model.add(Dense(int(nodes[i]), activation=hidden_activations, kernel_constraint=max_norm(1.0), kernel_regularizer=l1_l2(), bias_constraint=max_norm(1.0)))
        model.add(Dense(int(nodes[i]), kernel_initializer='lecun_normal', activation=hidden_activations, kernel_constraint=max_norm(1.0), kernel_regularizer=l1_l2(), bias_constraint=max_norm(1.0)))
        if bool(args.dropout): model.add(AlphaDropout(0.2)) 
    elif args.task == 'regressor':
        # model.add(Dense(int(nodes[i]), activation=hidden_activations, kernel_regularizer=l1_l2()))
        model.add(Dense(int(nodes[i]), kernel_initializer='lecun_normal', activation=hidden_activations, kernel_regularizer=l1_l2()))

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
    "param1_n24" : "lbn_inputs",
    # "param2_n6" : "jet_eta",
    # "param3_n6" : "jet_phi",
    "param4_n6" : "jet_btag",
    # "param5_n3" : "boosted_pt",
    # "param5_n3" : "delta_R",
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