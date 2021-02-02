"""
This program trains a neural network with hyperparameters that are read from a
user-specified configuration file. The user may also provide a hyperparameter
and value to override the configuration file.

Command line arguments:
    1. path/to/config_file.cfg
    2. run_code (used to identify iterative runs with the same hyperparameters)
    3. hyperparameter_to_modify
    4. hyperparameter_value
"""

from sys import argv
from os import path, makedirs
import configparser

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
from keras.callbacks import EarlyStopping
from tensorflow import compat
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR) # suppress Keras/TF warnings

import numpy as np
from pandas import DataFrame
from inputprocessor import InputProcessor

def info(string):
    print("nntrainer.py -- [INFO] -- " + string)
def error(string):
    print("nntrainer.py !! [ERROR] !! " + string)

# Check for required user-provided input.
try:
    config_file              = argv[1]
    run_code                 = argv[2]
    info(f"Configuration file: {config_file}")
    info(f"Current run number: {run_code}")
except IndexError:
    error("No configuration file and/or run code specified.")
except Exception as err:
    print(err)

# Check for optional user-provided input.
# Used if user desires to change the value of a hyperparameter.
if len(argv) > 3:
    hyperparameter_to_modify = argv[3]
    hyperparameter_value     = argv[4]
    info(f"Modifying hyperparameter: {hyperparameter_to_modify}")
    info(f"New hyperparameter value: {hyperparameter_value}")

# Use 'test' as the input for run_code 
if run_code != 'test':
    
    # Primary folder in first line
    # Seconday folder in second line
    save_loc = (f'Experiments/{hyperparameter_to_modify}/' +
                f'{hyperparameter_to_modify}_{hyperparameter_value}/')
    info(f"Output will be saved in {save_loc}")

    # If directory does not exist, create it
    if not path.exists(save_loc):
        makedirs(save_loc)
        info(f"Creating folder {save_loc}")
else:

    # If running a test, throw output in quick_test folder
    save_loc = 'quick_test/'
    info(f"Running test script! Output will be saved in {save_loc}")

# Load the configuration using configparser
config = configparser.ConfigParser()
config.optionxform = str
config.read(config_file)

# Hidden hyperparameters
activation         = config['HIDDEN']['HiddenActivations']
nodes              = config['HIDDEN']['Nodes'].split('\n')
num_hidden         = len(nodes)
hidden_activations = ['selu'] * num_hidden

# Output hyperparameters
output_activation = config['OUTPUT']['OutputActivation']
output_nodes      = int(config['OUTPUT']['OutputNodes'])

# Fitting hyperparameters
optimizer     = config['OPTIMIZER']['OptimizerFunction']
loss_function = config['LOSS']['LossFunction']
num_epochs    = int(config['TRAINING']['NumEpochs'])
batch_size    = int(config['TRAINING']['BatchSize'])
    
inputs_filename = config['INPUTS']['InputFile']
info(f"Loading inputs from file: {inputs_filename}")

# Load training examples
inputs = InputProcessor(inputs_filename)
inputs.normalize_inputs(save_loc=save_loc)
inputs.split_input_examples(save_loc=save_loc, run_code=run_code)

# Split training examples into training, testing, and validation sets
x_train, x_test, x_val = inputs.get_x()
y_train, y_test, y_val = inputs.get_y()

param_dim = inputs.param_dim
input_nodes = int(inputs.param_dim*1.2)

if len(argv) > 3:
    # Change value of hyperparameter
    hyperparam_dict = { 'num_hidden':num_hidden-1,
                        'batch_size':batch_size,
                        'num_epochs':num_epochs,
                        'input_nodes':input_nodes
                        }

    try:
        hyperparam_dict[hyperparameter_to_modify] = int(hyperparameter_value)
    except KeyError:
        error("Incorrect key input into hyperparameter dictionary!")
        error(f"Available keys are: {hyperparam_dict.keys()}")



# Define the keras model
model = Sequential()

# Input layers
model.add(Dense(input_nodes, input_dim=param_dim, activation=activation))

# # Hidden layers
for i in range(num_hidden):
    model.add(Dense(int(nodes[i]), activation=hidden_activations[i]))

# Output layer
model.add(Dense(output_nodes, activation=output_activation))

# Optimizer
optimizer = optimizers.Nadam()

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

lr_metric = get_lr_metric(optimizer)

# Stop after epoch in which loss no longer decreases but save the best model.
es = EarlyStopping(monitor='loss', restore_best_weights=True)

model.compile(loss=loss_function, 
              optimizer=optimizer, 
              metrics=['accuracy', lr_metric])

print()
info("Preparing to fit the model!\n")
info(f"Training examples have {param_dim} features.")
info(f"Model has {num_hidden} hidden layers.")
info(f"Hidden layers have nodes: {nodes}")

# fit the keras model on the dataset
history = model.fit(x_train,
                    y_train, 
                    validation_data=(x_val, y_val), 
                    epochs=num_epochs, 
                    batch_size=batch_size, 
                    callbacks=[es])

# Record number of last epoch
nepochs = len(history.history['accuracy'])

# convert the history.history dict to a pandas DataFrame   
hist_df = DataFrame(history.history) 

# Save to json:  
hist_json_file = save_loc + 'history_' + run_code + '.json' 

with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# Save model to json and weights to h5
model_json = model.to_json()

json_save = save_loc + "model_" + run_code + ".json"
h5_save   = json_save.replace(".json", ".h5")

with open(json_save, "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(h5_save)

info(f"Saved model and history to disk in location:"
      f"\n   {json_save}\n   {h5_save}\n   {hist_json_file}")

h_act = ''
for funcs in hidden_activations:
    h_act = h_act + funcs + ', '

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
    f"Input parameters:            {param_dim}, " +
        f"{inputs.inputs['params'][:param_dim]}\n",
    f"Optimizer:                   {optimizer}\n",
    f"Loss:                        {loss_function}\n",
    f"Num epochs:                  {nepochs}\n",
    f"Batch size:                  {batch_size}\n",
    f"Num hidden layers:           {hyperparameter_value}\n",
    f"Num nodes in  1st hidden:    {input_nodes}\n",
    f"Input activation function:   {activation}\n",
    f"Hidden layer nodes:          {nodes[:num_hidden-1]}\n",
    f"Hidden activation functions: {h_act[:-2]}\n",
    f"Num output nodes:            {output_nodes}\n",
    f"Output activation function:  {output_activation}"]

for info in nn_info_list:
    print(info)

with open(save_loc + 'nn_info.txt', "w") as f:
    for line in nn_info_list:
        f.writelines(line)
