# This code requires the user to plug in a command line argument 

from sys import argv
from os import path, makedirs
import configparser

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
from keras.callbacks import EarlyStopping
from tensorflow import compat
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)

import numpy as np
from pandas import DataFrame
from inputprocessor import InputProcessor



config_file              = argv[1]
run_code                 = argv[2]
hyperparameter_to_modify = argv[3]
hyperparameter_value     = argv[4]

save_loc = 'Experiments/' + hyperparameter_to_modify + '/' + hyperparameter_to_modify + '_' + hyperparameter_value + '_repeat/'
if not path.exists(save_loc):
    makedirs(save_loc)

config = configparser.ConfigParser()
config.optionxform = str
config.read(config_file)
print(f"----------[INFO] Opening configuration file:\n       {config_file}")

# Hidden hyperparameters
activation = config['HIDDEN']['HiddenActivations']
nodes      = config['HIDDEN']['Nodes'].split('\n')

hidden_activations = ['selu'] * len(nodes)

assert(len(nodes) == len(hidden_activations))

num_hidden = len(nodes)

# Output hyperparameters
output_activation = config['OUTPUT']['OutputActivation']
output_nodes      = int(config['OUTPUT']['OutputNodes'])

# Fitting hyperparameters
optimizer     = config['OPTIMIZER']['OptimizerFunction']
loss_function = config['LOSS']['LossFunction']
num_epochs    = int(config['TRAINING']['NumEpochs'])
batch_size    = int(config['TRAINING']['BatchSize'])
    
inputs_filename = config['INPUTS']['InputFile']
print(f"----------[INFO] Loading inputs from file:\n       {inputs_filename}")

inputs = InputProcessor(inputs_filename)
inputs.normalize_inputs(save_loc=save_loc)
inputs.split_input_examples(save_loc=save_loc, run_code=run_code)

x_train, x_val, x_test = inputs.get_x()
y_train, y_val, y_test = inputs.get_y()

param_dim = inputs.param_dim
input_nodes = int(inputs.param_dim*1.2)

# define the keras model
model = Sequential()

# Input layers
model.add(Dense(input_nodes, input_dim=param_dim, activation=activation))

# Hidden layers
for i in range(num_hidden):
    model.add(Dense(int(nodes[i]), activation=hidden_activations[i]))

# Output layer
model.add(Dense(output_nodes, activation=output_activation))

# modify and compile the keras model
optimizer = optimizers.Nadam()

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

lr_metric = get_lr_metric(optimizer)

es = EarlyStopping(monitor='loss', restore_best_weights=True)

model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy', lr_metric])

# fit the keras model on the dataset
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                    epochs=num_epochs, batch_size=batch_size, callbacks=[es])

# convert the history.history dict to a pandas DataFrame:     
hist_df = DataFrame(history.history) 

# save to json:  
hist_json_file = save_loc + 'history_' + run_code + '.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

model_json = model.to_json()
json_save = save_loc + "model_" + run_code + ".json"
h5_save = json_save.replace(".json", ".h5")
with open(json_save, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(h5_save)
print(f"----------[INFO] Saved model and history to disk in location:\n       {json_save}\n       {h5_save}\n       {hist_json_file}")

h_act = ''
for funcs in hidden_activations:
    h_act = h_act + funcs + ', '

print("Training set:                {:d}%, {:d}".format(int(len(x_train)/len(inputs.X)*100), int(len(x_train))))
print("Validation set:              {:d}%, {:d}".format(int(len(x_val)/len(inputs.X)*100), int(len(x_val))))
print("Testing set:                 {:d}%, {:d}".format(int(len(x_test)/len(inputs.X)*100), int(len(x_test))))
print("Class 0 (Non-Higgs  Pair):   {:.0f}%, {:d}".format(np.sum(inputs.y == 0)/len(inputs.y)*100, np.sum(inputs.y == 0)))
print("Class 1 (Higgs Pair):        {:.0f}%, {:d}".format(np.sum(inputs.y == 1)/len(inputs.y)*100, np.sum(inputs.y == 1)))
print("Input parameters:           ", param_dim, inputs.inputs['params'])
print("Optimizer:                  ", optimizer)
print("Loss:                       ", loss_function)
print("Num epochs:                 ", num_epochs)
print("Batch size:                 ", batch_size)
print("Num input nodes:            ", input_nodes)
print("Input activation function:  ", activation)
print("Num hidden layers:          ", num_hidden)
print("Hidden layer nodes:         ", nodes)
print("Hidden activation functions:", h_act[:-2])
print("Num output nodes:           ", output_nodes)
print("Output activation function: ", output_activation)

# nn_dict = {"Training set"                :{'percentage':int(len(x_train)/len(inputs.X)*100), 'size':int(len(x_train))},
#            "Validation set"              :{'percentage':int(len(x_val)/len(inputs.X)*100), 'size':int(len(x_val))},
#            "Testing set"                 :{'percentage':int(len(x_test)/len(inputs.X)*100), 'size':int(len(x_test))},
#            "Class 0"                     :{'percentage':np.sum(inputs.y == 0)/len(inputs.y)*100, 'size':np.sum(inputs.y == 0)},
#            "Class 1"                     :{'percentage':np.sum(inputs.y == 1)/len(inputs.y)*100, 'size':np.sum(inputs.y == 1)},
#            "Input parameters"            :{'Num Features':param_dim, 'Features':inputs.inputs['params']},
#            "Optimizer"                   : optimizer,
#            "Loss"                        : loss_function,
#            "Num epochs"                  : num_epochs,
#            "Batch size"                  : batch_size,
#            "Num input nodes:"            : input_nodes,
#            "Input activation"            : activation,
#            "Num hidden layers"           : num_hidden,
#            "Hidden layer nodes"          : nodes,
#            "Hidden activation functions" : h_act[:-2],
#            "Num output nodes"            : output_nodes,
#            "Output activation function"  : output_activation}

# # convert the nn architecture dict to a pandas DataFrame:     
# nn_df = DataFrame(nn_dict) 

# # save to json:  
# nn_json_file = save_loc + 'nn_architecture_' + run_code + '.json' 
# with open(nn_json_file, mode='w') as f:
#     nn_df.to_json(f)