try:
    from keras.models import model_from_json
except ModuleNotFoundError:
    print("You must activate the conda environment! (Use alias 'csixb')")
import numpy as np

import matplotlib.pyplot as plt 
# plt.style.use('../../Config/plots.mplstyle')

from os import environ, path
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF CPU warnings

# custom libraries and modules
from logger import info, error

print()
info("Beginning model analysis.\n")

# User parameters
num_hidden = 2
num_models_to_evaluate = 30

filepath = f'num_hidden/num_hidden_{num_hidden}/'

info(f"Evaluating {num_models_to_evaluate} models with {num_hidden} hidden layers from location {filepath}")

predictions = np.array(())

for i in np.arange(1, num_models_to_evaluate + 1):

    info(f"Evaluating model {filepath + 'model_' + str(i) + '.json'}")

    # load json and create model
    model_json_file = open(filepath + 'model_' + str(i) + '.json', 'r')
    model_json = model_json_file.read()
    model_json_file.close()
    model = model_from_json(model_json)

    # load weights into new model
    model.load_weights(filepath + 'model_' + str(i) + '.h5')

    # Load inputs
    inputs = np.load(filepath + 'test_set_' + str(i) + '.npz')

    x = inputs['x_test']
    y = inputs['y_test']
    X = inputs['X_test']
    m = inputs['mjj_test']

    pred = model.predict(x)
    predictions = np.append(predictions, pred)


predictions = predictions.reshape(num_models_to_evaluate, len(x))

pred_save = f"evaluation/hidden_layers_{num_hidden}_predictions.npz"
info(f"Saving .npz file of predictions to {pred_save}")
np.savez(pred_save, predictions=predictions)

fig, ax = plt.subplots()

high_peak = np.array(())
for arr in predictions:
    n, edges, _ = ax.hist(arr, bins=100, histtype='step', align='mid')
    pos_of_peak = np.argmax(n[10:]) + 10 # Need to skip the peak at 0, which is higher than the peak near 1. 10 was arbitrarily chosen.
    high_peak = np.append(high_peak, (edges[pos_of_peak] + edges[pos_of_peak + 1]) / 2)

print(f"Maximum: {np.max(high_peak):.3f}")
print(f"Minimum: {np.min(high_peak):.3f}")
print(f"Average: {np.average(high_peak):.3f}")

ax.set_xlabel('Classification Score')
ax.set_ylabel('Count')
ax.text(0.3, 0.9, f"Number of Hidden Layers: {num_hidden}\nMax: {np.max(high_peak):.3f}, Min: {np.min(high_peak):.3f}, Avg: {np.average(high_peak):.3f}", transform=ax.transAxes)
scores = f'evaluation/hidden_layers_{num_hidden}_score_dist.pdf'

info(f"Saving histogram of classification score distributions to {scores}")
fig.savefig(scores, bbox_inches='tight')

