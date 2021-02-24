import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF CPU warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from argparse import ArgumentParser
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.models import model_from_json
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False

# custom libraries and modules
from logger import info, error
from consistent_plots import hist, hist2d
from kinematics import calcDeltaR


### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')
parser.add_argument('--tag'       , dest = 'tag'       , help = 'production tag'              ,  required = True  )
parser.add_argument('--pred_file' , dest = 'pred_file' , help = 'predictions.npz'             ,  default = False  )
parser.add_argument('--nlayers'   , dest = 'nlayers'   , help = 'number of hidden layers'     ,  default = 4      ,  type = int  )
parser.add_argument('--nmodels'   , dest = 'nmodels'   , help = 'number of models trained'    ,  default = 1      ,  type=int)

args = parser.parse_args()

input_dir = f"layers/layers_{args.nlayers}/{args.tag}/"
model_dir = input_dir + "model/"
eval_dir = input_dir + "evaluation/"
info(f"Evaluating {args.nmodels} models with {args.nlayers} hidden layers and tag {args.tag} from location {input_dir}")


nn_info =  model_dir + 'nn_info.txt'
with open(nn_info, 'r') as nn_info:
    lines = nn_info.readlines()

for line in lines:
    print(line)


def return_predictions(nmodels, tag, pred_file, nlayers):

    dirname = f'layers/layers_{nlayers}/{tag}/'
    dir_model = dirname + 'model/'
    if pred_file:
        predictions = np.load(dirname + 'evaluation/test_predictions.npz')
        return predictions['predictions']

    else:
        predictions = np.array(())
        high_peak = np.array(())

        for i in np.arange(1, int(nmodels) + 1):

            # Load inputs
            inputs = np.load(dir_model + f'test_set_{i}.npz')

            x = inputs['x_test']
            y = inputs['y_test']
            X = inputs['X_test']
            m = inputs['mjj_test']

            info(f"Evaluating model {dir_model + 'model_' + str(i) + '.json'}")

            # load json and create model
            model_json_file = open(dir_model + 'model_' + str(i) + '.json', 'r')
            model_json = model_json_file.read()
            model_json_file.close()
            model = model_from_json(model_json)

            # load weights into new model
            model.load_weights(dir_model + 'model_' + str(i) + '.h5')

            pred = model.predict(x)

            predictions = np.append(predictions, pred)

            bins = np.linspace(0,1,100)
            n, edges = np.histogram(pred, bins=bins)
            a = 10 # arbitrarily chosen
            pos_of_peak = np.argmax(n[a:]) + a # Need to skip the peak at 0, which is higher than the peak near 1. 
            high_peak = np.append(high_peak, (edges[pos_of_peak] + edges[pos_of_peak + 1]) / 2)

        predictions = predictions.reshape(nmodels, len(x))

        # pred_save = f"evaluation/1_hidden_layer_{tag}_predictions.npz"
        # info(f"Saving .npz file of predictions to {pred_save}")
        # np.savez(pred_save, predictions=predictions)

    return predictions, high_peak

def get_roc(y, predictions):
    fpr, tpr, thresholds_keras = roc_curve(y, predictions)
    nn_auc = auc(fpr, tpr)
    return fpr, tpr, nn_auc


print()
info("Beginning model analysis.")

### ------------------------------------------------------------------------------------
## Load or generate predictions npz file

predictions, high_peak = return_predictions(args.nmodels, args.tag, args.pred_file, args.nlayers)


### ------------------------------------------------------------------------------------
## Sample the ROC curve for a random model


i = np.random.randint(1, args.nmodels+1)
ex_file = f"{model_dir}test_set_{i}.npz"
info(f"Importing test set from file: {ex_file}")
examples = np.load(ex_file)
y = examples['y_test']
fpr, tpr, nn_auc = get_roc(y, predictions[i-1,:])

fig, ax = plt.subplots(figsize=(15,15))
ax.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(nn_auc))
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.legend()
ax.set_title(f'ROC curve for model {i} with {args.nlayers} hidden layers and tag {args.tag}')
fig.savefig(eval_dir + 'roc')


### ------------------------------------------------------------------------------------
## Sample mass scatter plot for the same random model

m = examples['mjj_test']

H_mask = (y == 1)

m_bins = np.linspace(0, 600, 100)
p_bins = np.linspace(0,1,100)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
fig.suptitle(rf"NN Test Scores vs. $m_{{bb}}$ for model {i} with {args.nlayers} hidden layers and tag {args.tag}")

ax = axs[0]
m_nH = m[~H_mask]
p_nH = predictions[i-1,:][~H_mask]
n, xedges, yedges, im = hist2d(ax, m_nH, p_nH, xbins=m_bins, ybins=p_bins)
ax.set_xlabel(r"Non-Higgs pair $m_{jj}$ [GeV]")
ax.set_ylabel("NN Score")
ax.set_title("Non-Higgs Pairs")
fig.colorbar(im, ax=ax)

ax = axs[1]
m_H = m[H_mask]
p_H = predictions[i-1,:][H_mask]
n, xedges, yedges, im = hist2d(ax, m_H, p_H, xbins=m_bins, ybins=p_bins)
ax.set_xlabel(r"Higgs pair $m_{jj}$ [GeV]")
ax.set_ylabel("NN Score")
ax.set_title("Higgs Pairs")
fig.colorbar(im, ax=ax)

fig.savefig(eval_dir + 'mass_scatter')

### ------------------------------------------------------------------------------------
## Sample DeltaR scatter plot for the same random model

eta1 = examples['X_test'][:,1]
phi1 = examples['X_test'][:,2]
eta2 = examples['X_test'][:,4]
phi2 = examples['X_test'][:,5]
dR = calcDeltaR(eta1, eta2, phi1, phi2)

H_mask = (y == 1)

dR_bins = np.linspace(0, 7, 100)
p_bins = np.linspace(0,1,100)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
fig.suptitle(rf"NN Test Scores vs. $\Delta R_{{bb}}$ for model {i} with {args.nlayers} hidden layers and tag {args.tag}")

ax = axs[0]
dR_nH = dR[~H_mask]
p_nH = predictions[i-1,:][~H_mask]
n, xedges, yedges, im = hist2d(ax, dR_nH, p_nH, xbins=dR_bins, ybins=p_bins)
ax.set_xlabel(r"Non-Higgs pair $\Delta R_{jj}$")
ax.set_ylabel("NN Score")
ax.set_title("Non-Higgs Pairs")
fig.colorbar(im, ax=ax)

ax = axs[1]
dR_H = dR[H_mask]
p_H = predictions[i-1,:][H_mask]
n, xedges, yedges, im = hist2d(ax, dR_H, p_H, xbins=dR_bins, ybins=p_bins)
ax.set_xlabel(r"Higgs pair $\Delta R_{jj}$")
ax.set_ylabel("NN Score")
ax.set_title("Higgs Pairs")
fig.colorbar(im, ax=ax)

fig.savefig(eval_dir + 'dR_scatter')

### ------------------------------------------------------------------------------------
## Plot distribution of test masses

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,7))
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

ax1 = axs[0]
ax0 = axs[1]
hist(ax0, m_H, bins=m_bins)
hist(ax1, m_nH, bins=m_bins)


ax0.set_title("True Higgs Pairs")
ax1.set_title("True Non-Higgs Pairs")

ax0.set_xlabel('Prediction')
ax0.set_ylabel('Count')

ax1.set_xlabel('Prediction')
ax1.set_ylabel('Count')

# ax.text(0.2, 0.8, f"Number of Hidden Layers: {{{args.nlayers}}}\nMax: {{{np.max(high_peak):.3f}}}, Min: {{{np.min(high_peak):.3f}}}, Avg: {{{np.average(high_peak):.3f}}}\nNumber of distributions: {{{len(predictions):d}}}", transform=ax.transAxes)

fig.savefig(eval_dir + 'test_mass')


### ------------------------------------------------------------------------------------
## Plot distribution of test masses

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,7))
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

ax1 = axs[0]
ax0 = axs[1]
hist(ax0, dR_H, bins=dR_bins)
hist(ax1, dR_nH, bins=dR_bins)

ax0.set_title("True Higgs Pairs")
ax1.set_title("True Non-Higgs Pairs")

ax0.set_xlabel(r'$\Delta R_{jj}$')
ax0.set_ylabel('Count')

ax1.set_xlabel(r'$\Delta R_{jj}$')
ax1.set_ylabel('Count')

# ax.text(0.2, 0.8, f"Number of Hidden Layers: {{{args.nlayers}}}\nMax: {{{np.max(high_peak):.3f}}}, Min: {{{np.min(high_peak):.3f}}}, Avg: {{{np.average(high_peak):.3f}}}\nNumber of distributions: {{{len(predictions):d}}}", transform=ax.transAxes)

fig.savefig(eval_dir + 'test_dR')

### ------------------------------------------------------------------------------------
## Plot distribution of NN Test Scores

fig = plt.figure(figsize=(15,7))
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

ax0 = fig.add_subplot(122, label="Higgs")
ax1 = fig.add_subplot(121, label="Regular")
ax2 = fig.add_subplot(121, label="LogPlot", frameon=False)

c0 = 'C0'
c1 = 'C1'
c2 = 'C2'

for pred in predictions:
    hist(ax0, pred[H_mask], bins=p_bins, label='True Higgs Pair')
    hist(ax1, pred[~H_mask], bins=p_bins, label='True Non-Higgs Pair', color=c1)
    hist(ax2, pred[~H_mask], bins=p_bins, label='True Non-Higgs Pair', color=c2)


ax0.set_title("True Higgs Pairs")
ax1.set_title("True Non-Higgs Pairs")

ax0.set_xlabel('Prediction')
ax0.set_ylabel('Count')

ax1.set_xlabel('Prediction')
ax1.set_ylabel('Count', color=c1)
ax1.tick_params(axis='y', colors=c1)

ax2.tick_params(axis='x', labelbottom=False)
ax2.set_yscale('log')
ax2.set_ylabel('Log(Count)', color=c2, rotation=270, labelpad=25)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')
ax2.tick_params(axis='y',  colors=c2)
ax2.axis()

ax.text(0.2, 0.8, f"Number of Hidden Layers: {{{args.nlayers}}}\nMax: {{{np.max(high_peak):.3f}}}, Min: {{{np.min(high_peak):.3f}}}, Avg: {{{np.average(high_peak):.3f}}}\nNumber of distributions: {{{len(predictions):d}}}", transform=ax.transAxes)

fig.savefig(eval_dir + 'test_predictions')
