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
from matplotlib.backends.backend_pdf import PdfPages

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

threshold = 0.5

nn_info =  model_dir + 'nn_info.txt'
with open(nn_info, 'r') as nn_info:
    lines = nn_info.readlines()

for line in lines:
    print(line)


def return_predictions(nmodels, tag, pred_file=False, nlayers=4):

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

pdf_file = f'Model_Eval/{args.tag}.pdf'

with PdfPages(pdf_file) as pdf:

    for nmodel in np.arange(1,args.nmodels+1):

        # i = np.random.randint(1, args.nmodels+1)
        ex_file = f"{model_dir}test_set_{nmodel}.npz"
        info(f"Importing test set from file: {ex_file}")
        examples = np.load(ex_file)

        y = examples['y_test']
        H_mask = (y == 1)

        m = examples['mjj_test']

        eta1 = examples['X_test'][:,1]
        phi1 = examples['X_test'][:,2]
        eta2 = examples['X_test'][:,4]
        phi2 = examples['X_test'][:,5]

        dR = calcDeltaR(eta1, eta2, phi1, phi2)

        dR_nH = dR[~H_mask]
        dR_H = dR[H_mask]

        p_nH = predictions[nmodel-1,:][~H_mask]
        p_H = predictions[nmodel-1,:][H_mask]

        m_nH = m[~H_mask]
        m_H = m[H_mask]

        high_nH = p_nH > threshold
        low_H   = p_H  < threshold


        ### ------------------------------------------------------------------------------------
        ## Prep plots

        dR_bins = np.linspace(0, 7, 100)
        m_bins = np.linspace(0, 600, 100)
        p_bins = np.linspace(0,1,100)

        fig = plt.figure(figsize=(10,20))
        fig.suptitle(f"{args.tag}  {nmodel}")
        gs = fig.add_gridspec(7, 8)

        # Mass distribution plots
        mnH_ax       = fig.add_subplot(gs[0, 0:4])
        mH_ax        = fig.add_subplot(gs[0, 4:])

        dRnH_ax      = fig.add_subplot(gs[1, 0:4])
        dRH_ax       = fig.add_subplot(gs[1, 4:])

        mscatnH_ax   = fig.add_subplot(gs[2, 0:4])
        mscatH_ax    = fig.add_subplot(gs[2, 4:])

        dRscatnH_ax  = fig.add_subplot(gs[3, 0:4])
        dRscatH_ax   = fig.add_subplot(gs[3, 4:])

        massdRnH_ax   = fig.add_subplot(gs[4, 0:4])
        massdRH_ax    = fig.add_subplot(gs[4, 4:])

        scoreH_ax    = fig.add_subplot(gs[5, 4:], label="Higgs")
        scorenH_ax1  = fig.add_subplot(gs[5, 0:3], label="Regular")
        scorenH_ax2  = fig.add_subplot(gs[5, 0:3], label="LogPlot", frameon=False)

        roc_ax       = fig.add_subplot(gs[6, 2:5])

        ### ------------------------------------------------------------------------------------
        ## Plot distribution of test masses

        hist(mH_ax, m_H, bins=m_bins, label='All Scores')
        hist(mH_ax, m_H[low_H], bins=m_bins, label='Low Scores')
        hist(mnH_ax, m_nH, bins=m_bins, label='All Scores')
        hist(mnH_ax, m_nH[high_nH], bins=m_bins, label='Low Scores')

        mH_ax.set_title("True Higgs Pairs", pad=12)
        mnH_ax.set_title("True Non-Higgs Pairs", pad=12)

        mH_ax.set_xlabel(r'$m_{bb}$ [GeV]')
        mH_ax.set_ylabel('Count')

        mnH_ax.set_xlabel(r'$m_{bb}$ [GeV]')
        mnH_ax.set_ylabel('Count')


        ### ------------------------------------------------------------------------------------
        ## Plot distribution of test DeltaR

        hist(dRH_ax, dR_H, bins=dR_bins, label='All Scores')
        hist(dRH_ax, dR_H[low_H], bins=dR_bins, label='Low Scores')
        hist(dRnH_ax, dR_nH, bins=dR_bins, label='All Scores')
        hist(dRnH_ax, dR_nH[high_nH], bins=dR_bins, label='Low Scores')
        

        dRH_ax.set_xlabel(r'$\Delta R_{jj}$')
        dRH_ax.set_ylabel('Count')

        dRnH_ax.set_xlabel(r'$\Delta R_{jj}$')
        dRnH_ax.set_ylabel('Count')


        ### ------------------------------------------------------------------------------------
        ## Sample mass scatter plot for the same random model

        n, xedges, yedges, im = hist2d(mscatnH_ax, m_nH, p_nH, xbins=m_bins, ybins=p_bins)
        mscatnH_ax.set_xlabel(r"Non-Higgs pair $m_{jj}$ [GeV]")
        mscatnH_ax.set_ylabel("NN Score")
        fig.colorbar(im, ax=mscatnH_ax)

        n, xedges, yedges, im = hist2d(mscatH_ax, m_H, p_H, xbins=m_bins, ybins=p_bins)
        mscatH_ax.set_xlabel(r"Higgs pair $m_{jj}$ [GeV]")
        mscatH_ax.set_ylabel("NN Score")
        fig.colorbar(im, ax=mscatH_ax)


        ### ------------------------------------------------------------------------------------
        ## Sample DeltaR scatter plot for the same random model


        n, xedges, yedges, im = hist2d(dRscatnH_ax, dR_nH, p_nH, xbins=dR_bins, ybins=p_bins)
        dRscatnH_ax.set_xlabel(r"Non-Higgs pair $\Delta R_{jj}$")
        dRscatnH_ax.set_ylabel("NN Score")
        fig.colorbar(im, ax=dRscatnH_ax)

        n, xedges, yedges, im = hist2d(dRscatH_ax, dR_H, p_H, xbins=dR_bins, ybins=p_bins)
        dRscatH_ax.set_xlabel(r"Higgs pair $\Delta R_{jj}$")
        dRscatH_ax.set_ylabel("NN Score")
        fig.colorbar(im, ax=dRscatH_ax)


        ### ------------------------------------------------------------------------------------
        ## Sample DeltaR scatter plot for the same random model


        n, xedges, yedges, im = hist2d(massdRnH_ax, dR_nH, m_nH, xbins=dR_bins, ybins=m_bins)
        massdRnH_ax.set_xlabel(r"Non-Higgs pair $\Delta R_{jj}$")
        massdRnH_ax.set_ylabel(r"Higgs pair $m_{jj}$ [GeV]")
        fig.colorbar(im, ax=massdRnH_ax)

        n, xedges, yedges, im = hist2d(massdRH_ax, dR_H, m_H, xbins=dR_bins, ybins=m_bins)
        massdRH_ax.set_xlabel(r"Higgs pair $\Delta R_{jj}$")
        massdRH_ax.set_ylabel(r"Higgs pair $m_{jj}$ [GeV]")
        fig.colorbar(im, ax=massdRH_ax)


        ### ------------------------------------------------------------------------------------
        ## Plot distribution of NN Test Scores


        c0 = 'C0'
        c1 = 'C0'
        c2 = 'C1'

        # for pred in predictions:
        hist(scoreH_ax, p_H, bins=p_bins, label='True Higgs Pair')
        hist(scorenH_ax1, p_nH, bins=p_bins, label='True Non-Higgs Pair', color=c1)
        hist(scorenH_ax2, p_nH, bins=p_bins, label='True Non-Higgs Pair', color=c2)

        scoreH_ax.set_xlabel('NN Score')
        scoreH_ax.set_ylabel('Count')

        scorenH_ax1.set_xlabel('NN Score')
        scorenH_ax1.set_ylabel('Count', color=c1)
        scorenH_ax1.tick_params(axis='y', colors=c1)

        scorenH_ax2.tick_params(axis='x', labelbottom=False)
        scorenH_ax2.set_yscale('log')
        scorenH_ax2.set_ylabel('Log(Count)', color=c2, rotation=270, labelpad=25)
        scorenH_ax2.yaxis.tick_right()
        scorenH_ax2.yaxis.set_label_position('right')
        scorenH_ax2.tick_params(axis='y',  colors=c2)
        scorenH_ax2.axis()



        ### ------------------------------------------------------------------------------------
        ## Sample the ROC curve for a random model

        fpr, tpr, nn_auc = get_roc(y, predictions[nmodel-1,:])

        roc_ax.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(nn_auc))
        roc_ax.set_xlabel('False positive rate')
        roc_ax.set_ylabel('True positive rate')
        roc_ax.legend()


        ### ------------------------------------------------------------------------------------
        ## Save!

        pdf.savefig()
