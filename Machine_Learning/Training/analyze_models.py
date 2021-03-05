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
import pandas as pd

# custom libraries and modules
from logger import info, error
from consistent_plots import hist, hist2d
from kinematics import calcDeltaR


### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')
parser.add_argument('--tag'       , dest = 'tag'       , help = 'production tag'              ,  required = True  )
parser.add_argument('--pred_file' , dest = 'pred_file' , help = 'scores.npz'             ,  default = False  )
parser.add_argument('--nlayers'   , dest = 'nlayers'   , help = 'number of hidden layers'     ,  default = 4      ,  type = int  )
parser.add_argument('--nmodels'   , dest = 'nmodels'   , help = 'number of models trained'    ,  default = 1      ,  type=int)
parser.add_argument('--twoclass'  , dest = 'twoclass'  , help = 'two class output sample'     ,  action='store_true', default=False)

args = parser.parse_args()

input_dir = f"layers/layers_{args.nlayers}/{args.tag}/"
if args.twoclass: input_dir = f"twoclass/{args.tag}/"
model_dir = input_dir + "model/"
eval_dir = input_dir + "evaluation/"
info(f"Evaluating {args.nmodels} models with {args.nlayers} hidden layers and tag {args.tag} from location {input_dir}")

threshold = 0.5

nn_info =  input_dir + 'nn_info.txt'
with open(nn_info, 'r') as nn_info:
    lines = nn_info.readlines()

for line in lines:
    print(line)


def get_scores(x, i, pred_file=False):
    if pred_file:
        scores = np.load(model_dir + 'evaluation/test_scores_{i}.npz')
        return scores['scores']

    else:
        scores = np.array(())

        info(f"Evaluating model {model_dir + 'model_' + str(i) + '.json'}")

        # load json and create model
        model_json_file = open(model_dir + 'model_' + str(i) + '.json', 'r')
        model_json = model_json_file.read()
        model_json_file.close()
        model = model_from_json(model_json)

        # load weights into new model
        model.load_weights(model_dir + 'model_' + str(i) + '.h5')

        pred = model.predict(x)

    return pred

def get_history(i):
    history_json_file = open(model_dir + 'history_' + str(i) + '.json', 'r')
    history = pd.read_json(history_json_file)
    history_json_file.close()
    return history


def get_roc(y, scores):
    fpr, tpr, thresholds_keras = roc_curve(y, scores)
    print("thresholds_keras",thresholds_keras)
    nn_auc = auc(fpr, tpr)
    return fpr, tpr, nn_auc


print()
info("Beginning model analysis.")

### ------------------------------------------------------------------------------------
## Load or generate scores npz file

# if not os.exists('Model_Eval'):
    

pdf_file = f'Model_Eval/{args.tag}.pdf'
if args.twoclass: pdf_file = f'Model_Eval/twoclass/{args.tag}.pdf'

with PdfPages(pdf_file) as pdf:

    for nmodel in np.arange(1,args.nmodels+1):

        ex_file = f"{model_dir}test_set_{nmodel}.npz"
        info(f"Importing test set from file: {ex_file}")
        examples = np.load(ex_file)

        x = examples['x_test']
        y = examples['y_test']

        print(y[:6,:])

        scores = get_scores(x, nmodel, args.pred_file)

        scaled_scores = scores / np.max(scores)

        if args.twoclass:
            y_nH = y[:,1]
            y_H = y[:,1]
        else:
            y_nH = y
            y_H = y

        for nH, H in zip(scores[:,0], scores[:,1]):
            if (nH + H - 1 > 1e-3):
                print("ERROR")
        
        np.savez(f"{eval_dir}scores_{nmodel}.npz", scores=scores)

        H_mask = (y_H == 1)
        nH_mask = (y_nH == 0)

        m = examples['mjj_test']

        eta1 = examples['X_test'][:,1]
        phi1 = examples['X_test'][:,2]
        eta2 = examples['X_test'][:,4]
        phi2 = examples['X_test'][:,5]

        dR = calcDeltaR(eta1, eta2, phi1, phi2)

        dR_nH = dR[nH_mask]
        dR_H = dR[H_mask]

        if args.twoclass:
            p_nH = scores[:,1][nH_mask]
            p_H = scores[:,1][H_mask]
            # assert(p_nH + p_H - 1 < 1e-3)
        else:
            p_nH = scores[nH_mask]
            p_H = scores[H_mask]

        print(scores[:6,:])

        m_nH = m[nH_mask]
        m_H = m[H_mask]

        high_nH = p_nH > threshold
        low_H   = p_H  < threshold


        ### ------------------------------------------------------------------------------------
        ## Prep plots

        dR_bins = np.linspace(0, 7, 100)
        m_bins = np.linspace(0, 600, 100)
        p_bins = np.linspace(0,1,100)

        fig = plt.figure(figsize=(20,10))
        fig.suptitle(f"{args.tag}  {nmodel}")

        gs = fig.add_gridspec(3, 20)

        # Mass distribution plots
        mnH_ax       = fig.add_subplot(gs[0, 0:4])
        mH_ax        = fig.add_subplot(gs[0, 5:9])

        dRnH_ax      = fig.add_subplot(gs[1, 0:4])
        dRH_ax       = fig.add_subplot(gs[1, 5:9])

        mscatnH_ax   = fig.add_subplot(gs[0, 10:15])
        mscatH_ax    = fig.add_subplot(gs[0, 15:20])

        dRscatnH_ax  = fig.add_subplot(gs[1, 10:15])
        dRscatH_ax   = fig.add_subplot(gs[1, 15:20])

        # massdRnH_ax   = fig.add_subplot(gs[4, 0:4])
        # massdRH_ax    = fig.add_subplot(gs[4, 4:])

        scorenH_ax1  = fig.add_subplot(gs[2, 10:14], label="Regular")
        scorenH_ax2  = fig.add_subplot(gs[2, 10:14], label="LogPlot", frameon=False)
        scoreH_ax    = fig.add_subplot(gs[2, 15:19], label="Higgs")

        roc_ax   = fig.add_subplot(gs[2, 0:4])
        history_ax = fig.add_subplot(gs[2, 5:9])

        line = plt.Line2D((.48,.48),(.1,.9), color="k", linewidth=1)
        fig.add_artist(line)

        line = plt.Line2D((.05,.95),(.9,.9), color="k", linewidth=1)
        fig.add_artist(line)

        ### ------------------------------------------------------------------------------------
        ## Plot distribution of test masses

        hist(mH_ax, m_H, bins=m_bins, label='All Scores')
        # hist(mH_ax, m_H[low_H], bins=m_bins, label='Low Scores')
        hist(mnH_ax, m_nH, bins=m_bins, label='All Scores')
        # hist(mnH_ax, m_nH[high_nH], bins=m_bins, label='Low Scores')

        # mnH_ax.legend()

        mH_ax.set_title("True Higgs Pairs", pad=20)
        mnH_ax.set_title("True Non-Higgs Pairs", pad=20)

        mH_ax.set_xlabel(r'Higgs Pair $m_{bb}$ [GeV]')
        mH_ax.set_ylabel('Count')

        mnH_ax.set_xlabel(r'Non-Higgs Pair $m_{bb}$ [GeV]')
        mnH_ax.set_ylabel('Count')


        ### ------------------------------------------------------------------------------------
        ## Plot distribution of test DeltaR

        hist(dRH_ax, dR_H, bins=dR_bins, label='All Scores')
        # hist(dRH_ax, dR_H[low_H], bins=dR_bins, label='Low Scores')
        hist(dRnH_ax, dR_nH, bins=dR_bins, label='All Scores')
        # hist(dRnH_ax, dR_nH[high_nH], bins=dR_bins, label='Low Scores')

        # dRnH_ax.legend()
        

        dRH_ax.set_xlabel(r"Higgs pair $\Delta R_{jj}$")
        dRH_ax.set_ylabel('Count')

        dRnH_ax.set_xlabel(r"Non-Higgs pair $\Delta R_{jj}$")
        dRnH_ax.set_ylabel('Count')


        ### ------------------------------------------------------------------------------------
        ## Sample mass scatter plot for the same random model

        mscatH_ax.set_title("True Higgs Pairs", pad=12)
        mscatnH_ax.set_title("True Non-Higgs Pairs", pad=12)

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


        # ### ------------------------------------------------------------------------------------
        # ## Sample DeltaR scatter plot for the same random model


        # n, xedges, yedges, im = hist2d(massdRnH_ax, dR_nH, m_nH, xbins=dR_bins, ybins=m_bins)
        # massdRnH_ax.set_xlabel(r"Non-Higgs pair $\Delta R_{jj}$")
        # massdRnH_ax.set_ylabel(r"Non-Higgs pair $m_{jj}$ [GeV]")
        # fig.colorbar(im, ax=massdRnH_ax)

        # n, xedges, yedges, im = hist2d(massdRH_ax, dR_H, m_H, xbins=dR_bins, ybins=m_bins)
        # massdRH_ax.set_xlabel(r"Higgs pair $\Delta R_{jj}$")
        # massdRH_ax.set_ylabel(r"Higgs pair $m_{jj}$ [GeV]")
        # fig.colorbar(im, ax=massdRH_ax)


        ### ------------------------------------------------------------------------------------
        ## Plot distribution of NN Test Scores


        c0 = 'C0'
        c1 = 'C0'
        c2 = 'C1'

        hist(scoreH_ax, p_H, bins=p_bins, label='True Higgs Pair')
        hist(scorenH_ax1, p_nH, bins=p_bins, label='True Non-Higgs Pair', color=c1)
        hist(scorenH_ax2, p_nH, bins=p_bins, label='True Non-Higgs Pair', color=c2)

        scoreH_ax.set_xlabel('NN Score')
        scoreH_ax.set_ylabel('Count')

        scorenH_ax1.set_xlabel('NN Score')
        scorenH_ax1.set_ylabel('Count')
        scorenH_ax1.tick_params(axis='y', colors=c1)

        scorenH_ax2.tick_params(axis='x', labelbottom=False)
        scorenH_ax2.set_yscale('log')
        # scorenH_ax2.set_ylabel('Log(Count)', rotation=270, labelpad=25)
        scorenH_ax2.yaxis.tick_right()
        scorenH_ax2.yaxis.set_label_position('right')
        scorenH_ax2.tick_params(axis='y',  colors=c2)
        scorenH_ax2.axis()

        ### ------------------------------------------------------------------------------------
        ## Plot distribution of NN Test Scores (SCALED)


        # c0 = 'C0'
        # c1 = 'C0'
        # c2 = 'C1'

        # hist(scaled_scoreH_ax, p_H / np.max(scores), bins=p_bins, label='True Higgs Pair')
        # hist(scaled_scorenH_ax1, p_nH / np.max(scores), bins=p_bins, label='True Non-Higgs Pair', color=c1)
        # hist(scaled_scorenH_ax2, p_nH / np.max(scores), bins=p_bins, label='True Non-Higgs Pair', color=c2)

        # scaled_scoreH_ax.set_xlabel('NN Score (SCALED)')
        # scaled_scoreH_ax.set_ylabel('Count')

        # scaled_scorenH_ax1.set_xlabel('NN Score  (SCALED)')
        # scaled_scorenH_ax1.set_ylabel('Count', color=c1)
        # scaled_scorenH_ax1.tick_params(axis='y', colors=c1)

        # scaled_scorenH_ax2.tick_params(axis='x', labelbottom=False)
        # scaled_scorenH_ax2.set_yscale('log')
        # scaled_scorenH_ax2.set_ylabel('Log(Count)', color=c2, rotation=270, labelpad=25)
        # scaled_scorenH_ax2.yaxis.tick_right()
        # scaled_scorenH_ax2.yaxis.set_label_position('right')
        # scaled_scorenH_ax2.tick_params(axis='y',  colors=c2)
        # scaled_scorenH_ax2.axis()

        ### ------------------------------------------------------------------------------------
        ## Sample the ROC curve for a random model
        # if not args.twoclass:
        fpr, tpr, nn_auc = get_roc(y[:,1], scores[:,1])

        roc_ax.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(nn_auc))
        roc_ax.set_xlabel('False positive rate')
        roc_ax.set_ylabel('True positive rate')
        roc_ax.legend()

        ### ------------------------------------------------------------------------------------
        ## Sample the ROC curve for a random model
        # if not args.twoclass:
        history = get_history(nmodel)

        train_acc = history['accuracy']
        valid_acc = history['val_accuracy']
        nepochs = np.arange(len(train_acc))

        history_ax.plot(nepochs, train_acc, label='Training')
        history_ax.plot(nepochs, valid_acc, label='Validation')
        history_ax.set_xlabel('Epoch')
        history_ax.set_ylabel('Accuracy')
        history_ax.legend()


        ### ------------------------------------------------------------------------------------
        ## Save!
        info(f"Saving pdf to {pdf_file}")
        pdf.savefig()
