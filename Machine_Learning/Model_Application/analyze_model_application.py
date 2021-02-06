import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from logger import info
from argparse import ArgumentParser
mpl.use('Agg')

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--tag'       , dest = 'tag'       , help = 'production tag'                      ,  required = True               )
parser.add_argument('--nlayers'   , dest = 'nlayers'   , help = 'number of hidden layers'             ,  required = False , type = int )

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Load predictions for each pairing (15 possible distinct)

predictions = np.load('signal_predictions_layers{args.nlayers}_{args.tag}.npz')
p = predictions['p']

info("Preparing to loop.")

eff = np.array(())
thresholds = np.linspace(0,1,100)
for cut in thresholds:
    HX = p[:,0] > cut # HX is pair with index 0
    HY1 = p[:,9] > cut # HY1 is pair with index 9
    HY2 = p[:,14] > cut # HY2 is pair with index 14

    all_three_pass = HX & HY1 & HY2

    eff = np.append(eff, np.sum(all_three_pass)/len(all_three_pass))
    
info("Loop ended! Preparing to plot.")

### ------------------------------------------------------------------------------------
## Plot and save efficiencies.

fig, ax = plt.subplots()
info("fig, ax defined")

ax.plot(thresholds, eff)
info("data plotted")
ax.set_title("Efficiency of selecting all three Higgs in signal events")
ax.set_xlabel('Prediction Threshold')
ax.set_ylabel('Efficiency')
fig.savefig("4layers_unsmeared_efficiency")