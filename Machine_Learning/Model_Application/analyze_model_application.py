import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from logger import info
from argparse import ArgumentParser
mpl.use('Agg')
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--tag'       , dest = 'tag'       , help = 'production tag'                      ,  required = True               )
parser.add_argument('--nlayers'   , dest = 'nlayers'   , help = 'number of hidden layers'             ,  required = False , type = int )

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Load predictions for each pairing (15 possible distinct)

predictions = np.load(f'signal_predictions_layers{args.nlayers}_{args.tag}.npz')
p = predictions['p']

info("Preparing to loop.")

H_pairs = [0, 9, 14]

def get_ratio(pair, cut):
    return np.sum(p[:,pair] > cut)/len(p[:,pair])

def multi_eff(pair, cut):
    mask = p[:,pair[0]] > cut
    for i in range(1,len(pair)):
        mask = mask & (p[:,pair[i]] > cut)
    return np.sum(mask)/len(p[:,0])

empty =  []
d = {i:empty for i in range(15)}
d['3H'] = []

thresholds = np.linspace(0.01,1,99)
for cut in thresholds:
    for i in range(15):
        eff = get_ratio(i, cut)
        if i not in H_pairs: eff = 1 - eff
        d[i] = d[i] + [eff]

    d['3H'] = d['3H'] +  [multi_eff([0,9,14], cut)]

print(d['3H'][0])


info("Loop ended! Preparing to plot.")


fig, axs = plt.subplots(nrows=3, ncols=5,  figsize=(16,8), sharex=True, sharey=True)
fig.suptitle(r"Ratio of signal events with predict(pair$_i$) > threshold", size=20)
for i in range(15):
    ax = axs[i//5, i%5]
    ax.plot(thresholds, d[i])
    # ax.set_title(rf"pair$_{{i}}$", size=16, )
    ax.text(0.5, 0.5, rf"pair$_{{{i}}}$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=14)
    if i // 5 == 3:
        ax.set_xlabel("Prediction Threshold", size=16)
    if i%5 == 0:
        ax.set_ylabel("Efficiency",size=16)

plt.tight_layout()
fig.savefig("efficiencies_15pairs.pdf", bbox_inches='tight')

for i,cut in enumerate(thresholds):
    if d['3H'][i] < 0.9*d['3H'][0]:
        print(cut)
        break
### ------------------------------------------------------------------------------------
## Plot and save efficiencies.

fig, ax = plt.subplots()
info("fig, ax defined")

ax.plot(thresholds, d['3H'])
info("data plotted")
ax.set_title("Efficiency of selecting all three Higgs in signal events")
ax.set_xlabel('Prediction Threshold')
ax.set_ylabel('Efficiency')
fig.savefig(f"4layers_{args.tag}_efficiency_all.pdf", bbox_inches="tight")






fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(4,10))
info("fig, ax defined")
plt.autoscale()

ax = axs[0]
ax.plot(thresholds, d[0])
info("data plotted")
ax.set_title("Efficiency of selecting Higgs from X in signal events")
# ax.set_xlabel('Prediction Threshold')
ax.set_ylabel('Efficiency')

ax = axs[1]
info("fig, ax defined")

ax.plot(thresholds, d[9])
info("data plotted")
ax.set_title("Efficiency of selecting Higgs 1 from Y in signal events")
# ax.set_xlabel('Prediction Threshold')
ax.set_ylabel('Efficiency')

ax = axs[2]
info("fig, ax defined")

ax.plot(thresholds, d[14])
info("data plotted")
ax.set_title("Efficiency of selecting Higgs 2 from Y in signal events")
ax.set_xlabel('Prediction Threshold')
ax.set_ylabel('Efficiency')
plt.tight_layout()
fig.savefig(f"4layers_{args.tag}_efficiency_HY2.pdf", bbox_inches="tight")






fig, ax = plt.subplots()

ax.plot(thresholds, d[0] , label="X")
ax.plot(thresholds, d[9], label="Y1")
ax.plot(thresholds, d[14], label="Y2")

ax.set_xlabel('Prediction Threshold')
ax.set_ylabel('Efficiency')
ax.legend()

info("data plotted")
fig.savefig(f"4layers_{args.tag}_efficiency_3Hoverlay.pdf", bbox_inches="tight")

