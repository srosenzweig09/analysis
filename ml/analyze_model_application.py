import PyQt5
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from logger import info
from argparse import ArgumentParser
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']

from matplotlib.widgets import Slider

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--type'   , dest = 'type'   , help = 'reco, parton, smeared'   ,  required = True)
parser.add_argument('--task'   , dest = 'task'   , help = 'class or reg'            ,  required = True)
parser.add_argument('--nmodels', dest = 'nmodels', help = 'number of models trained',  default = 1    , type = int)

args = parser.parse_args()

### ------------------------------------------------------------------------------------
## Load predictions for each pairing (15 possible distinct)

if args.task == 'class':
    task = 'classifier'
if args.task == 'reg':
    task = 'regressor'

predictions = np.load(f'Evaluations/{task}/scores_{args.run}.npz')
p = predictions['p']

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

thresholds = np.round(np.linspace(0,1.01,100), 2)
for cut in thresholds:
    for i in range(15):
        eff = get_ratio(i, cut)
        if i not in H_pairs: eff = 1 - eff
        d[i] = d[i] + [eff]

    d['3H'] = d['3H'] +  [multi_eff([0,9,14], cut)]


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
fig.savefig("efficiencies_15pairs.pdf")


### ------------------------------------------------------------------------------------
## Plot and save efficiencies.

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,8))
info("fig, ax defined")

ax = axs[0]
ax.plot(thresholds, d['3H'])
info("data plotted")
ax.set_title("Efficiency of selecting all three Higgs in signal events")
ax.set_xlabel('Prediction Threshold')
ax.set_ylabel('Efficiency')


ax = axs[1]
ax.plot(thresholds, d[0] , label="X")
ax.plot(thresholds, d[9], label="Y1")
ax.plot(thresholds, d[14], label="Y2")

ax.set_xlabel('Prediction Threshold')
ax.set_ylabel('Efficiency')
ax.legend()

info("data plotted")
fig.savefig(f"efficiency_{args.tag}.pdf")


### ------------------------------------------------------------------------------------
## Interactive plot.
def get_bars(j):
    return [d['3H'][j]] + [d[i][j] for i in np.arange(15)]

def update(val):
    eff = np.floor(s.val*100)
    f.set_data(x, get_bars(eff))
    fig.canvas.draw_idle()

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2, top=0.75)

ax = fig.add_axes([0.3, 0.85, 0.4,  0.05])
s = Slider(ax=ax, label='Threshold', valmin=0.0, valmax=1.0, valstep=0.01, valfmt=' %1.2f')

x = [r'$3H$', r'$H_X$', r'$H_{Y,1}$', r'$H_{Y,2}$', r'$b_{X,1}$, $b_{Y_1,1}$', r'$b_{X,1}$, $b_{Y_1,2}$', r'$b_{X,1}$, $b_{Y_2,1}$', r'$b_{X,1}$, $b_{Y_2,2}$', r'$b_{X,2}$, $b_{Y_1,1}$', r'$b_{X,2}$, $b_{Y_1,2}$', r'$b_{X,2}$, $b_{Y_2,1}$', r'$b_{X,2}$, $b_{Y_2,2}$', r'$b_{Y_1,1}$, $b_{Y_2,1}$', r'$b_{Y_1,1}$, $b_{Y_2,2}$', r'$b_{Y_1,2}$, $b_{Y_2,1}$', r'$b_{Y_1,2}$, $b_{Y_2,2}$']

j = 0
y = get_bars(0)
f = ax.bar(x, y)
ax.set_ylabel('Events bove threshold')
ax.set_xlabel('Pairs')


plt.show()

s.on_changed(update)