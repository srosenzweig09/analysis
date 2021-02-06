import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from logger import info
mpl.use('Agg')

predictions = np.load('signal_event_model_predictions_15pairs.npz')
p = predictions['p']

info("Preparing to loop.")

eff = np.array(())
thresholds = np.linspace(0,1,100)
for cut in thresholds:
    Higgs1 = p[:,0] > cut
    Higgs2 = p[:,9] > cut
    Higgs3 = p[:,14] > cut

    all_three_pass = Higgs1 & Higgs2 & Higgs3

    eff = np.append(eff, np.sum(all_three_pass)/len(all_three_pass))
    
info("Loop ended! Preparing to plot.")

fig, ax = plt.subplots()
info("fig, ax defined")

ax.plot(thresholds, eff)
info("data plotted")
ax.set_title("Efficiency of selecting all three Higgs in signal events")
ax.set_xlabel('Prediction Threshold')
ax.set_ylabel('Efficiency')
fig.savefig("4layers_unsmeared_efficiency")