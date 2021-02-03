import numpy as np
import matplotlib.pyplot as plt

from logger import info, error

num_hidden =  4
pred_save = f"evaluation/hidden_layers_{num_hidden}_predictions.npz"
info(f"Reading predictions from {pred_save}")

p = np.load(pred_save)
preds =  p['predictions']

info("Plotting distributions of model predictions")
high_peak = np.array(())
fig, ax = plt.subplots()
for pred in preds:
    n, edges, _ = ax.hist(pred, bins=100, histtype='step', align='mid')
    pos_of_peak = np.argmax(n[10:]) + 10 # Need to skip the peak at 0, which is higher than the peak near 1. 10 was arbitrarily chosen.
    high_peak = np.append(high_peak, (edges[pos_of_peak] + edges[pos_of_peak + 1]) / 2)

ax.text(0.2, 0.8, f"Number of Hidden Layers: {num_hidden}\nMax: {np.max(high_peak):.3f}, Min: {np.min(high_peak):.3f}, Avg: {np.average(high_peak):.3f}\nNumber of distributions: {len(preds):d}", transform=ax.transAxes)
ax.set_xlabel('Classification Score')

fig_save = f'evaluation/hidden_layers_{num_hidden}_score_dist.pdf'
info(f"Saving histogram to {fig_save}")
fig.savefig(fig_save, bbox_inches='tight')