import numpy as np
import matplotlib.pyplot as plt

inputs = np.load('Gen_Inputs/nn_input_MX700_MY400_class.npz')

x = inputs['x']
nparams = np.shape(x)[1]
feature = inputs['params']

y = inputs['y']

fig, axs = plt.subplots(nrows = nparams, ncols = nparams, figsize=(16,16), sharex='col')

for i in np.arange(nparams):
    for j in np.arange(nparams):

        ax = axs[i][j]

        if i == j:
            ax.hist(x[:,i], bins=100, histtype='step', align='mid')
            ax.set_xlabel(feature[i])

        else:
            ax.scatter(x[:,j], x[:,i], s=.1, c=y, cmap='cool')
            ax.set_xlabel(feature[j])
            ax.set_ylabel(feature[i])


fig.savefig('Gen_Inputs/parton_visualizer')