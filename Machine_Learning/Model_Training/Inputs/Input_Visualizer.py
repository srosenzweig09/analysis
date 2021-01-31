import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

inputs = np.load('../Inputs/MX700_MY400_classifier_allpairs_dR_presel.npz')

X = inputs['x']
nparams = np.shape(X)[1]
feature = inputs['params']

xscaler = MinMaxScaler()
xscaler.fit(X)
x = xscaler.transform(X)

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


plt.tight_layout()
plt.show()