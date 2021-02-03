from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_roc(y, predictions):
    fpr, tpr, thresholds_keras = roc_curve(y, predictions)
    return fpr, tpr, auc(fpr, tpr)

num_hidden = 1
num_models = 100


predictions = np.load(f"evaluation/hidden_layers_{num_hidden}_predictions.npz")

with PdfPages(f'evaluation/num_hidden_{num_hidden}_roc.pdf') as pdf:

    for i in range(1,num_models+1):
        examples = np.load(f"num_hidden/num_hidden_{num_hidden}/test_set_{i}.npz")
        print(examples.files)
        y = examples['y_test']
        fpr, tpr, nn_auc = get_roc(y[i,:], predictions[i,:])

        fig = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(nn_auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc=4)

        pdf.savefig(fig)
        plt.close()
