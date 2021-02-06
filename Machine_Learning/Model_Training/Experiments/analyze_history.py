from pandas import read_json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from logger import info

def get_history(filepath, i):
    filename = filepath + 'history_' + str(i) + '.json'
    info(f"Opening model history from {filename}")
    history_json = open(filename, 'r')
    history = read_json(history_json)
    history_json.close()

    return history



num_hidden = 4
num_models = 100

filepath = f"num_hidden/num_hidden_{num_hidden}/"

with PdfPages(f'evaluation/num_hidden_{num_hidden}_accuracy.pdf') as pdf:

    for i in range(1,num_models+1):
        history = get_history(filepath, i)

        training_acc   = history['accuracy']
        validation_acc = history['val_accuracy']
        epochs = np.arange(len(training_acc))

        fig, ax  = plt.subplots()
        ax. plot(epochs, training_acc, epochs, validation_acc, label=['Training Accuracy', 'Validation Accuracy'])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy [threshold = 0.5]")

        pdf.savefig(fig)
        plt.close()