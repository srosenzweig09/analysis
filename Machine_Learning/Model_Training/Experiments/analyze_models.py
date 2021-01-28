
# open model
from keras.models import model_from_json
from pandas import read_json
from pickle import load
import numpy as np
from tensorflow import compat
compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)


# extract metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# plot metrics
import matplotlib.pyplot as plt 
# plt.style.use('../../Config/plots.mplstyle')

N = 3

filepath = f'hidden_layers/hidden_layers_{4}/'
scaler = load(open(filepath + 'scaler.pkl', 'rb'))

# fig1, axs1 = plt.subplots(nrows=6, ncols=5, figsize=(20,20))
# fig2, axs2 = plt.subplots(nrows=6, ncols=5, figsize=(20,20))

auc_scores = np.array(())
predictions = np.array(())

for i in np.arange(1,31):

    # if i == 18 or i == 20: continue

    # load json and create model
    model_json_file = open(filepath + 'model_' + str(i) + '.json', 'r')
    model_json = model_json_file.read()
    model_json_file.close()
    model = model_from_json(model_json)

    history_json_file = open(filepath + 'history_' + str(i) + '.json', 'r')
    history = read_json(history_json_file)
    history_json_file.close()
    print(history.keys)

    # load weights into new model
    model.load_weights(filepath + 'model_' + str(i) + '.h5')

    print(f"{i}: [INFO] Loaded model and scaler from disk")

    # Load inputs
    inputs = np.load(filepath + 'test_set_' + str(i) + '.npz')

    x = inputs['x_test']
    y = inputs['y_test']
    X = inputs['X_test']
    m = inputs['mjj_test']

    # training_accuracy = history['accuracy']
    # validation_accuracy = history['val_accuracy']

    # num_epochs = len(training_accuracy)
    # epoch_count = range(1, num_epochs + 1)
    # if num_epochs <= 65: 
    #     tick_max = num_epochs + (num_epochs % 5)
    #     epoch_ticks = np.arange(0, tick_max + 1, 5)
    # else: 
    #     tick_max = num_epochs + (num_epochs % 10)
    #     epoch_ticks = np.arange(0, tick_max + 1, 10)

    # ax = axs1[(i-1)//5, (i-1)%5] 
    # ax.plot(epoch_count, training_accuracy, 'r--')
    # ax.plot(epoch_count, validation_accuracy, 'b-')
    # ax.legend(['Training Accuracy', 'Validation Accuracy'])
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Accuracy')
    # ax.set_ylim(0.55,1.00)
    # ax.set_xticks(epoch_ticks)
    # plt.tight_layout()

    # training_loss = history['loss']
    # validation_loss = history['val_loss']

    # ax = axs2[(i-1)//5, (i-1)%5] 
    # ax.plot(epoch_count, training_loss, 'r--')
    # ax.plot(epoch_count, validation_loss, 'b-')
    # ax.legend(['Training Loss', 'Validation Loss'])
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Loss')
    # ax.set_xticks(epoch_ticks)
    # plt.tight_layout()

    pred = model.predict(x)
    predictions = np.append(predictions, pred)

    fpr, tpr, thresholds_keras = roc_curve(y, pred)
    nn_auc = auc(fpr, tpr)

    auc_scores = np.append(auc_scores, nn_auc)

# fig3, ax = plt.subplots()
# ax.hist(auc_scores, bins=20, histtype='step', align='mid')
# ax.set_xlabel('AUC')
# ax.set_ylabel('Count')
# fig3.savefig('auc_dist.pdf', bbox_inches='tight')

# fig1.savefig(f'evaluation/hidden_layers_{N}_repeat_acc.pdf', bbox_inches='tight')
# fig2.savefig(f'evaluation/hidden_layers_{N}_repeat_loss.pdf', bbox_inches='tight')

# print(f'auc_scores = \n {auc_scores}\n')
# print(f'Average = {np.average(auc_scores)*100:.1f}%')

predictions = predictions.shape(30,len(x))

fig4, ax = plt.subplots()
ax.hist(predictions, bins=20, histtype='step', align='mid')
ax.set_xlabel('Classification Score')
ax.set_ylabel('Count')
fig3.savefig('evaluation/hidden_layers_4_score_dist.pdf', bbox_inches='tight')