import numpy as np
from logger import info
from keras.models import model_from_json

def return_predictions(nmodels, tag, pred_file=False, nlayers=4):

    dirname = f'layers/layers_{nlayers}/{tag}/'
    dir_model = dirname + 'model/'
    if pred_file:
        predictions = np.load(dirname + 'evaluation/test_predictions.npz')
        return predictions['predictions']

    else:
        predictions = np.array(())
        high_peak = np.array(())

        for i in np.arange(1, int(nmodels) + 1):

            # Load inputs
            inputs = np.load(dir_model + f'test_set_{i}.npz')

            x = inputs['x_test']
            y = inputs['y_test']
            X = inputs['X_test']
            m = inputs['mjj_test']

            info(f"Evaluating model {dir_model + 'model_' + str(i) + '.json'}")

            # load json and create model
            model_json_file = open(dir_model + 'model_' + str(i) + '.json', 'r')
            model_json = model_json_file.read()
            model_json_file.close()
            model = model_from_json(model_json)

            # load weights into new model
            model.load_weights(dir_model + 'model_' + str(i) + '.h5')

            pred = model.predict(x)
            pred = pred.ravel()

            predictions = np.append(predictions, pred)

            bins = np.linspace(0,1,100)
            n, edges = np.histogram(pred, bins=bins)
            a = 10 # arbitrarily chosen
            pos_of_peak = np.argmax(n[a:]) + a # Need to skip the peak at 0, which is higher than the peak near 1. 
            high_peak = np.append(high_peak, (edges[pos_of_peak] + edges[pos_of_peak + 1]) / 2)

        # predictions = predictions.reshape(nmodels, len(x))

        # pred_save = f"evaluation/1_hidden_layer_{tag}_predictions.npz"
        # info(f"Saving .npz file of predictions to {pred_save}")
        # np.savez(pred_save, predictions=predictions)

    return predictions, high_peak



preds = return_predictions(15, tag='smeared_7param_dR')
print(preds.ravel())
print(np.shape(preds.ravel()))
mask = preds < 0.01
preds[mask]