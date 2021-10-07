from keras.models import model_from_json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress Keras/TF warnings
from pickle import load
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_model(location, tag, get_history=False):
    try:
        json_file = open(location + f'models/{tag}/model/model_1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(location + f'models/{tag}/model/model_1.h5')
        scaler = load(open(location + f'models/{tag}/model/scaler_1.pkl', 'rb'))
    except:
        json_file = open(location + f'models/{tag}/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        history_json_file = open(location + f'models/{tag}/history.json', 'r')
        loaded_history_json = history_json_file.read()
        history_json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        history = model_from_json(loaded_history_json)
        loaded_model.load_weights(location + f'models/{tag}/model.h5')
        scaler = load(open(location + f'models/{tag}/scaler.pkl', 'rb'))

    if get_history: return history
    return scaler, loaded_model