from flask import Flask
from flask import request
import pickle
import os
import numpy as np

from src.models.model_functions import predict, load_model, load_processed_data
from src.data.data_functions import _normalize

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Welcome!'


@app.route('/get_prediction', methods = ['POST'])
def get_prediction():
    print(request.is_json)
    content = request.json

    preprocess_settings = pickle.load(open(os.path.join(os.getcwd(), "data", "processed", "preprocess_settings.pickle"), 'rb'))
    pp_mean = preprocess_settings['preprocess_mean']
    pp_std = preprocess_settings['preprocess_std']

    input_data = np.array(content).reshape(-1, 3, 32, 32)
    input_data = _normalize(input_data, pp_mean, pp_std)

    label_to_names = load_processed_data()['label_to_names']

    model = load_model()
    prediction = predict(model, input_data)

    return str(label_to_names[prediction[0]])


if __name__ == '__main__':
   app.run()