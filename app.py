from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'knn_applequality_model.pkl')

with open(model_path, 'rb') as file:
    knn_model = pickle.load(file)


# install flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input_features = np.array([[
        float(data['Size']),
        float(data['Weight']),
        float(data['Sweetness']),
        float(data['Crunchiness']),
        float(data['Juiciness']),
        float(data['Ripeness']),
        float(data['Acidity'])
    ]])
    
    prediction = knn_model.predict(input_features)
    
    if prediction[0] == 1:
        result = 'Apel Bagus'
    else:
        result = 'Apel Tidak Bagus'
    
    return render_template('index.html', prediction_text=f'{result}')

if __name__ == '__main__':
    app.run(debug=True)
