import os
import pickle
from flask import Flask, render_template, request, jsonify

# Import the prediction function from the refactored predict.py
from predict import predict_gender, extract_features

app = Flask(__name__)

# Define file paths for the model and vectorizer
MODEL_PATH = "gender_predictor_model.pkl"
VECTORIZER_PATH = "gender_predictor_vectorizer.pkl"

# Load the model and vectorizer once when the app starts
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    # print("✅ Model and vectorizer loaded successfully.")
    is_model_loaded = True
except FileNotFoundError:
    print("❌ Error: Model files not found. Please run predict.py to train and save them.")
    is_model_loaded = False
    model, vectorizer = None, None

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not is_model_loaded:
        return jsonify({'error': 'Prediction model is not available.'}), 503 # Service Unavailable

    name = request.form.get('name')
    if not name:
        return jsonify({'error': 'Name is required'}), 400

    # Pass the loaded model, vectorizer, and the name to the prediction function
    prediction = predict_gender(MODEL_PATH, VECTORIZER_PATH, name)
    
    return jsonify({
        'name': name,
        'gender_prediction': prediction
    })

if __name__ == '__main__':
    app.run(port=3000, debug=True)
