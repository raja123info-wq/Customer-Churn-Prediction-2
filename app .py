
import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os

# --- Configuration ---
MODEL_PATH = 'churn_model.pkl'

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load the model ---
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model is saved.")
    model = None # Set model to None if not found
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the expected feature columns based on your training data (from `X.columns` in original notebook)
# This is a crucial part: the input data to the API must match the training data's columns
# For simplicity, we are listing them here. In a real application, you might infer them or embed them.
EXPECTED_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure churn_model.pkl exists.'}), 500

    try:
        data = request.get_json(force=True)

        # Convert input data to a Pandas DataFrame
        # Ensure the order of columns matches the training data
        input_df = pd.DataFrame([data], columns=EXPECTED_FEATURES)

        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]

        return jsonify({
            'prediction': int(prediction[0]), # 1 for churn, 0 for no churn
            'churn_probability': float(probability[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return "Welcome to the Customer Churn Prediction API! Send POST requests to /predict."

if __name__ == '__main__':
    # To run this locally, you would typically use:
    # app.run(debug=True, host='0.0.0.0', port=5000)
    # For Colab deployment, you might use ngrok or similar, or deploy to a platform.
    print("To run this application, save it, then execute `python app.py` in your terminal.")
    print("Or, for local testing, if you have Flask installed: `flask run`")
    print("Remember to have 'churn_model.pkl' in the same directory.")
