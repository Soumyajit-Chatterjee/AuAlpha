from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
api = Api(app)

# Global variables for model
model = None
scaler = None
feature_names = None
model_loaded = False

def load_model():
    global model, scaler, feature_names, model_loaded
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        model_loaded = True
        print("✅ Model and preprocessing objects loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False

# Load model on startup
load_model()

class HeartDiseasePrediction(Resource):
    def post(self):
        try:
            if not model_loaded:
                return {
                    'error': 'Model not loaded properly. Please check server logs.',
                    'prediction': None,
                    'probability': None
                }, 503

            # Get JSON data from request
            data = request.get_json()
            
            # Validate required fields
            required_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                               'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                               'ca', 'thal']
            
            missing_features = [feature for feature in required_features if feature not in data]
            if missing_features:
                return {
                    'error': f'Missing features: {missing_features}',
                    'prediction': None,
                    'probability': None
                }, 400

            # Extract features in correct order
            features = [float(data[feature]) for feature in feature_names]
            
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            # Prepare response
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': 'High' if prediction == 1 else 'Low',
                'confidence': f"{probability:.2%}",
                'message': 'Heart disease risk detected' if prediction == 1 else 'Low risk of heart disease'
            }
            
            return jsonify(result)
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'prediction': None,
                'probability': None
            }, 500

class HealthCheck(Resource):
    def get(self):
        return {
            'status': 'healthy',
            'model_loaded': model_loaded,
            'service': 'Heart Disease Prediction API',
            'python_version': os.environ.get('PYTHON_VERSION', 'Unknown')
        }

# Add resources to API
api.add_resource(HeartDiseasePrediction, '/predict')
api.add_resource(HealthCheck, '/health')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
