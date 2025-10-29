from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

# Load the model and preprocessing objects
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

class HeartDiseasePrediction(Resource):
    def post(self):
        try:
            if model is None or scaler is None:
                return {
                    'error': 'Model not loaded properly',
                    'prediction': None,
                    'probability': None
                }, 500

            # Get JSON data from request
            data = request.get_json()
            
            # Extract features in correct order
            features = []
            for feature in feature_names:
                if feature in data:
                    features.append(float(data[feature]))
                else:
                    return {
                        'error': f'Missing feature: {feature}',
                        'prediction': None,
                        'probability': None
                    }, 400

            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features (assuming the best model requires scaling)
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
                'error': str(e),
                'prediction': None,
                'probability': None
            }, 500

class HealthCheck(Resource):
    def get(self):
        return {
            'status': 'healthy',
            'model_loaded': model is not None,
            'service': 'Heart Disease Prediction API'
        }

# Add resources to API
api.add_resource(HeartDiseasePrediction, '/predict')
api.add_resource(HealthCheck, '/health')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
