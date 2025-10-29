from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)
api = Api(app)

# Load model files
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("✅ All model files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    model = None
    scaler = None
    feature_names = None

class Predict(Resource):
    def post(self):
        if model is None:
            return {'error': 'Model not loaded'}, 500
            
        try:
            data = request.get_json()
            
            # Extract features in correct order
            features = []
            for feature in feature_names:
                if feature in data:
                    features.append(float(data[feature]))
                else:
                    return {'error': f'Missing feature: {feature}'}, 400
            
            # Prepare data for prediction
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = int(model.predict(features_scaled)[0])
            probability = float(model.predict_proba(features_scaled)[0][1])
            
            return {
                'prediction': prediction,
                'probability': probability,
                'risk_level': 'High' if prediction == 1 else 'Low',
                'message': 'Heart disease risk detected' if prediction == 1 else 'Low risk of heart disease'
            }
            
        except Exception as e:
            return {'error': str(e)}, 500

class Health(Resource):
    def get(self):
        return {
            'status': 'running',
            'model_loaded': model is not None
        }

api.add_resource(Predict, '/predict')
api.add_resource(Health, '/health')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
