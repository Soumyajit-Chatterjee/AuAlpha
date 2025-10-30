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
    print(f"📋 Features: {feature_names}")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    model = None
    scaler = None
    feature_names = None

def get_risk_level(probability):
    """Enhanced risk classification with all 5 levels"""
    if probability < 0.15:
        return "Very Low"
    elif probability < 0.35:
        return "Low" 
    elif probability < 0.55:
        return "Moderate"
    elif probability < 0.75:
        return "High"
    else:
        return "Very High"

def get_risk_message(risk_level, probability):
    """Get appropriate message based on risk level"""
    messages = {
        "Very Low": f"Excellent heart health! Very low risk ({probability:.1%}) of heart disease. Maintain your healthy lifestyle.",
        "Low": f"Good heart health. Low risk ({probability:.1%}) of heart disease. Regular checkups recommended.",
        "Moderate": f"Moderate risk ({probability:.1%}) detected. Consider lifestyle improvements and consult a healthcare professional.",
        "High": f"High risk ({probability:.1%}) of heart disease. Recommended to consult a cardiologist for evaluation.",
        "Very High": f"Very high risk ({probability:.1%}) detected! Immediate medical consultation strongly recommended."
    }
    return messages.get(risk_level, f"Risk assessment completed. Probability: {probability:.1%}")

class Predict(Resource):
    def post(self):
        if model is None:
            return {'error': 'Model not loaded. Please check server logs.'}, 500
            
        try:
            data = request.get_json()
            
            if not data:
                return {'error': 'No JSON data provided'}, 400
            
            # Extract features in correct order
            features = []
            missing_features = []
            
            for feature in feature_names:
                if feature in data:
                    try:
                        features.append(float(data[feature]))
                    except (ValueError, TypeError):
                        return {'error': f'Invalid value for {feature}. Must be a number.'}, 400
                else:
                    missing_features.append(feature)
            
            if missing_features:
                return {'error': f'Missing features: {missing_features}'}, 400
            
            # Prepare data for prediction
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = int(model.predict(features_scaled)[0])
            probability = float(model.predict_proba(features_scaled)[0][1])
            
            # Enhanced risk classification (based on probability only)
            risk_level = get_risk_level(probability)
            message = get_risk_message(risk_level, probability)
            
            response = {
                'prediction': prediction,
                'probability': probability,
                'risk_level': risk_level,
                'confidence': f"{probability:.1%}",
                'message': message,
                'interpretation': '0 = No heart disease, 1 = Heart disease present'
            }
            
            print(f"📊 Prediction: {prediction}, Probability: {probability:.3f}, Risk: {risk_level}")
            return response
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {'error': f'Prediction failed: {str(e)}'}, 500

class Health(Resource):
    def get(self):
        return {
            'status': 'running',
            'model_loaded': model is not None,
            'features_loaded': feature_names is not None,
            'total_features': len(feature_names) if feature_names else 0,
            'model_type': str(type(model).__name__) if model else 'None'
        }

# Test endpoint to verify all risk levels
class Test(Resource):
    def get(self):
        if model is None:
            return {'error': 'Model not loaded'}, 500
            
        # Test cases for all risk levels
        test_cases = {
            'Very Low Risk': {
                "age": 35, "sex": 0, "cp": 0, "trestbps": 110, 
                "chol": 160, "fbs": 0, "restecg": 0, "thalach": 180,
                "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 1
            },
            'Low Risk': {
                "age": 45, "sex": 0, "cp": 1, "trestbps": 120, 
                "chol": 180, "fbs": 0, "restecg": 0, "thalach": 160,
                "exang": 0, "oldpeak": 0.8, "slope": 1, "ca": 0, "thal": 2
            },
            'Moderate Risk': {
                "age": 55, "sex": 1, "cp": 2, "trestbps": 140, 
                "chol": 220, "fbs": 0, "restecg": 1, "thalach": 140,
                "exang": 1, "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
            },
            'High Risk': {
                "age": 65, "sex": 1, "cp": 3, "trestbps": 180, 
                "chol": 280, "fbs": 1, "restecg": 1, "thalach": 120,
                "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 3
            },
            'Very High Risk': {
                "age": 70, "sex": 1, "cp": 3, "trestbps": 200, 
                "chol": 300, "fbs": 1, "restecg": 2, "thalach": 100,
                "exang": 1, "oldpeak": 4.0, "slope": 2, "ca": 3, "thal": 3
            }
        }
        
        results = {}
        for case_name, test_data in test_cases.items():
            try:
                features = [float(test_data[feature]) for feature in feature_names]
                features_array = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_array)
                
                prediction = int(model.predict(features_scaled)[0])
                probability = float(model.predict_proba(features_scaled)[0][1])
                risk_level = get_risk_level(probability)
                
                results[case_name] = {
                    'prediction': prediction,
                    'probability': probability,
                    'risk_level': risk_level,
                    'expected_risk': case_name.split(' ')[0]  # Get "Very", "Low", etc.
                }
            except Exception as e:
                results[case_name] = {'error': str(e)}
        
        return {'test_results': results}

# Sample data endpoint for frontend reference
class Samples(Resource):
    def get(self):
        samples = {
            'very_low_risk': {
                "age": 35, "sex": 0, "cp": 0, "trestbps": 110, 
                "chol": 160, "fbs": 0, "restecg": 0, "thalach": 180,
                "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 1
            },
            'low_risk': {
                "age": 45, "sex": 0, "cp": 1, "trestbps": 120, 
                "chol": 180, "fbs": 0, "restecg": 0, "thalach": 160,
                "exang": 0, "oldpeak": 0.8, "slope": 1, "ca": 0, "thal": 2
            },
            'moderate_risk': {
                "age": 55, "sex": 1, "cp": 2, "trestbps": 140, 
                "chol": 220, "fbs": 0, "restecg": 1, "thalach": 140,
                "exang": 1, "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
            },
            'high_risk': {
                "age": 65, "sex": 1, "cp": 3, "trestbps": 180, 
                "chol": 280, "fbs": 1, "restecg": 1, "thalach": 120,
                "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 3
            },
            'very_high_risk': {
                "age": 70, "sex": 1, "cp": 3, "trestbps": 200, 
                "chol": 300, "fbs": 1, "restecg": 2, "thalach": 100,
                "exang": 1, "oldpeak": 4.0, "slope": 2, "ca": 3, "thal": 3
            }
        }
        return {'sample_cases': samples}

api.add_resource(Predict, '/predict')
api.add_resource(Health, '/health')
api.add_resource(Test, '/test')
api.add_resource(Samples, '/samples')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
