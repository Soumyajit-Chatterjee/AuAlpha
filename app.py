from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)
api = Api(app)

# Load enhanced Framingham model files
try:
    model = joblib.load('framingham_heart_model.pkl')
    scaler = joblib.load('framingham_scaler.pkl')
    feature_names = joblib.load('framingham_feature_names.pkl')
    metadata = joblib.load('model_metadata.pkl')
    
    print("✅ Enhanced Framingham model loaded successfully!")
    print(f"📊 Model Info: {metadata}")
    print(f"📋 Features ({len(feature_names)}): {feature_names}")
except Exception as e:
    print(f"❌ Error loading enhanced model: {e}")
    model = None
    scaler = None
    feature_names = None
    metadata = None

def get_risk_level(probability):
    """Enhanced risk classification for 10-year CHD risk"""
    if probability < 0.10:
        return "Very Low"
    elif probability < 0.20:
        return "Low"
    elif probability < 0.30:
        return "Moderate"
    elif probability < 0.40:
        return "High"
    else:
        return "Very High"

class Predict(Resource):
    def post(self):
        if model is None:
            return {'error': 'Enhanced Framingham model not loaded'}, 500
            
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
            
            # Make prediction - Framingham predicts 10-year CHD risk
            probability = float(model.predict_proba(features_scaled)[0][1])
            risk_level = get_risk_level(probability)
            
            # Enhanced messages based on 10-year CHD risk
            messages = {
                "Very Low": "Excellent! Your 10-year heart disease risk is very low. Maintain your healthy lifestyle.",
                "Low": "Good! Your 10-year heart disease risk is low. Continue with regular health checkups.",
                "Moderate": "Moderate 10-year heart disease risk detected. Consider lifestyle improvements and consult your doctor.",
                "High": "High 10-year heart disease risk detected. Recommended to consult a healthcare professional for evaluation.",
                "Very High": "Very high 10-year heart disease risk detected. Strongly recommend immediate medical consultation and lifestyle changes."
            }
            
            response = {
                'probability': probability,
                'risk_level': risk_level,
                'confidence': f"{probability:.1%}",
                'message': messages.get(risk_level, "10-year heart disease risk assessment complete."),
                'timeframe': '10-year CHD risk',
                'model_version': 'framingham_enhanced_v1',
                'prediction': 1 if probability > 0.20 else 0  # Binary prediction for compatibility
            }
            
            print(f"📊 Prediction - Probability: {probability:.3f}, Risk: {risk_level}")
            return response
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {'error': f'Prediction failed: {str(e)}'}, 500

class Health(Resource):
    def get(self):
        return {
            'status': 'running',
            'model_loaded': model is not None,
            'model_type': metadata.get('best_model', 'Unknown') if metadata else 'Unknown',
            'dataset': metadata.get('dataset', 'Unknown') if metadata else 'Unknown',
            'accuracy': metadata.get('accuracy', 0) if metadata else 0,
            'feature_count': len(feature_names) if feature_names else 0,
            'service': 'Framingham Heart Disease Prediction API'
        }

# Test endpoint with Framingham features
class Test(Resource):
    def get(self):
        if model is None:
            return {'error': 'Model not loaded'}, 500
            
        # Test cases for Framingham features
        test_cases = {
            'Low Risk Profile': {
                "male": 0,  # Female
                "age": 45,
                "education": 2,
                "currentSmoker": 0,
                "cigsPerDay": 0,
                "BPMeds": 0,
                "prevalentStroke": 0,
                "prevalentHyp": 0,
                "diabetes": 0,
                "totChol": 180,
                "sysBP": 110,
                "diaBP": 70,
                "BMI": 22,
                "heartRate": 65,
                "glucose": 80,
                "mean_BP": 83.3,  # (sysBP + 2*diaBP)/3
                "bmi_age": 9.9    # (BMI * age)/100
            },
            'High Risk Profile': {
                "male": 1,  # Male
                "age": 65,
                "education": 1,
                "currentSmoker": 1,
                "cigsPerDay": 20,
                "BPMeds": 1,
                "prevalentStroke": 0,
                "prevalentHyp": 1,
                "diabetes": 1,
                "totChol": 280,
                "sysBP": 160,
                "diaBP": 100,
                "BMI": 30,
                "heartRate": 80,
                "glucose": 150,
                "mean_BP": 120.0,  # (160 + 2*100)/3
                "bmi_age": 19.5    # (30 * 65)/100
            }
        }
        
        results = {}
        for case_name, test_data in test_cases.items():
            try:
                features = [float(test_data[feature]) for feature in feature_names]
                features_array = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_array)
                
                probability = float(model.predict_proba(features_scaled)[0][1])
                risk_level = get_risk_level(probability)
                
                results[case_name] = {
                    'probability': probability,
                    'risk_level': risk_level,
                    'confidence': f"{probability:.1%}"
                }
            except Exception as e:
                results[case_name] = {'error': str(e)}
        
        return {'test_results': results}

# Sample data endpoint for frontend reference
class Samples(Resource):
    def get(self):
        samples = {
            'low_risk_sample': {
                "male": 0,  # Female
                "age": 45,
                "education": 2,
                "currentSmoker": 0,
                "cigsPerDay": 0,
                "BPMeds": 0,
                "prevalentStroke": 0,
                "prevalentHyp": 0,
                "diabetes": 0,
                "totChol": 180,
                "sysBP": 110,
                "diaBP": 70,
                "BMI": 22,
                "heartRate": 65,
                "glucose": 80,
                "mean_BP": 83.3,
                "bmi_age": 9.9
            },
            'high_risk_sample': {
                "male": 1,  # Male
                "age": 65,
                "education": 1,
                "currentSmoker": 1,
                "cigsPerDay": 20,
                "BPMeds": 1,
                "prevalentStroke": 0,
                "prevalentHyp": 1,
                "diabetes": 1,
                "totChol": 280,
                "sysBP": 160,
                "diaBP": 100,
                "BMI": 30,
                "heartRate": 80,
                "glucose": 150,
                "mean_BP": 120.0,
                "bmi_age": 19.5
            }
        }
        return {'sample_cases': samples}

# Feature info endpoint
class Features(Resource):
    def get(self):
        if not feature_names:
            return {'error': 'Features not loaded'}, 500
            
        feature_info = {
            'male': 'Biological sex (1 = Male, 0 = Female)',
            'age': 'Age in years',
            'education': 'Education level (1-4)',
            'currentSmoker': 'Current smoking status (1 = Yes, 0 = No)',
            'cigsPerDay': 'Cigarettes per day',
            'BPMeds': 'Blood pressure medication (1 = Yes, 0 = No)',
            'prevalentStroke': 'Previous stroke (1 = Yes, 0 = No)',
            'prevalentHyp': 'Hypertension (1 = Yes, 0 = No)',
            'diabetes': 'Diabetes (1 = Yes, 0 = No)',
            'totChol': 'Total cholesterol (mg/dL)',
            'sysBP': 'Systolic blood pressure (mm Hg)',
            'diaBP': 'Diastolic blood pressure (mm Hg)',
            'BMI': 'Body Mass Index',
            'heartRate': 'Heart rate (bpm)',
            'glucose': 'Blood glucose (mg/dL)',
            'mean_BP': 'Mean blood pressure (calculated)',
            'bmi_age': 'BMI-age interaction (calculated)'
        }
        
        return {
            'features': feature_names,
            'feature_descriptions': feature_info,
            'total_features': len(feature_names)
        }

api.add_resource(Predict, '/predict')
api.add_resource(Health, '/health')
api.add_resource(Test, '/test')
api.add_resource(Samples, '/samples')
api.add_resource(Features, '/features')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
