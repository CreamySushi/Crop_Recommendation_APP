# ---------------------- IMPORTS ------------------------
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import datetime

app = Flask(__name__)
CORS(app)


# ------------------------ FILE CONFIGURATION ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cr_xgbrfclassifier_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

# Account Key (Secret File)
PI_SECRET_PASSWORD = os.environ.get('PI_SECRET_TOKEN', 'Crop-recommendation-raspi-2026')
FIREBASE_KEY_PATH = '/etc/secrets/qacg-crop-recommendation-firebase-adminsdk-fbsvc-c573940045.json' 

# ------------------ INITIALIZATION ---------------------------
try:
    if not firebase_admin._apps:
        if os.path.exists(FIREBASE_KEY_PATH):
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred, {
                'projectId': 'qacg-crop-recommendation', 
            })
            db = firestore.client()
            print('Firebase connected')
    else:
        db = firestore.client()
except Exception as e:
    print(f"Setup Failed: {e}")
    db = None
    
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH) 
    print("Model and Encoder loaded successfully.")
except Exception as e:
    print(f" Error loading model: {e}")

# ------------------------------- ROUTING----------------------------

@app.route('/update_SensData', methods=['POST'])
def collect_sensor_data():
    try:
        data = request.get_json()
        
        if data.get('token') != PI_SECRET_PASSWORD:
            return jsonify({'error': 'Access Denied'}), 401
            
        if db is None:
            return jsonify({'error': 'Firebase server connection failed'}), 500
        
        sensor_data = {
            'N': data.get('N'),
            'P': data.get('P'),
            'K': data.get('K'),
            'pH': data.get('pH'),
            'Moisture': data.get('Moisture'),
        }
        sensor_data['timestamp'] = firestore.SERVER_TIMESTAMP
        db.collection('sensor_readings').add(sensor_data)
        return jsonify({'success': True, 'message': 'Data secured in Firestore'}), 200
    
    except Exception as e:
         return jsonify({'success': False, 'error': str(e)}), 500
        
@app.route('/', methods=['GET'])
def home():
    return "Crop Recommendation API is running!"

# EndPoint 
@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        data = request.get_json()
        
        n = data.get('N')
        p = data.get('P')
        k = data.get('K')
        ph = data.get('pH')
        moisture = data.get('Moisture')
        
        
        if None in (n, p, k, ph, moisture):
            return jsonify({'error': 'Missing sensor data. Please provide N, P, K, pH, and moisture.'}), 400

        
        features = pd.DataFrame([[n, p, k, ph, moisture]], columns=['N', 'P', 'K', 'pH', 'Moisture'])
        
        prediction_num = model.predict(features.values)[0]
        
        # Convert to original string name
        recommended_crop = encoder.inverse_transform([prediction_num])[0]
        
        return jsonify({
            'success': True,
            'recommended_crop': recommended_crop,
            'sensor_data_received': {
                'N': n,
                'P': p,
                'K': k,
                'pH': ph,
                'Moisture': moisture
            }
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
