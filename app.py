# ---------------------- IMPORTS ------------------------
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import hmac
import json
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)


# ------------------------ FILE CONFIGURATION ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'crop_recommendation_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

# Account Key (Secret File)
PI_SECRET_PASSWORD = os.environ.get('PI_SECRET_TOKEN', 'Crop-recommendation-raspi-2026')
FIREBASE_KEY_PATH = '/etc/secrets/qacg-crop-recommendation-firebase-adminsdk-fbsvc-c573940045.json' 

# ------------------ INITIALIZATION ---------------------------
# ------------------ Render Settings ------------------------
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

# ------------------ Railway Settings ------------------------
# try:
#     if not firebase_admin._apps:
#         # METHOD: Get JSON content directly from Variable string
#         firebase_json = os.environ.get('FIREBASE_CONFIG')
        
#         if firebase_json:
#             # Parse the string into a dictionary
#             cred_dict = json.loads(firebase_json)
#             cred = credentials.Certificate(cred_dict)
#             firebase_admin.initialize_app(cred, {
#                 'projectId': 'qacg-crop-recommendation', 
#             })
#             db = firestore.client()
#             print('Firebase connected via Environment Variable')
#         else:
#             print("CRITICAL: FIREBASE_CONFIG variable missing. Database disabled.")
#             db = None
#     else:
#         db = firestore.client()
# except Exception as e:
#     print(f"Setup Failed: {e}")
#     db = None
    
# try:
#     # Ensure the files exist before loading to avoid crash
#     if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
#         model = joblib.load(MODEL_PATH)
#         encoder = joblib.load(ENCODER_PATH) 
#         print("Model and Encoder loaded successfully.")
#     else:
#         print(f"Error: Model files not found at {MODEL_PATH}")
# except Exception as e:
#     print(f" Error loading model: {e}")

def has_zero_sensor_value(n, p, k, ph, moisture):
    return any(value == 0 for value in [n, p, k, ph, moisture])


def resolve_user_id_from_token(client_token, payload):
    if not client_token:
        return None

    # Primary path: per-user token lookup
    try:
        users = db.collection('users').where('apiToken', '==', client_token).limit(1).stream()
        matched_user = next(users, None)
        if matched_user is not None:
            return matched_user.id
    except Exception as e:
        print(f"Token lookup failed: {e}")
        return None

    # Legacy fallback path: static token + explicit userId
    if hmac.compare_digest(client_token, PI_SECRET_PASSWORD):
        legacy_user_id = str(payload.get('userId', '')).strip()
        if not legacy_user_id:
            return None

        try:
            user_doc = db.collection('users').document(legacy_user_id).get()
            if user_doc.exists:
                return legacy_user_id
        except Exception as e:
            print(f"Legacy userId lookup failed: {e}")
            return None

    return None

# ------------------------------- ROUTING----------------------------

@app.route('/update_SensData', methods=['POST'])
def collect_sensor_data():
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON format'}), 400
            
        if db is None:
            return jsonify({'error': 'Firebase server connection failed'}), 500

        client_token = str(data.get('token', '')).strip()
        owner_uid = resolve_user_id_from_token(client_token, data)
        if not owner_uid:
            if hmac.compare_digest(client_token, PI_SECRET_PASSWORD):
                return jsonify({'error': 'Legacy token requires valid userId in payload'}), 401
            return jsonify({'error': 'Access Denied'}), 401
        
        try:
            val_n = float(data.get('N'))
            val_p = float(data.get('P'))
            val_k = float(data.get('K'))
            val_ph = float(data.get('pH'))
            val_moisture = float(data.get('moisture', data.get('Moisture')))
   
            if not (0 <= val_ph <= 14): raise ValueError("pH out of bounds")
            if not (0 <= val_moisture <= 100): raise ValueError("Moisture out of bounds")
            if val_n < 0 or val_p < 0 or val_k < 0: raise ValueError("Macros cannot be negative")
            
        except (TypeError, ValueError) as e:
            return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

        sensor_data = {
            'userId': owner_uid,
            'N': val_n,
            'P': val_p,
            'K': val_k,
            'pH': val_ph,
            'moisture': val_moisture,
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        
        if has_zero_sensor_value(val_n, val_p, val_k, val_ph, val_moisture):
            sensor_data['cropLabel'] = None
            sensor_data['recommendationStatus'] = 'insufficient_data_zero_values'
        else:
            try:
                features = pd.DataFrame([[sensor_data['N'], sensor_data['P'], sensor_data['K'], sensor_data['pH'], sensor_data['moisture']]], columns=['N', 'P', 'K', 'pH', 'Moisture'])
                prediction_num = model.predict(features.values)[0]
                recommended_crop = encoder.inverse_transform([prediction_num])[0]
                sensor_data['cropLabel'] = recommended_crop
                sensor_data['recommendationStatus'] = 'ok'
            except Exception as pred_e:
                print(f"Prediction failed during sensor update: {pred_e}")
                sensor_data['cropLabel'] = 'Unknown'
                sensor_data['recommendationStatus'] = 'prediction_failed'

        db.collection('sensor_readings').add(sensor_data)
        
        return jsonify({
            'success': True, 
            'message': 'Data secured in Firestore',
            'userId': owner_uid,
            'recommended_crop': sensor_data.get('cropLabel'),
            'recommendation_status': sensor_data.get('recommendationStatus')
        }), 200
    
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
        if not data or not isinstance(data, dict):
             return jsonify({'error': 'Invalid JSON format'}), 400
        
        # Strict validation & type-casting
        try:
            n = float(data.get('N'))
            p = float(data.get('P'))
            k = float(data.get('K'))
            ph = float(data.get('pH'))
            moisture = float(data.get('moisture', data.get('Moisture')))
            
            if not (0 <= ph <= 14) or not (0 <= moisture <= 100) or n < 0 or p < 0 or k < 0:
                raise ValueError("Values out of reasonable bounds")
        except (TypeError, ValueError) as e:
            return jsonify({'error': f'Missing or invalid sensor data: {str(e)}'}), 400

        if has_zero_sensor_value(n, p, k, ph, moisture):
            return jsonify({
                'success': False,
                'recommended_crop': None,
                'error': 'Cannot recommend crop when any required factor is 0',
                'sensor_data_received': {
                    'N': n,
                    'P': p,
                    'K': k,
                    'pH': ph,
                    'Moisture': moisture
                }
            }), 400

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

@app.route('/crop_requirements/<crop_name>', methods=['GET'])
def get_crop_requirements(crop_name):
    # Standardize crop name to lowercase
    crop = crop_name.lower().strip()
    
    # Realistic requirements for crops known to our LabelEncoder
    CROP_REQUIREMENTS = {
        'banana': {'n': [80, 120], 'p': [70, 95], 'k': [45, 55], 'ph': [5.5, 7.0], 'moisture': [75, 85]},
        'calamansi': {'n': [90, 110], 'p': [40, 60], 'k': [40, 50], 'ph': [5.5, 6.5], 'moisture': [60, 80]},
        'chili': {'n': [60, 80], 'p': [40, 55], 'k': [40, 60], 'ph': [5.5, 6.8], 'moisture': [60, 70]},
        'coconut': {'n': [10, 40], 'p': [10, 30], 'k': [20, 50], 'ph': [5.2, 8.0], 'moisture': [60, 80]},
        'coffee': {'n': [80, 120], 'p': [15, 40], 'k': [25, 35], 'ph': [5.5, 7.0], 'moisture': [50, 70]},
        'garlic': {'n': [60, 80], 'p': [40, 60], 'k': [40, 60], 'ph': [6.0, 7.5], 'moisture': [50, 70]},
        'ginger': {'n': [60, 80], 'p': [30, 50], 'k': [30, 50], 'ph': [5.5, 6.5], 'moisture': [60, 80]},
        'jute': {'n': [60, 100], 'p': [35, 60], 'k': [35, 45], 'ph': [6.0, 7.5], 'moisture': [70, 90]},
        'kidneybeans': {'n': [20, 40], 'p': [50, 80], 'k': [15, 25], 'ph': [5.5, 6.0], 'moisture': [80, 90]},
        'maize': {'n': [60, 100], 'p': [35, 60], 'k': [15, 25], 'ph': [5.5, 7.0], 'moisture': [60, 80]},
        'mango': {'n': [0, 45], 'p': [15, 40], 'k': [25, 35], 'ph': [4.5, 6.5], 'moisture': [60, 80]},
        'mungbean': {'n': [10, 30], 'p': [40, 60], 'k': [15, 25], 'ph': [6.0, 7.0], 'moisture': [80, 90]},
        'muskmelon': {'n': [80, 120], 'p': [10, 30], 'k': [45, 55], 'ph': [6.0, 6.8], 'moisture': [80, 90]},
        'okra': {'n': [60, 80], 'p': [40, 60], 'k': [40, 60], 'ph': [6.0, 7.5], 'moisture': [60, 80]},
        'onion': {'n': [60, 80], 'p': [40, 60], 'k': [40, 60], 'ph': [6.0, 7.5], 'moisture': [50, 70]},
        'orange': {'n': [0, 40], 'p': [5, 30], 'k': [5, 20], 'ph': [6.0, 7.5], 'moisture': [80, 90]},
        'papaya': {'n': [30, 70], 'p': [40, 70], 'k': [40, 60], 'ph': [6.0, 7.0], 'moisture': [80, 90]},
        'pigeonpeas': {'n': [10, 40], 'p': [50, 80], 'k': [15, 25], 'ph': [5.5, 7.0], 'moisture': [80, 90]},
        'pomegranate': {'n': [0, 40], 'p': [5, 30], 'k': [35, 45], 'ph': [5.5, 7.5], 'moisture': [60, 80]},
        'rice': {'n': [60, 100], 'p': [35, 60], 'k': [35, 45], 'ph': [5.0, 7.5], 'moisture': [80, 100]},
        'tomato': {'n': [40, 80], 'p': [20, 50], 'k': [20, 50], 'ph': [5.5, 7.0], 'moisture': [60, 80]},
        'watermelon': {'n': [80, 120], 'p': [10, 30], 'k': [45, 55], 'ph': [6.0, 7.0], 'moisture': [80, 90]}
    }
    
    DEFAULT_REQ = {'n': [40, 80], 'p': [20, 50], 'k': [20, 50], 'ph': [5.5, 7.0], 'moisture': [50, 80]}
    
    reqs = CROP_REQUIREMENTS.get(crop, DEFAULT_REQ)
    return jsonify({
        'success': True,
        'crop': crop.capitalize(),
        'requirements': reqs
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    
