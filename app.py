# ---------------------- IMPORTS ------------------------
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import hmac
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

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
db = None
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
            print(f"Firebase key not found at: {FIREBASE_KEY_PATH}")
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


def resolve_user_id_from_token(client_token, payload):
    """
    Resolve user id from immutable per-user apiToken in Firestore.
    Temporary migration fallback:
    - If legacy env token is used, require payload['userId'] and verify user exists.
    """
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
        
        try:
             features = pd.DataFrame([[sensor_data['N'], sensor_data['P'], sensor_data['K'], sensor_data['pH'], sensor_data['moisture']]], columns=['N', 'P', 'K', 'pH', 'Moisture'])
             prediction_num = model.predict(features.values)[0]
             recommended_crop = encoder.inverse_transform([prediction_num])[0]
             sensor_data['cropLabel'] = recommended_crop
        except Exception as pred_e:
             print(f"Prediction failed during sensor update: {pred_e}")
             sensor_data['cropLabel'] = 'Unknown'

        db.collection('sensor_readings').add(sensor_data)
        
        return jsonify({
            'success': True, 
            'message': 'Data secured in Firestore',
            'userId': owner_uid,
            'recommended_crop': sensor_data.get('cropLabel')
        }), 200
    
    except Exception as e:
         return jsonify({'success': False, 'error': str(e)}), 500
        
@app.route('/', methods=['GET'])
def home():
    return "Crop Recommendation API is running!"


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True}), 200

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
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=False)
