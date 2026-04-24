# ---------------------- IMPORTS ------------------------
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import hmac
import json
import joblib
import pandas as pd
import xgboost as xgb
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)


# ------------------------ FILE CONFIGURATION ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'crop_recommendation_xgbrf_model.json')
LEGACY_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'crop_recommendation_xgbrf_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

# Account Key (Secret File)
PI_SECRET_PASSWORD = os.environ.get('PI_SECRET_TOKEN', 'Crop-recommendation-raspi-2026')
FIREBASE_KEY_PATH = '/etc/secrets/qacg-crop-recommendation-firebase-adminsdk-fbsvc-c573940045.json' 

# ------------------------ DATASET CONFIGURATION ------------------------
# No more hardcoded crop list in code.
# Sources are loaded in priority order:
# 1) model labels from encoder (auto-includes newly trained crops)
# 2) JSON metadata file (optional)
# 3) Firestore `crop_metadata` collection (optional)
DEFAULT_REQ = {
    'n': [40.0, 80.0],
    'p': [20.0, 50.0],
    'k': [20.0, 50.0],
    'ph': [5.5, 7.0],
    'moisture': [50.0, 80.0],
}

CROP_METADATA_PATH = os.environ.get(
    'CROP_METADATA_PATH',
    os.path.join(BASE_DIR, 'crop_metadata.json'),
)
CROP_METADATA_COLLECTION = os.environ.get('CROP_METADATA_COLLECTION', 'crop_metadata')
CROP_DATASET = {}

# ------------------ INITIALIZATION ---------------------------
# ------------------ Render Settings ------------------------
db = None
model = None
encoder = None


def load_crop_model():
    if os.path.exists(MODEL_PATH):
        loaded_model = xgb.XGBRFClassifier()
        loaded_model.load_model(MODEL_PATH)
        return loaded_model

    if os.path.exists(LEGACY_MODEL_PATH):
        return joblib.load(LEGACY_MODEL_PATH)

    return None

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
            print('Firebase key file not found. Database disabled.')
    else:
        db = firestore.client()
except Exception as e:
    print(f"Setup Failed: {e}")
    db = None
    
try:
    model = load_crop_model()
    encoder = joblib.load(ENCODER_PATH) 
    print("Model and Encoder loaded successfully.")
except Exception as e:
    print(f" Error loading model: {e}")


def has_zero_sensor_value(n, p, k, ph, moisture):
    return any(value == 0 for value in [n, p, k, ph, moisture])


def normalize_crop_name(crop_name):
    return str(crop_name or '').strip().lower()


def copy_requirements(requirements):
    return {
        'n': [float(requirements['n'][0]), float(requirements['n'][1])],
        'p': [float(requirements['p'][0]), float(requirements['p'][1])],
        'k': [float(requirements['k'][0]), float(requirements['k'][1])],
        'ph': [float(requirements['ph'][0]), float(requirements['ph'][1])],
        'moisture': [float(requirements['moisture'][0]), float(requirements['moisture'][1])],
    }


def build_default_crop_entry(crop_name):
    display_name = str(crop_name or '').strip() or 'Unknown'
    return {
        'displayName': display_name,
        'category': 'General',
        'requirements': copy_requirements(DEFAULT_REQ),
    }


def coerce_requirements(raw_requirements):
    if not isinstance(raw_requirements, dict):
        return copy_requirements(DEFAULT_REQ)

    parsed = {}
    for key in ['n', 'p', 'k', 'ph', 'moisture']:
        value = raw_requirements.get(key)
        if isinstance(value, list) and len(value) == 2:
            try:
                low = float(value[0])
                high = float(value[1])
                if low > high:
                    low, high = high, low
                parsed[key] = [low, high]
                continue
            except (TypeError, ValueError):
                pass
        parsed[key] = [float(DEFAULT_REQ[key][0]), float(DEFAULT_REQ[key][1])]

    return parsed


def sanitize_dataset_entry(crop_key, raw_entry):
    fallback = build_default_crop_entry(crop_key)
    normalized_key = normalize_crop_name(crop_key)

    if not normalized_key:
        normalized_key = normalize_crop_name(fallback['displayName'])

    if not isinstance(raw_entry, dict):
        return normalized_key, fallback

    display_name = str(
        raw_entry.get('displayName')
        or raw_entry.get('name')
        or fallback['displayName']
    ).strip() or fallback['displayName']

    category = str(raw_entry.get('category') or fallback['category']).strip() or 'General'
    requirements = coerce_requirements(raw_entry.get('requirements'))

    return normalized_key, {
        'displayName': display_name,
        'category': category,
        'requirements': requirements,
    }


def load_dataset_from_encoder_labels():
    dataset = {}
    if encoder is None:
        return dataset

    try:
        for raw_label in list(getattr(encoder, 'classes_', [])):
            label = str(raw_label or '').strip()
            if not label:
                continue
            key, entry = sanitize_dataset_entry(label, {'displayName': label})
            if key:
                dataset[key] = entry
        if dataset:
            print(f"Loaded {len(dataset)} crop labels from encoder.")
    except Exception as e:
        print(f"Failed loading crop labels from encoder: {e}")

    return dataset


def load_dataset_from_json(path):
    dataset = {}

    if not path or not os.path.exists(path):
        return dataset

    try:
        with open(path, 'r', encoding='utf-8') as fp:
            payload = json.load(fp)

        if isinstance(payload, dict):
            for key, value in payload.items():
                normalized_key, entry = sanitize_dataset_entry(key, value)
                if normalized_key:
                    dataset[normalized_key] = entry
        elif isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                key = item.get('key') or item.get('name') or item.get('displayName')
                normalized_key, entry = sanitize_dataset_entry(key, item)
                if normalized_key:
                    dataset[normalized_key] = entry

        if dataset:
            print(f"Loaded {len(dataset)} crop metadata entries from JSON.")
    except Exception as e:
        print(f"Failed loading crop metadata JSON: {e}")

    return dataset


def load_dataset_from_firestore():
    dataset = {}

    if db is None:
        return dataset

    try:
        docs = db.collection(CROP_METADATA_COLLECTION).stream()
        for doc in docs:
            item = doc.to_dict() or {}
            key = item.get('key') or item.get('name') or item.get('displayName') or doc.id
            normalized_key, entry = sanitize_dataset_entry(key, item)
            if normalized_key:
                dataset[normalized_key] = entry

        if dataset:
            print(f"Loaded {len(dataset)} crop metadata entries from Firestore.")
    except Exception as e:
        print(f"Failed loading crop metadata Firestore collection: {e}")

    return dataset


def refresh_crop_dataset():
    global CROP_DATASET

    dataset = {}

    # Base source: encoder classes (auto-adds new crops when model is retrained)
    dataset.update(load_dataset_from_encoder_labels())

    # Optional overrides/additions from external metadata sources
    dataset.update(load_dataset_from_json(CROP_METADATA_PATH))
    dataset.update(load_dataset_from_firestore())

    CROP_DATASET = dataset
    print(f"Crop dataset ready with {len(CROP_DATASET)} entries.")


def ensure_crop_dataset_loaded():
    if not CROP_DATASET:
        refresh_crop_dataset()


def get_crop_display_name(crop_key):
    ensure_crop_dataset_loaded()
    normalized_key = normalize_crop_name(crop_key)
    item = CROP_DATASET.get(normalized_key)

    if not item:
        raw_value = str(crop_key or '').strip()
        return raw_value if raw_value else 'Unknown'

    return item.get('displayName', str(crop_key or '').strip())


def get_crop_category(crop_key):
    ensure_crop_dataset_loaded()
    normalized_key = normalize_crop_name(crop_key)
    item = CROP_DATASET.get(normalized_key)

    if not item:
        return None

    return item.get('category')


def get_available_crop_display_names():
    ensure_crop_dataset_loaded()

    names = []
    for item in CROP_DATASET.values():
        name = str(item.get('displayName') or '').strip()
        if name and name not in names:
            names.append(name)

    names.sort()
    return names


def score_value_against_range(value, min_value, max_value):
    span = abs(max_value - min_value)
    if span == 0:
        return 1.0 if value == min_value else 0.0

    center = (min_value + max_value) / 2
    half_span = span / 2

    if min_value <= value <= max_value:
        centeredness = 1 - min(abs(value - center) / half_span, 1.0)
        return 0.7 + (centeredness * 0.3)

    distance_outside = (min_value - value) if value < min_value else (value - max_value)
    penalty = min(distance_outside / (span * 2), 1.0)
    return 0.7 * (1 - penalty)


def crop_compatibility_score(crop_key, n, p, k, ph, moisture):
    ensure_crop_dataset_loaded()

    reqs = CROP_DATASET.get(crop_key, {}).get('requirements', DEFAULT_REQ)

    total = (
        score_value_against_range(n, reqs['n'][0], reqs['n'][1])
        + score_value_against_range(p, reqs['p'][0], reqs['p'][1])
        + score_value_against_range(k, reqs['k'][0], reqs['k'][1])
        + score_value_against_range(ph, reqs['ph'][0], reqs['ph'][1])
        + score_value_against_range(moisture, reqs['moisture'][0], reqs['moisture'][1])
    )

    return (total / 5) * 100


def get_model_ranked_crops(n, p, k, ph, moisture, limit=10):
    if model is None or encoder is None or not hasattr(model, 'predict_proba'):
        return []

    try:
        features = pd.DataFrame(
            [[n, p, k, ph, moisture]],
            columns=['N', 'P', 'K', 'pH', 'Moisture'],
        )

        probabilities = list(model.predict_proba(features.values)[0])
        labels = list(getattr(encoder, 'classes_', []))

        ranked_indices = sorted(
            range(len(probabilities)),
            key=lambda idx: probabilities[idx],
            reverse=True,
        )

        ranked_names = []
        for idx in ranked_indices:
            if idx >= len(labels):
                continue

            raw_label = str(labels[idx] or '').strip()
            if not raw_label:
                continue

            display_name = get_crop_display_name(raw_label)
            if display_name not in ranked_names:
                ranked_names.append(display_name)

            if len(ranked_names) == limit:
                break

        return ranked_names
    except Exception as e:
        print(f"Model probability ranking failed: {e}")
        return []


def get_top_crops(n, p, k, ph, moisture, category=None, preferred_crop_key=None, limit=3):
    ensure_crop_dataset_loaded()

    normalized_category = str(category or '').strip().lower()

    top = []
    normalized_preferred = normalize_crop_name(preferred_crop_key)

    if normalized_preferred:
        preferred_display_name = get_crop_display_name(normalized_preferred)
        preferred_category = normalize_crop_name(get_crop_category(normalized_preferred))
        if not normalized_category or preferred_category == normalized_category:
            top.append(preferred_display_name)

    # First priority: model probabilities (no manual crop metadata required)
    model_ranked = get_model_ranked_crops(
        n=n,
        p=p,
        k=k,
        ph=ph,
        moisture=moisture,
        limit=max(limit * 4, 12),
    )

    for display_name in model_ranked:
        if normalized_category:
            model_category = normalize_crop_name(get_crop_category(display_name))
            if model_category != normalized_category:
                continue

        if display_name not in top:
            top.append(display_name)

        if len(top) == limit:
            return top

    # Fallback: compatibility score against requirements metadata
    if normalized_category:
        pool = [
            key
            for key, value in CROP_DATASET.items()
            if str(value.get('category', '')).strip().lower() == normalized_category
        ]
    else:
        pool = list(CROP_DATASET.keys())

    scored = [
        (key, crop_compatibility_score(key, n=n, p=p, k=k, ph=ph, moisture=moisture))
        for key in pool
    ]
    scored.sort(key=lambda item: item[1], reverse=True)

    for crop_key, _score in scored:
        display_name = get_crop_display_name(crop_key)
        if display_name not in top:
            top.append(display_name)
        if len(top) == limit:
            break

    return top


def build_recommendation_summary(n, p, k, ph, moisture, predicted_crop_name):
    ensure_crop_dataset_loaded()

    predicted_key = normalize_crop_name(predicted_crop_name)

    recommended_crop = get_crop_display_name(predicted_key)
    category = get_crop_category(predicted_key) or 'General'

    category_filter = category if category and category != 'General' else None

    top_crops = get_top_crops(
        n=n,
        p=p,
        k=k,
        ph=ph,
        moisture=moisture,
        category=category_filter,
        preferred_crop_key=predicted_key,
        limit=3,
    )

    if not top_crops and recommended_crop and recommended_crop != 'Unknown':
        top_crops = [recommended_crop]

    return {
        'recommended_crop': recommended_crop,
        'recommended_category': category,
        'top_3_crops': top_crops,
    }


try:
    refresh_crop_dataset()
except Exception as e:
    print(f"Initial crop dataset refresh failed: {e}")


def resolve_user_id_from_token(client_token, payload):
    if db is None:
        return None

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

        if model is None or encoder is None:
            return jsonify({'error': 'Model server not ready'}), 503

        ensure_crop_dataset_loaded()

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
            sensor_data['recommendationCategory'] = None
            sensor_data['topCategoryCrops'] = []
            sensor_data['recommendationStatus'] = 'insufficient_data_zero_values'
        else:
            try:
                features = pd.DataFrame([[sensor_data['N'], sensor_data['P'], sensor_data['K'], sensor_data['pH'], sensor_data['moisture']]], columns=['N', 'P', 'K', 'pH', 'Moisture'])
                prediction_num = model.predict(features.values)[0]
                recommended_crop = encoder.inverse_transform([prediction_num])[0]

                summary = build_recommendation_summary(
                    n=sensor_data['N'],
                    p=sensor_data['P'],
                    k=sensor_data['K'],
                    ph=sensor_data['pH'],
                    moisture=sensor_data['moisture'],
                    predicted_crop_name=recommended_crop,
                )

                sensor_data['cropLabel'] = summary['recommended_crop']
                sensor_data['recommendationCategory'] = summary['recommended_category']
                sensor_data['topCategoryCrops'] = summary['top_3_crops']
                sensor_data['recommendationStatus'] = 'ok'
            except Exception as pred_e:
                print(f"Prediction failed during sensor update: {pred_e}")
                sensor_data['cropLabel'] = 'Unknown'
                sensor_data['recommendationCategory'] = None
                sensor_data['topCategoryCrops'] = []
                sensor_data['recommendationStatus'] = 'prediction_failed'

        db.collection('sensor_readings').add(sensor_data)
        
        return jsonify({
            'success': True, 
            'message': 'Data secured in Firestore',
            'userId': owner_uid,
            'recommended_crop': sensor_data.get('cropLabel'),
            'recommended_category': sensor_data.get('recommendationCategory'),
            'top_3_crops': sensor_data.get('topCategoryCrops', []),
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
        if model is None or encoder is None:
            return jsonify({'error': 'Model server not ready'}), 503

        ensure_crop_dataset_loaded()

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
                'recommended_category': None,
                'top_3_crops': [],
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

        summary = build_recommendation_summary(
            n=n,
            p=p,
            k=k,
            ph=ph,
            moisture=moisture,
            predicted_crop_name=recommended_crop,
        )
        
        return jsonify({
            'success': True,
            'recommended_crop': summary['recommended_crop'],
            'recommended_category': summary['recommended_category'],
            'top_3_crops': summary['top_3_crops'],
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
    ensure_crop_dataset_loaded()

    # Standardize crop name to lowercase
    crop = normalize_crop_name(crop_name)

    crop_item = CROP_DATASET.get(crop)
    reqs = crop_item['requirements'] if crop_item else DEFAULT_REQ
    category = crop_item['category'] if crop_item else 'General'
    display_name = crop_item['displayName'] if crop_item else get_crop_display_name(crop_name)

    return jsonify({
        'success': True,
        'crop': display_name,
        'category': category,
        'requirements': reqs,
        'available_crops': get_available_crop_display_names(),
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    
