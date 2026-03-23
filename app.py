from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


try:
    model = joblib.load('models/cr_xgbrfclassifier_model.pkl')
    encoder = joblib.load('models/label_encoder.pkl') 
    print("Model and Encoder loaded successfully.")
except Exception as e:
    print(f" Error loading model: {e}")


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
        
        prediction_num = model.predict(features)[0]
        
        # Convert to original string name
        recommended_crop = encoder.inverse_transform([prediction_num])[0]
        
        return jsonify({
            'success': True,
            'recommended_crop': recommended_crop,
            'sensor_data_received': {'N': n, 'P': p, 'K': k, 'pH': ph, 'Moisture': moisture}
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)