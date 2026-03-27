import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd


cred = credentials.Certificate("C:\Users\aaron\Downloads\Firebase Credentials\qacg-crop-recommendation-firebase-adminsdk-fbsvc-c573940045.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

print("Downloading data from Firestore...")

docs = db.collection('sensor_readings').order_by('timestamp').stream()

dataset = []
for doc in docs:
    data = doc.to_dict()

    if 'timestamp' in data and data['timestamp'] is not None:
        data['timestamp'] = data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    dataset.append(data)

if len(dataset) > 0:
    df = pd.DataFrame(dataset)

    df = df[['timestamp', 'N', 'P', 'K', 'pH', 'Moisture']]
    df.to_csv('model_v1_training_data.csv', index=False)
    print(f"Success! Downloaded {len(dataset)} rows to model_v1_training_data.csv")
else:
    print("No data found in the database.")