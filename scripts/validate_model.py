"""
Validate a new model + encoder or pipeline placed in the `models/` folder.
Usage:
  python scripts/validate_model.py

It looks for, in priority order:
  1) models/pipeline.pkl        (preferred: scikit-learn Pipeline with transformer + estimator)
  2) models/pipeline.joblib
  3) models/pipeline.sav
  4) models/crop_recommendation_xgbrf_model.json (XGBoost JSON) + models/label_encoder.pkl
  5) models/*.pkl model + models/label_encoder.pkl

The script runs 50 randomized samples and prints prediction distribution and example probabilities.
"""
import os
import sys
import random
from collections import Counter

import pandas as pd
import joblib


def encoder_has_crop_labels(loaded_encoder):
    classes = list(getattr(loaded_encoder, 'classes_', []))
    if not classes:
        return False

    for item in classes:
        label = str(item or '').strip()
        if any(ch.isalpha() for ch in label):
            return True

    return False

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODELS_DIR = os.path.abspath(MODELS_DIR)

print('Models dir:', MODELS_DIR)

pipeline_path = None
for fname in ('pipeline.pkl', 'pipeline.joblib', 'pipeline.sav'):
    p = os.path.join(MODELS_DIR, fname)
    if os.path.exists(p):
        pipeline_path = p
        break

if pipeline_path:
    print('Found pipeline:', pipeline_path)
    pipe = joblib.load(pipeline_path)
    def predict(X):
        return pipe.predict(X)
    def predict_proba(X):
        try:
            return pipe.predict_proba(X)
        except Exception:
            return None
    label_from_pred = lambda x: x
else:
    # Try model + encoder
    from importlib import import_module
    try:
        import xgboost as xgb
    except Exception:
        print('xgboost not installed in this environment. Install requirements from requirements.txt')
        sys.exit(1)

    # try JSON model
    json_model = os.path.join(MODELS_DIR, 'crop_recommendation_xgbrf_model.json')
    pkl_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl') and 'label_encoder' not in f]

    model = None
    if os.path.exists(json_model):
        print('Found XGBoost JSON model:', json_model)
        model = xgb.XGBRFClassifier()
        model.load_model(json_model)
    elif pkl_models:
        chosen = pkl_models[0]
        print('Found pkl model:', chosen)
        model = joblib.load(os.path.join(MODELS_DIR, chosen))

    if model is None:
        print('No model found. Place a pipeline or model+label_encoder in models/')
        sys.exit(1)

    # load encoder
    enc_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    if not os.path.exists(enc_path):
        print('label_encoder.pkl not found alongside model. Place label_encoder.pkl in models/')
        sys.exit(1)

    enc = joblib.load(enc_path)
    classes = list(getattr(enc, 'classes_', []))
    if not encoder_has_crop_labels(enc):
        print('Invalid label_encoder.pkl: classes appear numeric-only (e.g. 0..N).')
        print('Fix: fit encoder on crop name strings and re-save label_encoder.pkl.')
        print('Current classes sample:', classes[:10])
        sys.exit(1)

    def predict(X):
        pred_nums = model.predict(X)
        try:
            return enc.inverse_transform(pred_nums)
        except Exception:
            # fallback: map numeric indices
            return [classes[int(p)] if int(p) < len(classes) else str(p) for p in pred_nums]

    def predict_proba(X):
        try:
            return model.predict_proba(X)
        except Exception:
            return None


# Run randomized tests
cnt = Counter()
probs_examples = []
for i in range(50):
    n = random.randint(20,120)
    p = random.randint(20,100)
    k = random.randint(20,100)
    ph = round(random.uniform(5.5,7.5), 1)
    moisture = round(random.uniform(30.0,90.0), 1)
    df = pd.DataFrame([[n,p,k,ph,moisture]], columns=['N','P','K','pH','Moisture'])
    try:
        pred = predict(df.values if 'model' in globals() else df)
        lab = pred[0] if hasattr(pred, '__iter__') else pred
    except Exception as e:
        lab = f'ERR:{e}'
    cnt[lab]+=1

    proba = None
    try:
        pvec = predict_proba(df.values if 'model' in globals() else df)
        if pvec is not None:
            top_idx = int(pvec[0].argmax())
            top_prob = float(pvec[0][top_idx])
            if 'classes' in globals():
                top_label = classes[top_idx]
            else:
                top_label = lab
            proba = (top_label, top_prob)
    except Exception:
        proba = None

    probs_examples.append((lab, proba, (n,p,k,ph,moisture)))

print('\nPrediction distribution (top results):')
for lab, c in cnt.most_common(10):
    print(f'  {lab}: {c}')

print('\nExample outputs:')
for ex in probs_examples[:10]:
    print(' ', ex)

print('\nDone.')
