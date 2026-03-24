import os
import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, PROCESSED_DATA_DIR, FEATURE_COLUMNS


def load_model(name='random_forest.pkl'):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError('Model artifact not found: %s' % path)
    return joblib.load(path)


def load_best_model():
    # choose the newest model available with priority xgboost > random_forest
    xgb_path = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
    rf_path = os.path.join(MODEL_DIR, 'random_forest.pkl')
    if os.path.exists(xgb_path):
        return load_model('xgboost_model.pkl')
    if os.path.exists(rf_path):
        return load_model('random_forest.pkl')
    raise FileNotFoundError('No trained model found; please run /train')


def load_encoder():
    path = os.path.join(PROCESSED_DATA_DIR, 'district_encoder.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError('District encoder not found: %s' % path)
    return joblib.load(path)


def predict(model, payload):
    encoder = load_encoder()
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        raise ValueError('payload must be dict or list of dicts')

    # Encode district
    if 'district' in df.columns:
        df['district_encoded'] = encoder.transform(df['district'])

    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    proba = model.predict_proba(df)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(df))
    preds = model.predict(df)

    return [{
        'input': dict(zip(df.columns, row)),
        'prediction': int(pred),
        'confidence': float(conf),
    } for row, pred, conf in zip(df.itertuples(index=False), preds, proba)]


def batch_predict(model, df):
    encoder = load_encoder()
    if 'district' in df.columns:
        df = df.copy()
        df['district_encoded'] = encoder.transform(df['district'])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return predict(model, df.to_dict(orient='records'))
