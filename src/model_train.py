import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import MODEL_DIR, FEATURE_COLUMNS, TARGET_COLUMN
from src.data_pipeline import split_data
from src.db import init_db, SessionLocal, TrainingLog


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(y_pred))
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)) if y_proba.sum() > 0 else 0.0,
    }


def save_model(model, name='random_forest.pkl'):
    ensure_dirs()
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(model, path)
    return path


from xgboost import XGBClassifier


def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, random_state=42):
    model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_pipeline(df, model_type='random_forest'):
    X_train, X_test, y_train, y_test = split_data(df)

    if model_type == 'xgboost':
        model = train_xgboost(X_train, y_train)
        model_name = 'xgboost_model.pkl'
    else:
        model = train_random_forest(X_train, y_train)
        model_name = 'random_forest.pkl'

    metrics = evaluate_model(model, X_test, y_test)
    path = save_model(model, model_name)

    init_db()
    db = SessionLocal()
    log = TrainingLog(model_type=model_type, metrics=metrics)
    db.add(log)
    db.commit()
    db.close()

    return {'model_path': path, 'metrics': metrics, 'model_type': model_type}


if __name__ == '__main__':
    from src.data_pipeline import preprocess, load_raw_data
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

    raw_file = os.path.join(RAW_DATA_DIR, 'ghana_outbreak_raw.csv')
    if not os.path.exists(raw_file):
        raise FileNotFoundError('Put CSV data in %s' % raw_file)

    df = preprocess(load_raw_data('ghana_outbreak_raw.csv'))
    result = train_pipeline(df)

    print('Training complete')
    print(result)
