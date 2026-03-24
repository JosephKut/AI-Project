import os
import glob
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import RAW_DATA_DIR, REAL_DATA_DIR, PROCESSED_DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN, DATA_SOURCE_URLS


def ensure_dirs():
    for path in (RAW_DATA_DIR, REAL_DATA_DIR, PROCESSED_DATA_DIR):
        os.makedirs(path, exist_ok=True)


def download_data(url, destination_path):
    """Download data from HTTP(S) to a local file."""
    import requests

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    logging.info('Downloading %s -> %s', url, destination_path)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(destination_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return destination_path


def ingest_real_sources(force=False):
    """Download real-world sources listed in config.DATA_SOURCE_URLS."""
    ensure_dirs()
    downloaded = []

    if not DATA_SOURCE_URLS:
        logging.warning('DATA_SOURCE_URLS is empty; no real data sources configured.')
        return downloaded

    for source in DATA_SOURCE_URLS:
        url = source.get('url')
        if not url:
            logging.warning('Skipping invalid source config: %s', source)
            continue

        filename = os.path.basename(url.split('?')[0])
        dest = os.path.join(REAL_DATA_DIR, filename)

        if force or not os.path.exists(dest):
            try:
                download_data(url, dest)
            except Exception as exc:
                logging.error('Failed to download source %s: %s', url, exc)
                continue

        downloaded.append(dest)

    return downloaded


def load_raw_data(file_name=None):
    """Load a single raw CSV from raw directory."""
    ensure_dirs()

    if file_name:
        path = os.path.join(RAW_DATA_DIR, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Raw data file does not exist: {path}')
        return pd.read_csv(path)

    raise ValueError('file_name is required')


def load_combined_data():
    """Load all raw + real CSV files into a single dataframe."""
    ensure_dirs()
    files = glob.glob(os.path.join(RAW_DATA_DIR, '*.csv')) + glob.glob(os.path.join(REAL_DATA_DIR, '*.csv'))
    if not files:
        raise FileNotFoundError('No CSV files found in raw or real data directories')

    frames = []
    for f in files:
        logging.info('Loading CSV: %s', f)
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:
            logging.warning('Could not read %s: %s', f, exc)

    if not frames:
        raise ValueError('No readable CSV files found')

    df = pd.concat(frames, ignore_index=True)
    return df


def preprocess(df):
    ensure_dirs()
    df = df.copy()

    if 'district' in df.columns and 'district_encoded' not in df.columns:
        le = LabelEncoder()
        df['district_encoded'] = le.fit_transform(df['district'].astype(str))
        import joblib
        joblib.dump(le, os.path.join(PROCESSED_DATA_DIR, 'district_encoder.pkl'))

    for col in ['rainfall_mm', 'temperature_c', 'humidity_pct', 'sanitation_score', 'population_density', 'previous_cases']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

    if 'date' in df.columns:
        df['week_of_year'] = pd.to_datetime(df['date'], errors='coerce').dt.isocalendar().week

    if 'rainfall_mm' in df.columns:
        df['rainfall_rolling'] = df['rainfall_mm'].rolling(window=4, min_periods=1).mean()
        df['rainfall_anomaly'] = df['rainfall_mm'] - df['rainfall_rolling']

    if TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = (df.get('previous_cases', 0) > 50).astype(int)

    valid_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not valid_cols:
        raise ValueError('No valid feature columns in dataset')

    df = df.dropna(subset=valid_cols + [TARGET_COLUMN])
    return df


def split_data(df, test_size=0.2, random_state=42):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f'Missing target column: {TARGET_COLUMN}')

    features = [col for col in FEATURE_COLUMNS if col in df.columns]
    X = df[features]
    y = df[TARGET_COLUMN]

    if len(y.unique()) > 1:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == '__main__':
    ensure_dirs()
    try:
        df_raw = load_combined_data()
    except FileNotFoundError:
        print('No combined data file found; expect CSV in data/raw or data/real.')
        raise

    df = preprocess(df_raw)
    processed_path = os.path.join(PROCESSED_DATA_DIR, 'ghana_outbreak_processed.csv')
    df.to_csv(processed_path, index=False)
    print('Preprocessed data written to', processed_path)



def create_dirs():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


if __name__ == '__main__':
    create_dirs()
    raw_file = 'ghana_outbreak_raw.csv'
    if not os.path.exists(os.path.join(RAW_DATA_DIR, raw_file)):
        print('Place raw dataset in', os.path.join(RAW_DATA_DIR, raw_file))
    else:
        print('Loading raw data')
        df_raw = load_raw_data(raw_file)
        df_pre = preprocess(df_raw)
        processed_path = os.path.join(PROCESSED_DATA_DIR, 'ghana_outbreak_processed.csv')
        df_pre.to_csv(processed_path, index=False)
        print('Preprocessed data written to', processed_path)
