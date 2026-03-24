import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
REAL_DATA_DIR = os.path.join(DATA_DIR, 'real')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

DB_URI = os.getenv('DB_URI', 'sqlite:///' + os.path.join(BASE_DIR, 'data', 'outbreak.db'))

FEATURE_COLUMNS = [
    'district_encoded',  # encoded from district
    'week_of_year',
    'rainfall_mm',
    'temperature_c',
    'humidity_pct',
    'sanitation_score',
    'population_density',
    'previous_cases',
]
TARGET_COLUMN = 'outbreak_label'

# Optional: configure real source data URLs
DATA_SOURCE_URLS = [
  {'name': 'ghana_health', 'url': 'https://your.bucket/ghana_health.csv'},
  {'name': 'ghana_meteo', 'url': 'https://your.bucket/ghana_meteo.csv'},
]

