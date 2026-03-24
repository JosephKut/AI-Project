# AI-Driven Disease Outbreak Prediction (Ghana)

## Overview
This project implements an outbreak prediction system using public health and environmental data in Ghana. It follows the architecture outlined in `AI.docx`:
- Data pipeline (ETL, cleaning, feature engineering)
- Model training (RandomForest/XGBoost + LSTM + hybrid ensemble)
- Flask API (`/predict`, `/train`, `/status`)
- Storage of outcomes for retraining and audits
- Dashboard + alerts (Streamlit)

## Structure
- `src/` : codebase
  - `data_pipeline.py` : data ingestion and preprocessing
  - `model_train.py` : model building, training, evaluation
  - `predictor.py` : model loading and inference
  - `api.py` : Flask endpoints for prediction and retraining
  - `config.py` : central config values
  - `db.py` : storage (SQLAlchemy wrapper)
- `data/` : example source data and unified dataset
- `models/` : serialized model artifacts (joblib/tf)
- `dashboard.py` : Streamlit web interface

## Quick start
1. Create venv:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Prepare data in `data/raw/` (CSV from Ghana Health Service etc.)
3. Run preprocessing:
   ```bash
   python src/data_pipeline.py
   ```
4. Train baseline models:
   ```bash
   python src/model_train.py
   ```
   - Or train with XGBoost via Python:
   ```python
   from src.data_pipeline import preprocess, load_raw_data, load_combined_data
   from src.model_train import train_pipeline
   
   df = preprocess(load_combined_data())
   result = train_pipeline(df, model_type='xgboost')
   print(result)
   ```

5. Real data ingestion
   - Configure real data sources in `src/config.py`:
     ```python
   DATA_SOURCE_URLS = [
       {'name': 'ghana_health', 'url': 'https://example.com/data/ghana_health.csv'},
       {'name': 'ghana_meteo', 'url': 'https://example.com/data/ghana_meteo.csv'},
   ]
     ```
   - Trigger ingestion and train via API:
     ```powershell
   $payload = @{ingest_real=$true; model_type='xgboost'} | ConvertTo-Json
   curl -X POST http://localhost:5000/train -H 'Content-Type: application/json' -d $payload
     ```
   - Or call endpoint in Python:
     ```python
   import requests
   r = requests.post('http://localhost:5000/train', json={'ingest_real': True, 'model_type': 'xgboost'})
   print(r.json())
     ```

6. Start API:
   ```bash
   python src/api.py
   ```
7. Launch dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## API Endpoints
- `GET /status` : Service health check
- `POST /predict` : Predict outbreak risk from input data
- `POST /train` : Retrain model with new data

## Dashboard Features
- Interactive input sliders for all factors
- Real-time risk prediction
- Visual charts of input factors
- Confidence scores

## Notes
- `model_train.py` includes RandomForest baseline; LSTM stub is included for extension.
- `data_pipeline.py` uses placeholder paths and feature-engineering examples.
- The architecture supports cron/Airflow for scheduled retraining.
