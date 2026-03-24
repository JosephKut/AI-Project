import json
from flask import Flask, request, jsonify
from src.model_train import train_pipeline
from src.predictor import load_best_model, predict
from src.data_pipeline import preprocess, load_raw_data, ingest_real_sources, load_combined_data
from src.db import init_db, PredictionHistory, SessionLocal

app = Flask(__name__)


@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'ok', 'service': 'outbreak-predictor'}), 200


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'No JSON payload provided'}), 400

    try:
        model = load_best_model()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500

    results = predict(model, data)
    db = SessionLocal()
    for r in results:
        row = PredictionHistory(
            district=r['input'].get('district', 'unknown'),
            features=r['input'],
            prediction=str(r['prediction']),
            confidence=r['confidence'],
        )
        db.add(row)
    db.commit()
    db.close()

    return jsonify({'predictions': results})


@app.route('/train', methods=['POST'])
def train_endpoint():
    payload = request.get_json(force=True) if request.data else {}
    source_file = payload.get('data_file')
    model_type = payload.get('model_type', 'random_forest').lower()
    ingest_real = bool(payload.get('ingest_real', False))

    if model_type not in ('random_forest', 'xgboost'):
        return jsonify({'error': 'model_type must be random_forest or xgboost'}), 400

    if ingest_real:
        try:
            sources = ingest_real_sources(force=payload.get('force_download', False))
        except Exception as e:
            return jsonify({'error': f'Real data ingestion failed: {e}'}), 500
        if not sources:
            return jsonify({'warning': 'No real sources found; check DATA_SOURCE_URLS in config'}), 400

    try:
        if source_file:
            df = preprocess(load_raw_data(source_file))
        else:
            df = preprocess(load_combined_data())
    except Exception as e:
        return jsonify({'error': f'Data load/preprocess failed: {e}'}), 500

    result = train_pipeline(df, model_type=model_type)
    return jsonify(result)


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
