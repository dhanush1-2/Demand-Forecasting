# src/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.data_prep import load_data, resample_weekly
from src.features import create_lag_features

app = Flask(__name__)
MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'

# load models if available
prophet_model = None
xgb_obj = None
try:
    prophet_model = joblib.load(MODEL_DIR / 'prophet_model.joblib')
except Exception:
    prophet_model = None
try:
    xgb_obj = joblib.load(MODEL_DIR / 'xgb_model.joblib')
except Exception:
    xgb_obj = None

@app.route('/')
def index():
    return "Demand Forecast API - /predict?model=prophet|xgb&horizon_weeks=12"

@app.route('/predict')
def predict():
    model = request.args.get('model', 'prophet')
    horizon = int(request.args.get('horizon_weeks', 12))
    df = load_data()
    weekly = resample_weekly(df)
    if model == 'prophet' and prophet_model is not None:
        future = weekly[['ds','promo']].copy()
        forecast = prophet_model.predict(future)
        out = forecast[['ds','yhat']].tail(horizon).rename(columns={'yhat':'y_pred'}).to_dict(orient='records')
        return jsonify({'model':'prophet','predictions':out})
    elif model == 'xgb' and xgb_obj is not None:
        obj = xgb_obj
        model_bst = obj['model']
        features = obj['features']
        df_feat = create_lag_features(weekly)
        # we'll return last n rows predictions
        import xgboost as xgb
        dmat = xgb.DMatrix(df_feat[features])
        preds = model_bst.predict(dmat)
        df_feat['y_pred'] = preds
        out = df_feat[['ds','y_pred']].tail(horizon).to_dict(orient='records')
        return jsonify({'model':'xgb','predictions':out})
    else:
        return jsonify({'error':'model not found or not trained yet'}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
