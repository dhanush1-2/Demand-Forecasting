# src/evaluate.py
import joblib
import pandas as pd
from pathlib import Path
from src.data_prep import load_data, resample_weekly
from src.features import create_lag_features, train_test_split_time
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
OUT_DIR = Path(__file__).resolve().parents[1] / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

def eval_prophet():
    df = load_data()
    weekly = resample_weekly(df)
    m = joblib.load(MODEL_DIR / 'prophet_model.joblib')
    future = weekly[['ds','promo']].copy()
    forecast = m.predict(future)
    preds = forecast[['ds','yhat']].set_index('ds')
    merged = weekly.set_index('ds').join(preds)
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    print("Prophet MAE:", mae)
    merged.reset_index().to_csv(OUT_DIR / 'prophet_forecast.csv', index=False)
    return merged

def eval_xgboost():
    df = load_data()
    weekly = resample_weekly(df)
    df_feat = create_lag_features(weekly)
    _, test = train_test_split_time(df_feat, test_weeks=12)
    obj = joblib.load(MODEL_DIR / 'xgb_model.joblib')
    model = obj['model']
    features = obj['features']
    import xgboost as xgb
    dtest = xgb.DMatrix(test[features])
    preds = model.predict(dtest)
    test = test.copy()
    test['y_pred'] = preds
    mae = mean_absolute_error(test['y'], preds)
    rmse = mean_squared_error(test['y'], preds, squared=False)
    print(f"XGBoost MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    test.to_csv(OUT_DIR / 'xgb_forecast.csv', index=False)
    return test

if __name__ == "__main__":
    eval_prophet()
    eval_xgboost()
    print("Outputs saved to", OUT_DIR)
