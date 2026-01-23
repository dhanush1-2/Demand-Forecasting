# src/train_prophet.py
from prophet import Prophet
import joblib
from pathlib import Path
from src.data_prep import load_data, resample_weekly

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models'
MODEL_PATH.mkdir(exist_ok=True)

def train_and_save(df=None):
    if df is None:
        df = load_data()
    df = resample_weekly(df)
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    m.add_regressor('promo')
    m.fit(df[['ds','y','promo']].rename(columns={'ds':'ds','y':'y'}))
    joblib.dump(m, MODEL_PATH / 'prophet_model.joblib')
    print("Saved Prophet model to", MODEL_PATH / 'prophet_model.joblib')
    return m

if __name__ == "__main__":
    train_and_save()
