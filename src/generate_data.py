# src/generate_data.py
import numpy as np
import pandas as pd
from pathlib import Path

def generate_daily_sales(start='2022-01-01', end='2024-12-31', seed=42):
    rng = pd.date_range(start=start, end=end, freq='D')
    np.random.seed(seed)
    # base demand trend
    days = (rng - rng[0]).days
    trend = 0.0005 * (days)  # small upward trend
    # weekly seasonality
    dow = rng.weekday
    weekly = 1 + 0.12 * np.sin(2 * np.pi * dow / 7)
    # yearly seasonality (approx)
    doy = rng.dayofyear
    yearly = 1 + 0.25 * np.sin(2 * np.pi * doy / 365.25)
    # promotions: occasional spikes
    promo = np.random.choice([0, 1], size=len(rng), p=[0.95, 0.05])
    promo_effect = 1 + promo * (0.6 + np.random.rand(len(rng)) * 0.8)
    # noise
    noise = np.random.normal(0, 0.08, size=len(rng))
    base = 50  # base units/day
    sales = base * (1 + trend) * weekly * yearly * promo_effect * (1 + noise)
    df = pd.DataFrame({'ds': rng, 'y': sales.round(2), 'promo': promo})
    df['store_id'] = 'store_1'
    return df

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / 'data'
    out.mkdir(exist_ok=True)
    df = generate_daily_sales()
    df.to_csv(out / 'sample_data.csv', index=False)
    print("Wrote", out / 'sample_data.csv')
