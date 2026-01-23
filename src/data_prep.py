# src/data_prep.py
import pandas as pd
from pathlib import Path

def load_data(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'data' / 'sample_data.csv'
    df = pd.read_csv(path, parse_dates=['ds'])
    return df

def resample_weekly(df):
    # Aggregate to weekly to reduce noise (example)
    df = df.set_index('ds').resample('W-MON').agg({'y':'sum', 'promo':'max'}).reset_index().rename(columns={'ds':'ds','y':'y','promo':'promo'})
    return df

if __name__ == "__main__":
    df = load_data()
    print("Loaded rows:", len(df))
    print(df.head())
