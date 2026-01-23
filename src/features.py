# src/features.py
import pandas as pd
import numpy as np

def create_lag_features(df, lags=[1,2,3,4,8,12]):
    df = df.sort_values('ds').reset_index(drop=True)
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['rolling_4'] = df['y'].shift(1).rolling(4).mean()
    df['dow'] = df['ds'].dt.weekday
    df['month'] = df['ds'].dt.month
    df['is_promo'] = df['promo']
    df = df.dropna().reset_index(drop=True)
    return df

def train_test_split_time(df, test_weeks=12):
    df = df.sort_values('ds').reset_index(drop=True)
    train = df.iloc[:-test_weeks].copy()
    test = df.iloc[-test_weeks:].copy()
    return train, test
