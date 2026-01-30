"""
Pytest Configuration and Fixtures

Shared fixtures for all tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n_samples, freq="D"),
            "product_id": np.random.randint(1, 10, n_samples),
            "category_id": np.random.randint(1, 5, n_samples),
            "store_id": np.random.randint(1, 5, n_samples),
            "historical_sales": np.random.uniform(50, 200, n_samples),
            "price": np.random.uniform(10, 100, n_samples),
            "promotion_flag": np.random.randint(0, 2, n_samples),
            "holiday_flag": np.random.randint(0, 2, n_samples),
            "economic_index": np.random.uniform(90, 110, n_samples),
            "target_demand": np.random.uniform(50, 300, n_samples),
        }
    )


@pytest.fixture
def sample_features(sample_raw_data):
    """Create sample feature data for testing."""
    df = sample_raw_data.copy()

    # Add basic features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Add lag features
    df["sales_lag_1"] = df["historical_sales"].shift(1).fillna(df["historical_sales"].mean())
    df["sales_lag_7"] = df["historical_sales"].shift(7).fillna(df["historical_sales"].mean())

    # Add rolling features
    df["sales_rolling_mean_7"] = df["historical_sales"].rolling(7, min_periods=1).mean()

    return df


@pytest.fixture
def train_test_data(sample_features):
    """Create train/test split."""
    df = sample_features.copy()

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    target_col = "target_demand"
    feature_cols = [c for c in df.columns if c not in ["date", target_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test outputs."""
    return tmp_path
