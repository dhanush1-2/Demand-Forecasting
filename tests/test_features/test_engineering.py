"""
Tests for Feature Engineering Module
"""

import pytest
import pandas as pd
import numpy as np
from src.features.engineering import (
    create_lag_features,
    create_rolling_features,
    create_time_features,
    create_features,
)


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_create_lag_features(self, sample_raw_data):
        """Test lag feature creation."""
        df = sample_raw_data.copy()
        
        result = create_lag_features(df, lags=[1, 7])
        
        # Actual column names from the function
        assert "lag_1" in result.columns
        assert "lag_7" in result.columns
    
    def test_create_rolling_features(self, sample_raw_data):
        """Test rolling feature creation."""
        df = sample_raw_data.copy()
        
        result = create_rolling_features(df, windows=[7], stats=["mean"])
        
        # Actual column name from the function
        assert "rolling_7_mean" in result.columns
    
    def test_create_time_features(self, sample_raw_data):
        """Test time feature creation."""
        df = sample_raw_data.copy()
        
        result = create_time_features(df)
        
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_weekend" in result.columns
    
    def test_create_features_increases_columns(self, sample_raw_data):
        """Test that feature creation adds new columns."""
        df = sample_raw_data.copy()
        original_cols = len(df.columns)
        
        result = create_features(df)
        
        assert len(result.columns) > original_cols
    
    def test_no_all_null_features(self, sample_raw_data):
        """Test that no feature is entirely null after creation."""
        df = sample_raw_data.copy()
        
        result = create_features(df)
        
        for col in result.columns:
            assert not result[col].isna().all(), f"Column {col} is all null"
    
    def test_weekend_flag_is_binary(self, sample_raw_data):
        """Test that weekend flag is 0 or 1."""
        df = sample_raw_data.copy()
        
        result = create_time_features(df)
        
        assert result["is_weekend"].isin([0, 1]).all()
