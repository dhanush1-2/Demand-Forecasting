"""
Tests for Data Transformation Module
"""

import numpy as np
import pandas as pd

from src.data.transformation import (
    convert_data_types,
    handle_missing_values,
    remove_duplicates,
)


class TestTransformation:
    """Test data transformation functions."""

    def test_remove_duplicates(self, sample_raw_data):
        """Test duplicate removal."""
        # Add duplicates
        df_with_dups = pd.concat([sample_raw_data, sample_raw_data.iloc[:5]])

        result = remove_duplicates(df_with_dups)

        assert len(result) == len(sample_raw_data)

    def test_handle_missing_values_numeric(self, sample_raw_data):
        """Test handling of missing numeric values."""
        df = sample_raw_data.copy()
        df.loc[0, "price"] = np.nan
        df.loc[1, "historical_sales"] = np.nan

        result = handle_missing_values(df)

        assert result["price"].notna().all()
        assert result["historical_sales"].notna().all()

    def test_convert_data_types(self, sample_raw_data):
        """Test data type conversion."""
        df = sample_raw_data.copy()
        df["date"] = df["date"].astype(str)  # Convert to string

        result = convert_data_types(df)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_transformation_preserves_rows(self, sample_raw_data):
        """Test that transformation doesn't lose rows unnecessarily."""
        result = remove_duplicates(sample_raw_data)

        assert len(result) == len(sample_raw_data)
