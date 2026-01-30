"""
Tests for Data Validation Module
"""

import pandas as pd


class TestDataValidation:
    """Test data validation functions."""

    def test_raw_data_has_required_columns(self, sample_raw_data):
        """Test that raw data has all required columns."""
        required_columns = [
            "date",
            "product_id",
            "category_id",
            "store_id",
            "historical_sales",
            "price",
            "promotion_flag",
            "holiday_flag",
            "economic_index",
            "target_demand",
        ]

        for col in required_columns:
            assert col in sample_raw_data.columns, f"Missing column: {col}"

    def test_no_negative_prices(self, sample_raw_data):
        """Test that prices are non-negative."""
        assert (sample_raw_data["price"] >= 0).all(), "Found negative prices"

    def test_binary_flags(self, sample_raw_data):
        """Test that flag columns are binary."""
        assert sample_raw_data["promotion_flag"].isin([0, 1]).all()
        assert sample_raw_data["holiday_flag"].isin([0, 1]).all()

    def test_date_column_is_datetime(self, sample_raw_data):
        """Test that date column is datetime type."""
        assert pd.api.types.is_datetime64_any_dtype(sample_raw_data["date"])

    def test_no_null_target(self, sample_raw_data):
        """Test that target column has no nulls."""
        assert sample_raw_data["target_demand"].notna().all()

    def test_positive_ids(self, sample_raw_data):
        """Test that ID columns are positive."""
        assert (sample_raw_data["product_id"] > 0).all()
        assert (sample_raw_data["category_id"] > 0).all()
        assert (sample_raw_data["store_id"] > 0).all()
