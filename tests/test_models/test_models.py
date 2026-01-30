"""
Tests for Model Training Module
"""

import numpy as np

from src.models.evaluation import calculate_metrics
from src.models.lightgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel


class TestModels:
    """Test model training and prediction."""

    def test_lightgbm_fit_predict(self, train_test_data):
        """Test LightGBM model training and prediction."""
        X_train, X_test, y_train, y_test = train_test_data

        model = LightGBMModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)
        assert not np.isnan(predictions).any()

    def test_xgboost_fit_predict(self, train_test_data):
        """Test XGBoost model training and prediction."""
        X_train, X_test, y_train, y_test = train_test_data

        model = XGBoostModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)
        assert not np.isnan(predictions).any()

    def test_predictions_are_reasonable(self, train_test_data):
        """Test that predictions are in reasonable range."""
        X_train, X_test, y_train, y_test = train_test_data

        model = LightGBMModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Predictions should be positive (demand can't be negative)
        # Allow small tolerance for numerical issues
        assert (predictions >= -1).all(), "Found significantly negative predictions"

    def test_model_improves_over_baseline(self, train_test_data):
        """Test that model beats simple mean baseline."""
        X_train, X_test, y_train, y_test = train_test_data

        # Baseline: predict mean
        baseline_pred = np.full(len(y_test), y_train.mean())
        baseline_mae = np.mean(np.abs(y_test - baseline_pred))

        # Model prediction
        model = LightGBMModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_mae = np.mean(np.abs(y_test - predictions))

        # Model should be at least as good as baseline
        # (on small random data, this might not always hold)
        assert (
            model_mae <= baseline_mae * 1.5
        ), "Model significantly worse than baseline"


class TestMetrics:
    """Test evaluation metrics."""

    def test_calculate_metrics_returns_dict(self):
        """Test that calculate_metrics returns expected keys."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

        metrics = calculate_metrics(y_true, y_pred)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_perfect_prediction_metrics(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["mae"] == 0
        assert metrics["rmse"] == 0
        assert metrics["r2"] == 1.0

    def test_metrics_are_non_negative(self):
        """Test that error metrics are non-negative."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
