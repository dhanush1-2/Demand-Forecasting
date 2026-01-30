"""
LightGBM Model Module

LightGBM (Light Gradient Boosting Machine) implementation.
Faster than XGBoost, especially on large datasets.
"""

from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.config import get_model_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LightGBMModel(BaseModel):
    """
    LightGBM model for demand forecasting.

    LightGBM advantages over XGBoost:
    - Faster training (leaf-wise growth vs level-wise)
    - Lower memory usage
    - Better handling of categorical features
    - Works well with large datasets

    Example:
        >>> model = LightGBMModel()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, **kwargs):
        """
        Initialize LightGBM model.

        Args:
            **kwargs: LightGBM parameters
        """
        super().__init__(name="lightgbm")

        # Get default config
        default_config = get_model_config("lightgbm")

        self.params = {
            "objective": kwargs.get("objective", default_config.get("objective", "regression")),
            "metric": kwargs.get("metric", default_config.get("metric", "rmse")),
            "max_depth": kwargs.get("max_depth", default_config.get("max_depth", 6)),
            "learning_rate": kwargs.get("learning_rate", default_config.get("learning_rate", 0.1)),
            "n_estimators": kwargs.get("n_estimators", default_config.get("n_estimators", 100)),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": kwargs.get("n_jobs", -1),
            "verbose": kwargs.get("verbose", -1),  # Suppress LightGBM output
        }

        self.early_stopping_rounds = kwargs.get(
            "early_stopping_rounds", default_config.get("early_stopping_rounds", 20)
        )

        logger.info(f"LightGBM params: {self.params}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "LightGBMModel":
        """
        Train the LightGBM model.
        """
        logger.info(f"Training LightGBM model on {len(X_train)} samples")

        self.feature_names = list(X_train.columns)

        # Create LightGBM model
        self.model = lgb.LGBMRegressor(**self.params)

        # Prepare callbacks for early stopping
        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(period=0),  # Suppress iteration logs
            ]
            logger.info(f"Using {len(X_val)} samples for validation")

        # Fit
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )

        self.is_fitted = True

        if hasattr(self.model, "best_iteration_"):
            logger.info(f"Best iteration: {self.model.best_iteration_}")

        logger.info("LightGBM training complete")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        predictions = self.model.predict(X)
        predictions = np.maximum(predictions, 0)  # Non-negative

        return predictions

    def get_params(self) -> dict:
        """Get model parameters."""
        return self.params.copy()

    def get_feature_importance(self) -> Optional[dict]:
        """Get feature importance scores."""
        if not self.is_fitted:
            return None

        importance = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importance))
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return importance_dict


if __name__ == "__main__":
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    from src.data.ingestion import load_raw_data
    from src.data.transformation import transform_data
    from src.features.engineering import create_features, get_feature_names
    from src.features.store import create_train_test_split

    # Prepare data
    df = load_raw_data()
    df = transform_data(df, add_features=False)
    df = create_features(df)
    train_df, test_df = create_train_test_split(df, test_size=0.2)

    feature_cols = get_feature_names(df)
    X_train, y_train = train_df[feature_cols], train_df["target_demand"]
    X_test, y_test = test_df[feature_cols], test_df["target_demand"]

    # Train
    model = LightGBMModel(n_estimators=50)
    model.fit(X_train, y_train, X_test, y_test)

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"\nTest MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
