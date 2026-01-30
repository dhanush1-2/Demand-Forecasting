"""
XGBoost Model Module

XGBoost (eXtreme Gradient Boosting) implementation for demand forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional
import xgboost as xgb

from src.models.base import BaseModel
from src.utils.logger import get_logger
from src.utils.config import get_model_config

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost model for demand forecasting.
    
    XGBoost is a gradient boosting algorithm that:
    - Builds trees sequentially, each fixing errors of previous
    - Handles missing values automatically
    - Has built-in regularization to prevent overfitting
    - Very fast and efficient
    
    Example:
        >>> model = XGBoostModel()
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            **kwargs: XGBoost parameters (overrides defaults from config)
        """
        super().__init__(name="xgboost")
        
        # Get default config
        default_config = get_model_config("xgboost")
        
        # Merge with provided kwargs (kwargs override defaults)
        self.params = {
            "objective": kwargs.get("objective", default_config.get("objective", "reg:squarederror")),
            "eval_metric": kwargs.get("eval_metric", default_config.get("eval_metric", "rmse")),
            "max_depth": kwargs.get("max_depth", default_config.get("max_depth", 6)),
            "learning_rate": kwargs.get("learning_rate", default_config.get("learning_rate", 0.1)),
            "n_estimators": kwargs.get("n_estimators", default_config.get("n_estimators", 100)),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": kwargs.get("n_jobs", -1),  # Use all CPU cores
        }
        
        # Early stopping
        self.early_stopping_rounds = kwargs.get(
            "early_stopping_rounds",
            default_config.get("early_stopping_rounds", 20)
        )
        
        logger.info(f"XGBoost params: {self.params}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> "XGBoostModel":
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target
            **kwargs: Additional parameters
            
        Returns:
            self
        """
        logger.info(f"Training XGBoost model on {len(X_train)} samples")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Create XGBoost model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Prepare eval set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            logger.info(f"Using {len(X_val)} samples for validation")
        
        # Fit the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=kwargs.get("verbose", False)
        )
        
        self.is_fitted = True
        
        # Log best iteration if early stopping was used
        if hasattr(self.model, "best_iteration"):
            logger.info(f"Best iteration: {self.model.best_iteration}")
        
        logger.info("XGBoost training complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions (demand can't be negative)
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return self.params.copy()
    
    def get_feature_importance(self) -> Optional[dict]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of {feature_name: importance_score}
        """
        if not self.is_fitted:
            return None
        
        importance = self.model.feature_importances_
        
        # Create dict with feature names
        importance_dict = dict(zip(self.feature_names, importance))
        
        # Sort by importance (descending)
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from src.data.ingestion import load_raw_data
    from src.data.transformation import transform_data
    from src.features.engineering import create_features, get_feature_names
    from src.features.store import create_train_test_split
    
    # Load and prepare data
    df = load_raw_data()
    df = transform_data(df, add_features=False)
    df = create_features(df)
    
    # Split data
    train_df, test_df = create_train_test_split(df, test_size=0.2)
    
    # Get feature columns
    feature_cols = get_feature_names(df)
    target_col = "target_demand"
    
    # Prepare X and y
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Train model
    model = XGBoostModel(n_estimators=50)  # Small for testing
    model.fit(X_train, y_train, X_test, y_test, verbose=True)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Evaluate
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"\nTest MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    # Feature importance
    print("\nTop 10 Features:")
    importance = model.get_feature_importance()
    for i, (feat, imp) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {feat}: {imp:.4f}")
