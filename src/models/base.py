"""
Base Model Module

Abstract base class for all ML models.
Ensures all models have the same interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.

    All models must implement:
    - fit(): Train the model
    - predict(): Make predictions
    - get_params(): Get model parameters

    This ensures consistency across different model types.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the base model.

        Args:
            name: Model name (e.g., "xgboost", "lightgbm")
            **kwargs: Model-specific parameters
        """
        self.name = name
        self.model = None  # The actual model object
        self.is_fitted = False  # Whether model has been trained
        self.feature_names = None  # List of feature names used
        self.params = kwargs  # Model parameters

        logger.info(f"Initialized {name} model")

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "BaseModel":
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, for early stopping)
            y_val: Validation target (optional)
            **kwargs: Additional training parameters

        Returns:
            self (for method chaining)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        pass

    def get_feature_importance(self) -> Optional[dict]:
        """
        Get feature importance scores.

        Returns:
            Dictionary of {feature_name: importance_score}
            or None if not available
        """
        return None

    def save(self, path: str | Path) -> Path:
        """
        Save model to disk.

        Args:
            path: Where to save the model

        Returns:
            Path to saved model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        model_data = {
            "model": self.model,
            "name": self.name,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "params": self.params,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to: {path}")

        return path

    def load(self, path: str | Path) -> "BaseModel":
        """
        Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            self (for method chaining)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.name = model_data["name"]
        self.is_fitted = model_data["is_fitted"]
        self.feature_names = model_data["feature_names"]
        self.params = model_data["params"]

        logger.info(f"Model loaded from: {path}")

        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, is_fitted={self.is_fitted})"
