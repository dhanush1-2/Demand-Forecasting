"""
API Dependencies

Dependency injection for FastAPI routes.
"""

from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

from src.utils.logger import get_logger
from src.utils.config import get_paths

logger = get_logger(__name__)


class ModelService:
    """
    Service class for handling model operations.
    """
    
    def __init__(self):
        self.model = None
        self.model_name = "unknown"
        self.model_type = "unknown"
        self.feature_columns = None
        self.trained_date = "unknown"
        self._load_model()
    
    def _load_model(self):
        """Load the best available model."""
        paths = get_paths()
        models_dir = paths["models"]
        
        lgb_path = models_dir / "lightgbm_model.pkl"
        xgb_path = models_dir / "xgboost_model.pkl"
        
        try:
            if lgb_path.exists():
                with open(lgb_path, "rb") as f:
                    model_data = pickle.load(f)
                self.model = model_data["model"]
                self.feature_columns = model_data.get("feature_columns", [])
                self.model_name = "lightgbm"
                self.model_type = "LightGBM Regressor"
                self.trained_date = datetime.fromtimestamp(
                    lgb_path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M")
                logger.info(f"LightGBM model loaded with {len(self.feature_columns)} features")
                
            elif xgb_path.exists():
                with open(xgb_path, "rb") as f:
                    model_data = pickle.load(f)
                self.model = model_data["model"]
                self.feature_columns = model_data.get("feature_columns", [])
                self.model_name = "xgboost"
                self.model_type = "XGBoost Regressor"
                self.trained_date = datetime.fromtimestamp(
                    xgb_path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M")
                logger.info(f"XGBoost model loaded with {len(self.feature_columns)} features")
                
            else:
                logger.warning("No trained model found")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def is_model_loaded(self) -> bool:
        return self.model is not None
    
    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "features_count": len(self.feature_columns) if self.feature_columns else 0,
            "trained_date": self.trained_date
        }
    
    def prepare_features(self, request) -> pd.DataFrame:
        """Convert prediction request to model features."""
        try:
            date_val = pd.to_datetime(request.prediction_date)
            
            # Start with base features
            data = {
                "product_id": request.product_id,
                "category_id": request.category_id,
                "store_id": request.store_id,
                "historical_sales": request.historical_sales,
                "price": request.price,
                "promotion_flag": request.promotion_flag,
                "holiday_flag": request.holiday_flag,
                "economic_index": request.economic_index,
            }
            
            # Time features
            data["day_of_week"] = date_val.dayofweek
            data["day_of_month"] = date_val.day
            data["month"] = date_val.month
            data["quarter"] = date_val.quarter
            data["year"] = date_val.year
            data["week_of_year"] = date_val.isocalendar()[1]
            data["is_weekend"] = 1 if date_val.dayofweek >= 5 else 0
            data["is_month_start"] = 1 if date_val.is_month_start else 0
            data["is_month_end"] = 1 if date_val.is_month_end else 0
            
            # Price features
            data["price_per_category"] = request.price / max(request.category_id, 1)
            data["price_times_promo"] = request.price * request.promotion_flag
            
            # Interaction features
            data["sales_price_ratio"] = request.historical_sales / (request.price + 1)
            data["promo_holiday"] = request.promotion_flag * request.holiday_flag
            data["store_category"] = request.store_id * 100 + request.category_id
            data["economic_sales"] = request.economic_index * request.historical_sales
            
            # Lag features (use historical_sales as proxy)
            for lag in [1, 3, 7, 14, 30]:
                data[f"sales_lag_{lag}"] = request.historical_sales
            
            # Rolling features
            for window in [7, 14, 30]:
                data[f"sales_rolling_mean_{window}"] = request.historical_sales
                data[f"sales_rolling_std_{window}"] = 0.0
                data[f"sales_rolling_min_{window}"] = request.historical_sales
                data[f"sales_rolling_max_{window}"] = request.historical_sales
            
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Match feature columns from training
            if self.feature_columns:
                # Add any missing columns with 0
                for col in self.feature_columns:
                    if col not in df.columns:
                        df[col] = 0
                
                # Select only training columns in correct order
                df = df[self.feature_columns]
            
            logger.info(f"Prepared {len(df.columns)} features for prediction")
            return df
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            raise ValueError(f"Failed to prepare features: {e}")
    
    def predict(self, features: pd.DataFrame) -> float:
        if self.model is None:
            raise ValueError("No model loaded")
        
        prediction = self.model.predict(features)[0]
        return float(max(0, prediction))
    
    def predict_batch(self, features_list: list[pd.DataFrame]) -> list[float]:
        if self.model is None:
            raise ValueError("No model loaded")
        
        combined = pd.concat(features_list, ignore_index=True)
        predictions = self.model.predict(combined)
        return [float(max(0, p)) for p in predictions]


@lru_cache()
def get_model_service() -> ModelService:
    return ModelService()
