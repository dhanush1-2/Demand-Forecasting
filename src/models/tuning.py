"""
Hyperparameter Tuning Module

Uses Optuna for automated hyperparameter optimization.
"""

import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from typing import Optional, Type
from sklearn.metrics import mean_squared_error

from src.models.base import BaseModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: Optional[int] = None
) -> dict:
    """
    Tune XGBoost hyperparameters with Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of trials
        timeout: Max time in seconds
        
    Returns:
        Dictionary with best params and results
    """
    logger.info(f"Starting XGBoost hyperparameter tuning ({n_trials} trials)")
    
    def objective(trial):
        # Define search space
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        
        # Train model
        model = XGBoostModel(**params)
        model.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate
        predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        
        return rmse
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42)
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    logger.info(f"Best trial RMSE: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return {
        "best_params": study.best_params,
        "best_rmse": study.best_trial.value,
        "n_trials": len(study.trials),
        "study": study
    }


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: Optional[int] = None
) -> dict:
    """
    Tune LightGBM hyperparameters with Optuna.
    """
    logger.info(f"Starting LightGBM hyperparameter tuning ({n_trials} trials)")
    
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        
        model = LightGBMModel(**params)
        model.fit(X_train, y_train, X_val, y_val)
        
        predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        
        return rmse
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    logger.info(f"Best trial RMSE: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return {
        "best_params": study.best_params,
        "best_rmse": study.best_trial.value,
        "n_trials": len(study.trials),
        "study": study
    }


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: Optional[int] = None
) -> dict:
    """
    Tune any supported model.
    
    Args:
        model_name: "xgboost" or "lightgbm"
        ... (same as above)
    """
    if model_name.lower() == "xgboost":
        return tune_xgboost(X_train, y_train, X_val, y_val, n_trials, timeout)
    elif model_name.lower() == "lightgbm":
        return tune_lightgbm(X_train, y_train, X_val, y_val, n_trials, timeout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
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
    
    # Quick tuning (10 trials for testing)
    print("\n" + "="*50)
    print("Tuning XGBoost...")
    print("="*50)
    xgb_results = tune_xgboost(X_train, y_train, X_test, y_test, n_trials=10)
    print(f"Best params: {xgb_results['best_params']}")
    print(f"Best RMSE: {xgb_results['best_rmse']:.4f}")
