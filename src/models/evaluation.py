"""
Model Evaluation Module

Functions for evaluating and comparing models.
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Ensure arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    
    # MAPE - handle zeros
    # Avoid division by zero
    mask = y_true != 0
    if mask.any():
        metrics["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics["mape"] = np.nan
    
    # Additional metrics
    metrics["mean_error"] = np.mean(y_pred - y_true)  # Bias
    metrics["std_error"] = np.std(y_pred - y_true)    # Error spread
    
    return metrics


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*50}")
    print(f"Evaluation Metrics: {model_name}")
    print(f"{'='*50}")
    print(f"  MAE:        {metrics['mae']:.4f}")
    print(f"  RMSE:       {metrics['rmse']:.4f}")
    print(f"  R²:         {metrics['r2']:.4f}")
    print(f"  MAPE:       {metrics['mape']:.2f}%")
    print(f"  Mean Error: {metrics['mean_error']:.4f}")
    print(f"  Std Error:  {metrics['std_error']:.4f}")
    print(f"{'='*50}")


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        
    Returns:
        DataFrame comparing all models
    """
    comparison = pd.DataFrame(results).T
    comparison = comparison.round(4)
    
    # Sort by RMSE (lower is better)
    comparison = comparison.sort_values("rmse")
    
    # Add rank
    comparison["rank"] = range(1, len(comparison) + 1)
    
    return comparison


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    print_results: bool = True
) -> dict:
    """
    Evaluate a single model.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: Test target
        print_results: Whether to print results
        
    Returns:
        Dictionary of metrics
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions)
    
    if print_results:
        print_metrics(metrics, model.name)
    
    return metrics


def cross_validate_model(
    model_class,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    **model_kwargs
) -> dict:
    """
    Perform time-series cross-validation.
    
    Uses expanding window: train on past, predict future.
    
    Args:
        model_class: Model class to instantiate
        X: Features
        y: Target
        n_splits: Number of CV splits
        **model_kwargs: Arguments for model
        
    Returns:
        Dictionary with CV results
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    logger.info(f"Running {n_splits}-fold time series cross-validation")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        logger.info(f"Fold {fold}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_val, y_val, print_results=False)
        metrics["fold"] = fold
        fold_metrics.append(metrics)
    
    # Aggregate results
    metrics_df = pd.DataFrame(fold_metrics)
    
    cv_results = {
        "mean_mae": metrics_df["mae"].mean(),
        "std_mae": metrics_df["mae"].std(),
        "mean_rmse": metrics_df["rmse"].mean(),
        "std_rmse": metrics_df["rmse"].std(),
        "mean_r2": metrics_df["r2"].mean(),
        "std_r2": metrics_df["r2"].std(),
        "fold_metrics": fold_metrics,
    }
    
    logger.info(f"CV Results - MAE: {cv_results['mean_mae']:.4f} (+/- {cv_results['std_mae']:.4f})")
    logger.info(f"CV Results - RMSE: {cv_results['mean_rmse']:.4f} (+/- {cv_results['std_rmse']:.4f})")
    
    return cv_results


if __name__ == "__main__":
    # Test metrics
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 18, 33, 38, 52])
    
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Model")
