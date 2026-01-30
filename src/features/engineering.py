"""
Feature Engineering Module

Creates features for demand forecasting models:
- Lag features (past values)
- Rolling statistics (moving averages, etc.)
- Time-based features (day, month, holidays)
- Price features
- Interaction features
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)


# =============================================================================
# LAG FEATURES
# =============================================================================

def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "target_demand",
    lags: Optional[list[int]] = None,
    group_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Create lag features (past values of target).
    
    Lag features help the model understand:
    "What was the demand X days ago?"
    
    Args:
        df: Input DataFrame (must be sorted by date)
        target_col: Column to create lags for
        lags: List of lag periods (e.g., [1, 7, 14] = 1 day ago, 7 days ago, 14 days ago)
        group_cols: Columns to group by before creating lags (e.g., ["product_id", "store_id"])
        
    Returns:
        DataFrame with lag features added
        
    Example:
        If today's demand is 100 and yesterday's was 80:
        - lag_1 = 80 (1 day ago)
        - lag_7 = demand from 7 days ago
    """
    if lags is None:
        config = get_config()
        lags = config["features"]["lags"]  # Default: [1, 3, 7, 14, 30]
    
    logger.info(f"Creating lag features for lags: {lags}")
    
    df = df.copy()
    
    for lag in lags:
        col_name = f"lag_{lag}"
        
        if group_cols:
            # Create lag within each group (product-store combination)
            df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
        else:
            # Simple shift without grouping
            df[col_name] = df[target_col].shift(lag)
        
        logger.debug(f"Created {col_name}")
    
    logger.info(f"Created {len(lags)} lag features")
    return df


# =============================================================================
# ROLLING FEATURES
# =============================================================================

def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "target_demand",
    windows: Optional[list[int]] = None,
    stats: Optional[list[str]] = None,
    group_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    Rolling features capture trends over time:
    - Rolling mean: Average demand over last X days
    - Rolling std: Volatility of demand
    - Rolling min/max: Range of demand
    
    Args:
        df: Input DataFrame (must be sorted by date)
        target_col: Column to calculate rolling stats for
        windows: Window sizes (e.g., [7, 14, 30] = 7-day, 14-day, 30-day windows)
        stats: Statistics to calculate (e.g., ["mean", "std", "min", "max"])
        group_cols: Columns to group by
        
    Returns:
        DataFrame with rolling features added
        
    Example:
        rolling_7_mean = average demand over last 7 days
        rolling_7_std = standard deviation over last 7 days
    """
    if windows is None:
        config = get_config()
        windows = config["features"]["rolling_windows"]  # Default: [7, 14, 30]
    
    if stats is None:
        config = get_config()
        stats = config["features"]["rolling_stats"]  # Default: ["mean", "std", "min", "max"]
    
    logger.info(f"Creating rolling features for windows: {windows}, stats: {stats}")
    
    df = df.copy()
    
    for window in windows:
        for stat in stats:
            col_name = f"rolling_{window}_{stat}"
            
            if group_cols:
                # Rolling within each group
                rolling = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).agg(stat)
                )
            else:
                # shift(1) to avoid data leakage (don't include current day)
                rolling = df[target_col].shift(1).rolling(window=window, min_periods=1).agg(stat)
            
            df[col_name] = rolling
            logger.debug(f"Created {col_name}")
    
    logger.info(f"Created {len(windows) * len(stats)} rolling features")
    return df


# =============================================================================
# TIME FEATURES
# =============================================================================

def create_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Create time-based features from date column.
    
    Time features help capture:
    - Weekly patterns (weekday vs weekend)
    - Monthly patterns (beginning vs end of month)
    - Seasonal patterns (summer vs winter)
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        
    Returns:
        DataFrame with time features added
    """
    logger.info("Creating time-based features")
    
    df = df.copy()
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Basic time features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["day_of_week"] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"] = df[date_col].dt.quarter
    
    # Binary features
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    df["is_quarter_start"] = df[date_col].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df[date_col].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding for day of week (captures that Sunday is close to Monday)
    # Uses sin/cos to represent cyclical nature
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    logger.info("Created 17 time-based features")
    return df


# =============================================================================
# PRICE FEATURES
# =============================================================================

def create_price_features(
    df: pd.DataFrame,
    price_col: str = "price",
    group_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Create price-related features.
    
    Price features help capture:
    - Price changes over time
    - Price relative to average
    - Price volatility
    
    Args:
        df: Input DataFrame
        price_col: Name of price column
        group_cols: Columns to group by (e.g., product_id)
        
    Returns:
        DataFrame with price features added
    """
    logger.info("Creating price features")
    
    df = df.copy()
    
    if group_cols:
        # Price statistics within each group
        df["price_mean"] = df.groupby(group_cols)[price_col].transform("mean")
        df["price_std"] = df.groupby(group_cols)[price_col].transform("std")
        df["price_min"] = df.groupby(group_cols)[price_col].transform("min")
        df["price_max"] = df.groupby(group_cols)[price_col].transform("max")
        
        # Price relative to group mean
        df["price_vs_mean"] = df[price_col] / df["price_mean"]
        
        # Price change from previous period
        df["price_change"] = df.groupby(group_cols)[price_col].diff()
        df["price_change_pct"] = df.groupby(group_cols)[price_col].pct_change()
    else:
        # Global statistics
        df["price_mean"] = df[price_col].mean()
        df["price_std"] = df[price_col].std()
        df["price_vs_mean"] = df[price_col] / df["price_mean"]
        df["price_change"] = df[price_col].diff()
        df["price_change_pct"] = df[price_col].pct_change()
    
    # Fill NaN in change columns with 0
    df["price_change"] = df["price_change"].fillna(0)
    df["price_change_pct"] = df["price_change_pct"].fillna(0)
    
    logger.info("Created 7 price features")
    return df


# =============================================================================
# INTERACTION FEATURES
# =============================================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between existing columns.
    
    Interaction features capture combined effects:
    - Promo during weekend might have different impact
    - Holiday + high price might reduce demand more
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with interaction features added
    """
    logger.info("Creating interaction features")
    
    df = df.copy()
    
    # Promotion interactions
    if "promotion_flag" in df.columns:
        if "is_weekend" in df.columns:
            df["promo_weekend"] = df["promotion_flag"] * df["is_weekend"]
        
        if "holiday_flag" in df.columns:
            df["promo_holiday"] = df["promotion_flag"] * df["holiday_flag"]
        
        if "price" in df.columns:
            # Discount effect: promo * (1 - normalized_price)
            price_norm = (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min())
            df["promo_discount_effect"] = df["promotion_flag"] * (1 - price_norm)
    
    # Economic interactions
    if "economic_index" in df.columns:
        if "price" in df.columns:
            # High price + low economy = low demand
            econ_norm = (df["economic_index"] - df["economic_index"].min()) / \
                       (df["economic_index"].max() - df["economic_index"].min())
            price_norm = (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min())
            df["price_economy_ratio"] = price_norm / (econ_norm + 0.01)  # +0.01 to avoid division by zero
    
    # Historical sales interactions
    if "historical_sales" in df.columns and "target_demand" in df.columns:
        # Demand trend: current vs historical
        df["demand_vs_historical"] = df["target_demand"] / (df["historical_sales"] + 1)  # +1 to avoid div by zero
    
    logger.info("Created interaction features")
    return df


# =============================================================================
# MAIN FEATURE CREATION FUNCTION
# =============================================================================

def create_features(
    df: pd.DataFrame,
    target_col: str = "target_demand",
    date_col: str = "date",
    group_cols: Optional[list[str]] = None,
    include_lags: bool = True,
    include_rolling: bool = True,
    include_time: bool = True,
    include_price: bool = True,
    include_interactions: bool = True,
) -> pd.DataFrame:
    """
    Main function to create all features.
    
    This is the entry point for feature engineering.
    
    Args:
        df: Input DataFrame (should be transformed/cleaned data)
        target_col: Target column name
        date_col: Date column name
        group_cols: Columns to group by for lag/rolling features
        include_lags: Whether to create lag features
        include_rolling: Whether to create rolling features
        include_time: Whether to create time features
        include_price: Whether to create price features
        include_interactions: Whether to create interaction features
        
    Returns:
        DataFrame with all features added
        
    Example:
        >>> df = load_processed_data()
        >>> df_features = create_features(df)
        >>> print(f"Features created: {len(df_features.columns)}")
    """
    logger.info("=" * 50)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 50)
    
    initial_cols = len(df.columns)
    df = df.copy()
    
    # Ensure data is sorted by date
    df = df.sort_values(by=[date_col]).reset_index(drop=True)
    
    # Create features in order
    if include_time:
        df = create_time_features(df, date_col)
    
    if include_lags:
        df = create_lag_features(df, target_col, group_cols=group_cols)
    
    if include_rolling:
        df = create_rolling_features(df, target_col, group_cols=group_cols)
    
    if include_price:
        df = create_price_features(df, group_cols=group_cols)
    
    if include_interactions:
        df = create_interaction_features(df)
    
    # Handle any remaining NaN values from lag/rolling features
    # Option 1: Drop rows with NaN (loses some data at the beginning)
    # Option 2: Fill with appropriate values
    
    # Count NaN before handling
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values after feature creation")
        
        # Fill NaN in lag features with 0 (or could use median)
        lag_cols = [col for col in df.columns if col.startswith("lag_")]
        for col in lag_cols:
            df[col] = df[col].fillna(0)
        
        # Fill NaN in rolling features with the column median
        rolling_cols = [col for col in df.columns if col.startswith("rolling_")]
        for col in rolling_cols:
            df[col] = df[col].fillna(df[col].median())
        
        logger.info("Filled NaN values in lag and rolling features")
    
    final_cols = len(df.columns)
    new_features = final_cols - initial_cols
    
    logger.info("=" * 50)
    logger.info(f"Feature engineering complete!")
    logger.info(f"Initial columns: {initial_cols}")
    logger.info(f"Final columns: {final_cols}")
    logger.info(f"New features created: {new_features}")
    logger.info("=" * 50)
    
    return df


def get_feature_names(df: pd.DataFrame, exclude_cols: Optional[list[str]] = None) -> list[str]:
    """
    Get list of feature column names (excluding target and identifiers).
    
    Args:
        df: DataFrame with features
        exclude_cols: Additional columns to exclude
        
    Returns:
        List of feature column names
    """
    # Default columns to exclude
    default_exclude = [
        "date", "target_demand", "product_id", "category_id", "store_id",
        "year"  # year often causes issues (data leakage if predicting future years)
    ]
    
    if exclude_cols:
        default_exclude.extend(exclude_cols)
    
    feature_cols = [col for col in df.columns if col not in default_exclude]
    
    return feature_cols


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from src.data.ingestion import load_raw_data
    from src.data.transformation import transform_data
    
    # Load and transform data
    df = load_raw_data()
    df = transform_data(df, add_features=False)  # Don't add basic features (we'll do it here)
    
    print(f"Data shape before features: {df.shape}")
    
    # Create features
    df_features = create_features(df)
    
    print(f"\nData shape after features: {df_features.shape}")
    print(f"\nAll columns ({len(df_features.columns)}):")
    print(list(df_features.columns))
    
    print(f"\nFeature columns only:")
    feature_cols = get_feature_names(df_features)
    print(f"Count: {len(feature_cols)}")
    print(feature_cols)
    
    print(f"\nSample data:")
    print(df_features.head())
