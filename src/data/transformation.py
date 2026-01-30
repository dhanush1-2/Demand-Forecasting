"""
Data Transformation Module

Handles data cleaning and transformation:
- Remove duplicates
- Handle missing values
- Convert data types
- Sort data
- Basic feature creation
"""

from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.

    Keeps the first occurrence of each duplicate.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with duplicates removed
    """
    initial_rows = len(df)

    # Remove exact duplicates (all columns match)
    df = df.drop_duplicates()

    removed = initial_rows - len(df)
    if removed > 0:
        logger.warning(f"Removed {removed} duplicate rows")
    else:
        logger.info("No duplicate rows found")

    return df


def handle_missing_values(
    df: pd.DataFrame, strategy: str = "drop", fill_value: Optional[dict] = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.

    Args:
        df: Input DataFrame
        strategy: How to handle missing values
            - "drop": Remove rows with any missing values
            - "fill": Fill with specified values
            - "ffill": Forward fill (use previous value)
            - "bfill": Backward fill (use next value)
        fill_value: Dictionary of {column: value} for fill strategy

    Returns:
        DataFrame with missing values handled
    """
    missing_before = df.isnull().sum().sum()

    if missing_before == 0:
        logger.info("No missing values found")
        return df

    logger.info(f"Found {missing_before} missing values")

    if strategy == "drop":
        df = df.dropna()
        logger.info(f"Dropped rows with missing values. Remaining: {len(df)} rows")

    elif strategy == "fill":
        if fill_value is None:
            # Default: fill numeric with median, categorical with mode
            for col in df.columns:
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df = df.fillna(fill_value)
        logger.info("Filled missing values")

    elif strategy == "ffill":
        df = df.fillna(method="ffill")
        logger.info("Forward filled missing values")

    elif strategy == "bfill":
        df = df.fillna(method="bfill")
        logger.info("Backward filled missing values")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with correct types
    """
    logger.info("Converting data types...")

    # config = get_config()

    # Ensure date is datetime
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
        logger.debug("Converted 'date' to datetime")

    # Convert integer columns
    int_columns = [
        "product_id",
        "category_id",
        "store_id",
        "historical_sales",
        "promotion_flag",
        "holiday_flag",
        "target_demand",
    ]
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype("int64")

    # Convert float columns
    float_columns = ["price", "economic_index"]
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    logger.info("Data types converted successfully")
    return df


def sort_data(df: pd.DataFrame, by: list[str] = ["date"]) -> pd.DataFrame:
    """
    Sort DataFrame by specified columns.

    Args:
        df: Input DataFrame
        by: Columns to sort by (default: date)

    Returns:
        Sorted DataFrame
    """
    logger.info(f"Sorting data by: {by}")

    df = df.sort_values(by=by).reset_index(drop=True)

    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic derived features.

    These are simple features that don't require complex calculations.
    More features will be added in the features module.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with additional features
    """
    logger.info("Adding basic features...")

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Year, month, day
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["date"].dt.dayofweek

    # Is weekend (Saturday=5, Sunday=6)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Week of year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Quarter
    df["quarter"] = df["date"].dt.quarter

    logger.info(f"Added {7} basic features")

    return df


def transform_data(df: pd.DataFrame, add_features: bool = True) -> pd.DataFrame:
    """
    Main transformation function - runs all transformations.

    This is the main entry point for data transformation.

    Args:
        df: Raw DataFrame
        add_features: Whether to add basic features

    Returns:
        Transformed DataFrame
    """
    logger.info("Starting data transformation pipeline...")

    # Step 1: Remove duplicates
    df = remove_duplicates(df)

    # Step 2: Handle missing values
    df = handle_missing_values(df, strategy="drop")

    # Step 3: Convert data types
    df = convert_data_types(df)

    # Step 4: Sort by date
    df = sort_data(df, by=["date", "product_id", "store_id"])

    # Step 5: Add basic features
    if add_features:
        df = add_basic_features(df)

    logger.info(f"Transformation complete. Final shape: {df.shape}")

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from src.data.ingestion import load_raw_data, save_processed_data

    # Load raw data
    df = load_raw_data()
    print(f"Raw data shape: {df.shape}")

    # Transform
    df_transformed = transform_data(df)
    print(f"Transformed data shape: {df_transformed.shape}")
    print(f"\nNew columns: {list(df_transformed.columns)}")
    print(f"\nSample:\n{df_transformed.head()}")

    # Save
    save_processed_data(df_transformed)
