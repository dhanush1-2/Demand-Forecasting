"""
Data Ingestion Module

Handles loading raw and processed data from files.
This is the entry point for all data in the pipeline.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import get_config, get_paths
from src.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw data from CSV file.

    Args:
        file_path: Path to CSV file. If None, uses path from config.

    Returns:
        pandas DataFrame with raw data

    Raises:
        FileNotFoundError: If the data file doesn't exist

    Example:
        >>> df = load_raw_data()
        >>> print(df.shape)
        (4921, 10)
    """
    # If no path provided, get from config
    if file_path is None:
        paths = get_paths()
        config = get_config()
        file_path = paths["raw_data"] / config["data"]["raw_file"]
    else:
        file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading raw data from: {file_path}")

    # Load CSV file
    # parse_dates converts the 'date' column to datetime objects
    df = pd.read_csv(file_path, parse_dates=["date"])

    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    logger.debug(f"Columns: {list(df.columns)}")

    return df


def load_processed_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load processed data from Parquet file.

    Parquet is a columnar format that's:
    - Faster to read/write than CSV
    - Smaller file size (compressed)
    - Preserves data types (no need to re-parse dates)

    Args:
        file_path: Path to Parquet file. If None, uses path from config.

    Returns:
        pandas DataFrame with processed data
    """
    if file_path is None:
        paths = get_paths()
        config = get_config()
        file_path = paths["processed_data"] / config["data"]["processed_file"]
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"Processed data not found: {file_path}")
        raise FileNotFoundError(f"Processed data not found: {file_path}")

    logger.info(f"Loading processed data from: {file_path}")

    # Read Parquet file
    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    return df


def save_processed_data(df: pd.DataFrame, file_path: Optional[str] = None) -> Path:
    """
    Save processed data to Parquet file.

    Args:
        df: DataFrame to save
        file_path: Where to save. If None, uses path from config.

    Returns:
        Path to saved file
    """
    if file_path is None:
        paths = get_paths()
        config = get_config()
        file_path = paths["processed_data"] / config["data"]["processed_file"]
    else:
        file_path = Path(file_path)

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed data to: {file_path}")

    # Save as Parquet with compression
    # 'snappy' compression is fast and efficient
    df.to_parquet(file_path, compression="snappy", index=False)

    logger.info(f"Saved {len(df)} rows to {file_path}")

    return file_path


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about a DataFrame.

    Useful for logging and debugging.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with data information
    """
    info = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
    }

    return info


# This block runs only when you execute this file directly
# python -m src.data.ingestion
if __name__ == "__main__":
    # Test the functions
    df = load_raw_data()
    print(f"Loaded data shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData info:\n{get_data_info(df)}")
