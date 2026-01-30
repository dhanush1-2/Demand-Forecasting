"""
Feature Store Module

Simple feature store for saving and loading features.

A feature store is a centralized place to:
- Save computed features
- Load features for training/inference
- Track feature versions
- Ensure consistency between training and serving
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_features(
    df: pd.DataFrame,
    name: str = "features",
    version: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save features to the feature store.

    Args:
        df: DataFrame with features
        name: Name for this feature set
        version: Version string (default: timestamp)
        metadata: Additional metadata to save

    Returns:
        Path to saved feature file

    Example:
        >>> df_features = create_features(df)
        >>> path = save_features(df_features, name="training_features", version="v1")
    """
    paths = get_paths()
    features_dir = paths["features"]
    features_dir.mkdir(parents=True, exist_ok=True)

    # Generate version if not provided
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # File paths
    feature_file = features_dir / f"{name}_{version}.parquet"
    metadata_file = features_dir / f"{name}_{version}_metadata.json"

    logger.info(f"Saving features to: {feature_file}")

    # Save features as Parquet
    df.to_parquet(feature_file, compression="snappy", index=False)

    # Prepare metadata
    feature_metadata = {
        "name": name,
        "version": version,
        "created_at": datetime.now().isoformat(),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "file_path": str(feature_file),
    }

    # Add custom metadata if provided
    if metadata:
        feature_metadata["custom"] = metadata

    # Save metadata
    with open(metadata_file, "w") as f:
        json.dump(feature_metadata, f, indent=2)

    logger.info(f"Saved {len(df)} rows with {len(df.columns)} columns")
    logger.info(f"Metadata saved to: {metadata_file}")

    # Also save as "latest" for easy access
    latest_file = features_dir / f"{name}_latest.parquet"
    df.to_parquet(latest_file, compression="snappy", index=False)
    logger.info(f"Also saved as latest: {latest_file}")

    return feature_file


def load_features(
    name: str = "features", version: Optional[str] = None
) -> pd.DataFrame:
    """
    Load features from the feature store.

    Args:
        name: Name of the feature set
        version: Version to load (default: "latest")

    Returns:
        DataFrame with features

    Example:
        >>> df = load_features(name="training_features")
        >>> # Or specific version:
        >>> df = load_features(name="training_features", version="20240115_103000")
    """
    paths = get_paths()
    features_dir = paths["features"]

    if version is None or version == "latest":
        feature_file = features_dir / f"{name}_latest.parquet"
    else:
        feature_file = features_dir / f"{name}_{version}.parquet"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    logger.info(f"Loading features from: {feature_file}")

    df = pd.read_parquet(feature_file)

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    return df


def get_feature_metadata(name: str = "features", version: Optional[str] = None) -> dict:
    """
    Get metadata for a feature set.

    Args:
        name: Name of the feature set
        version: Version (searches for latest if not provided)

    Returns:
        Metadata dictionary
    """
    paths = get_paths()
    features_dir = paths["features"]

    # Find metadata file
    if version:
        metadata_file = features_dir / f"{name}_{version}_metadata.json"
    else:
        # Find the most recent metadata file
        metadata_files = list(features_dir.glob(f"{name}_*_metadata.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata found for feature set: {name}")
        metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return metadata


def list_feature_versions(name: str = "features") -> list[dict]:
    """
    List all versions of a feature set.

    Args:
        name: Name of the feature set

    Returns:
        List of version info dictionaries
    """
    paths = get_paths()
    features_dir = paths["features"]

    # Find all metadata files for this feature set
    metadata_files = list(features_dir.glob(f"{name}_*_metadata.json"))

    versions = []
    for mf in metadata_files:
        with open(mf, "r") as f:
            metadata = json.load(f)
            versions.append(
                {
                    "version": metadata.get("version"),
                    "created_at": metadata.get("created_at"),
                    "num_rows": metadata.get("num_rows"),
                    "num_columns": metadata.get("num_columns"),
                }
            )

    # Sort by creation time
    versions.sort(key=lambda x: x["created_at"], reverse=True)

    return versions


def delete_old_versions(name: str = "features", keep_last: int = 3) -> int:
    """
    Delete old versions of a feature set, keeping the most recent ones.

    Args:
        name: Name of the feature set
        keep_last: Number of recent versions to keep

    Returns:
        Number of versions deleted
    """
    paths = get_paths()
    features_dir = paths["features"]

    # Find all files for this feature set
    parquet_files = list(features_dir.glob(f"{name}_*.parquet"))
    metadata_files = list(features_dir.glob(f"{name}_*_metadata.json"))

    # Exclude "latest" files
    parquet_files = [f for f in parquet_files if "latest" not in f.name]
    metadata_files = [f for f in metadata_files if "latest" not in f.name]

    # Sort by modification time (newest first)
    parquet_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Delete old files
    deleted = 0
    for f in parquet_files[keep_last:]:
        f.unlink()
        deleted += 1
        logger.info(f"Deleted: {f.name}")

    for f in metadata_files[keep_last:]:
        f.unlink()
        logger.info(f"Deleted: {f.name}")

    logger.info(f"Deleted {deleted} old feature versions")
    return deleted


# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_col: str = "date",
    shuffle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split features into train and test sets.

    For time series, we use temporal split (not random):
    - Train: Earlier data
    - Test: Later data

    Args:
        df: DataFrame with features
        test_size: Proportion of data for testing (0.2 = 20%)
        date_col: Date column for temporal split
        shuffle: If True, random split. If False, temporal split (recommended)

    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Creating train/test split (test_size={test_size})")

    if shuffle:
        # Random split (not recommended for time series)
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        logger.warning("Using random split - not recommended for time series!")
    else:
        # Temporal split (recommended)
        df = df.sort_values(by=date_col).reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Temporal split at index {split_idx}")
        logger.info(
            f"Train date range: {train_df[date_col].min()} to {train_df[date_col].max()}"
        )
        logger.info(
            f"Test date range: {test_df[date_col].min()} to {test_df[date_col].max()}"
        )

    logger.info(f"Train size: {len(train_df)} rows")
    logger.info(f"Test size: {len(test_df)} rows")

    return train_df, test_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from src.data.ingestion import load_raw_data
    from src.data.transformation import transform_data
    from src.features.engineering import create_features

    # Load and transform data
    df = load_raw_data()
    df = transform_data(df, add_features=False)

    # Create features
    df_features = create_features(df)

    # Save to feature store
    feature_path = save_features(
        df_features,
        name="demand_features",
        metadata={"description": "Full feature set for demand forecasting"},
    )

    print(f"\nSaved features to: {feature_path}")

    # Load back
    df_loaded = load_features(name="demand_features")
    print(f"Loaded features shape: {df_loaded.shape}")

    # List versions
    versions = list_feature_versions(name="demand_features")
    print("\nFeature versions:")
    for v in versions:
        print(f"  - {v['version']}: {v['num_rows']} rows, {v['num_columns']} cols")

    # Create train/test split
    train_df, test_df = create_train_test_split(df_features, test_size=0.2)
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
