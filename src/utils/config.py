"""
Configuration Module

Loads and manages configuration from YAML files.

Usage:
    from src.utils.config import get_config

    config = get_config()
    print(config["data"]["raw_path"])
"""

from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global config cache
_config: dict | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the config file. If None, uses default path.

    Returns:
        Dictionary containing configuration values.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    global _config

    if config_path is None:
        # Default config path
        config_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    with open(config_path, "r") as f:
        _config = yaml.safe_load(f)

    logger.info("Configuration loaded successfully")
    return _config


def get_config() -> dict[str, Any]:
    """
    Get the current configuration.

    Loads from default path if not already loaded.

    Returns:
        Dictionary containing configuration values.
    """
    global _config

    if _config is None:
        _config = load_config()

    return _config


def get_data_config() -> dict[str, Any]:
    """Get data-related configuration."""
    return get_config()["data"]


def get_model_config(model_name: str | None = None) -> dict[str, Any]:
    """
    Get model configuration.

    Args:
        model_name: Specific model name (xgboost, lightgbm, etc.)
                   If None, returns all model configs.

    Returns:
        Model configuration dictionary.
    """
    config = get_config()["models"]

    if model_name:
        return config.get(model_name, {})

    return config


def get_mlflow_config() -> dict[str, Any]:
    """Get MLflow configuration."""
    return get_config()["mlflow"]


def get_api_config() -> dict[str, Any]:
    """Get API configuration."""
    return get_config()["api"]


def get_paths() -> dict[str, Path]:
    """
    Get all paths from configuration as Path objects.

    Returns:
        Dictionary with path names as keys and Path objects as values.
    """
    config = get_config()
    base_path = Path(__file__).resolve().parents[2]

    paths = {
        "base": base_path,
        "data": base_path / config["paths"]["data"],
        "raw_data": base_path / config["data"]["raw_path"],
        "processed_data": base_path / config["data"]["processed_path"],
        "features": base_path / config["data"]["features_path"],
        "models": base_path / config["paths"]["models"],
        "outputs": base_path / config["paths"]["outputs"],
        "logs": base_path / config["paths"]["logs"],
        "configs": base_path / config["paths"]["configs"],
    }

    return paths


# Export functions
__all__ = [
    "load_config",
    "get_config",
    "get_data_config",
    "get_model_config",
    "get_mlflow_config",
    "get_api_config",
    "get_paths",
]
