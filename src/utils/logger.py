"""
Logger Module

Centralized logging configuration using Loguru.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("This is an info message")
    logger.error("This is an error message")
"""

import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Define log format
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Add console handler with colors
logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level="INFO",
    colorize=True,
)

# Create logs directory if it doesn't exist
log_dir = Path(__file__).resolve().parents[2] / "logs"
log_dir.mkdir(exist_ok=True)

# Add file handler with rotation
logger.add(
    log_dir / "app.log",
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="10 MB",      # Rotate when file reaches 10 MB
    retention="7 days",    # Keep logs for 7 days
    compression="zip",     # Compress rotated logs
)


def get_logger(name: str = __name__):
    """
    Get a logger instance with the specified name.

    Args:
        name: The name of the logger (usually __name__)

    Returns:
        A loguru logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data...")
        2024-01-15 10:30:00 | INFO     | module:function:10 | Processing data...
    """
    return logger.bind(name=name)


# Export the configured logger
__all__ = ["logger", "get_logger"]
