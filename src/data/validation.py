"""
Data Validation Module

Uses Pandera to define and validate data schemas.
This ensures data quality before processing.

Pandera lets you:
- Define expected column types
- Set value constraints (min, max, allowed values)
- Check for missing values
- Run custom validation functions
"""

from typing import Optional

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# Define the schema for raw data
# This describes what the data SHOULD look like
raw_data_schema = DataFrameSchema(
    columns={
        # Date column: must be datetime, no nulls allowed
        "date": Column(
            dtype="datetime64[ns]",  # Expected data type
            nullable=False,  # No missing values allowed
            description="Transaction date",
        ),
        # Product ID: integer between 1000-1100
        "product_id": Column(
            dtype="int64",
            nullable=False,
            checks=[
                Check.ge(1000),  # Greater than or equal to 1000
                Check.le(1100),  # Less than or equal to 1100
            ],
            description="Product identifier",
        ),
        # Category ID: integer 1-5
        "category_id": Column(
            dtype="int64",
            nullable=False,
            checks=[
                Check.isin([1, 2, 3, 4, 5]),  # Must be one of these values
            ],
            description="Product category",
        ),
        # Store ID: integer 1-10
        "store_id": Column(
            dtype="int64",
            nullable=False,
            checks=[
                Check.ge(1),
                Check.le(10),
            ],
            description="Store identifier",
        ),
        # Historical sales: non-negative integer
        "historical_sales": Column(
            dtype="int64",
            nullable=False,
            checks=[
                Check.ge(0),  # Sales can't be negative
            ],
            description="Historical sales volume",
        ),
        # Price: positive float
        "price": Column(
            dtype="float64",
            nullable=False,
            checks=[
                Check.gt(0),  # Price must be > 0
                Check.le(100),  # Price <= 100 (based on data)
            ],
            description="Product price",
        ),
        # Promotion flag: binary (0 or 1)
        "promotion_flag": Column(
            dtype="int64",
            nullable=False,
            checks=[
                Check.isin([0, 1]),  # Only 0 or 1 allowed
            ],
            description="Whether promotion is active",
        ),
        # Holiday flag: binary (0 or 1)
        "holiday_flag": Column(
            dtype="int64",
            nullable=False,
            checks=[
                Check.isin([0, 1]),
            ],
            description="Whether it's a holiday",
        ),
        # Economic index: positive float
        "economic_index": Column(
            dtype="float64",
            nullable=False,
            checks=[
                Check.gt(0),  # Must be positive
                Check.le(200),  # Reasonable upper bound
            ],
            description="Economic indicator",
        ),
        # Target demand: non-negative integer (what we're predicting)
        "target_demand": Column(
            dtype="int64",
            nullable=False,
            checks=[
                Check.ge(0),  # Demand can't be negative
            ],
            description="Target variable - demand to predict",
        ),
    },
    # DataFrame-level checks
    checks=[
        # Ensure no duplicate rows (same date + product + store)
        Check(
            lambda df: ~df.duplicated(subset=["date", "product_id", "store_id"]).any(),
            error="Duplicate rows found (same date, product, store)",
        ),
    ],
    # Other settings
    strict=True,  # Fail if unexpected columns exist
    coerce=True,  # Try to convert types if needed
)


# Schema for processed data (after transformation)
processed_data_schema = DataFrameSchema(
    columns={
        "date": Column("datetime64[ns]", nullable=False),
        "product_id": Column("int64", nullable=False),
        "category_id": Column("int64", nullable=False),
        "store_id": Column("int64", nullable=False),
        "historical_sales": Column("int64", nullable=False),
        "price": Column("float64", nullable=False),
        "promotion_flag": Column("int64", nullable=False),
        "holiday_flag": Column("int64", nullable=False),
        "economic_index": Column("float64", nullable=False),
        "target_demand": Column("int64", nullable=False),
    },
    strict=False,  # Allow additional columns (we'll add features later)
    coerce=True,
)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_raw_data(
    df: pd.DataFrame, raise_error: bool = True
) -> tuple[bool, Optional[str]]:
    """
    Validate raw data against the schema.

    Args:
        df: DataFrame to validate
        raise_error: If True, raises exception on failure. If False, returns error message.

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> df = load_raw_data()
        >>> is_valid, error = validate_raw_data(df, raise_error=False)
        >>> if not is_valid:
        ...     print(f"Validation failed: {error}")
    """
    logger.info("Validating raw data against schema...")

    try:
        # Validate and potentially coerce types
        validated_df = raw_data_schema.validate(df)
        logger.info("✓ Raw data validation passed!")
        return True, None

    except pa.errors.SchemaError as e:
        error_msg = str(e)
        logger.error(f"✗ Raw data validation failed: {error_msg}")

        if raise_error:
            raise

        return False, error_msg


def validate_processed_data(
    df: pd.DataFrame, raise_error: bool = True
) -> tuple[bool, Optional[str]]:
    """
    Validate processed data against the schema.

    Args:
        df: DataFrame to validate
        raise_error: If True, raises exception on failure.

    Returns:
        Tuple of (is_valid, error_message)
    """
    logger.info("Validating processed data against schema...")

    try:
        validated_df = processed_data_schema.validate(df)
        logger.info("✓ Processed data validation passed!")
        return True, None

    except pa.errors.SchemaError as e:
        error_msg = str(e)
        logger.error(f"✗ Processed data validation failed: {error_msg}")

        if raise_error:
            raise

        return False, error_msg


def get_validation_report(df: pd.DataFrame) -> dict:
    """
    Generate a detailed validation report.

    This doesn't raise errors - it just reports what it finds.
    Useful for EDA and debugging.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with validation results
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": {},
    }

    for col in df.columns:
        col_report = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": round(df[col].isnull().sum() / len(df) * 100, 2),
            "unique_count": int(df[col].nunique()),
        }

        # Add numeric statistics if applicable
        if pd.api.types.is_numeric_dtype(df[col]):
            col_report.update(
                {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                }
            )

        report["columns"][col] = col_report

    return report


def validate_data(df: pd.DataFrame, schema_type: str = "raw") -> pd.DataFrame:
    """
    Validate data and return validated DataFrame.

    This is a convenience function that combines validation and type coercion.

    Args:
        df: DataFrame to validate
        schema_type: "raw" or "processed"

    Returns:
        Validated DataFrame with correct types
    """
    if schema_type == "raw":
        return raw_data_schema.validate(df)
    elif schema_type == "processed":
        return processed_data_schema.validate(df)
    else:
        raise ValueError(f"Unknown schema type: {schema_type}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test validation
    from src.data.ingestion import load_raw_data

    df = load_raw_data()

    # Validate
    is_valid, error = validate_raw_data(df, raise_error=False)
    print(f"Validation passed: {is_valid}")

    if error:
        print(f"Error: {error}")

    # Get report
    report = get_validation_report(df)
    print("\nValidation Report:")
    print(f"Total rows: {report['total_rows']}")
    print(f"Total columns: {report['total_columns']}")
