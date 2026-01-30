"""
Data Module

Handles data ingestion, validation, and transformation.
"""

from src.data.ingestion import (
    load_raw_data,
    load_processed_data,
    save_processed_data,
    get_data_info,
)
from src.data.validation import (
    validate_raw_data,
    validate_processed_data,
    validate_data,
    get_validation_report,
)
from src.data.transformation import (
    transform_data,
    remove_duplicates,
    handle_missing_values,
    convert_data_types,
    sort_data,
    add_basic_features,
)
from src.data.profiling import (
    generate_profile_report,
    generate_basic_stats,
    print_profile_summary,
)

__all__ = [
    # Ingestion
    "load_raw_data",
    "load_processed_data",
    "save_processed_data",
    "get_data_info",
    # Validation
    "validate_raw_data",
    "validate_processed_data",
    "validate_data",
    "get_validation_report",
    # Transformation
    "transform_data",
    "remove_duplicates",
    "handle_missing_values",
    "convert_data_types",
    "sort_data",
    "add_basic_features",
    # Profiling
    "generate_profile_report",
    "generate_basic_stats",
    "print_profile_summary",
]
