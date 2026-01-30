"""
Data Module

Handles data ingestion, validation, and transformation.
"""

from src.data.ingestion import (
    get_data_info,
    load_processed_data,
    load_raw_data,
    save_processed_data,
)
from src.data.profiling import (
    generate_basic_stats,
    generate_profile_report,
    print_profile_summary,
)
from src.data.transformation import (
    add_basic_features,
    convert_data_types,
    handle_missing_values,
    remove_duplicates,
    sort_data,
    transform_data,
)
from src.data.validation import (
    get_validation_report,
    validate_data,
    validate_processed_data,
    validate_raw_data,
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
