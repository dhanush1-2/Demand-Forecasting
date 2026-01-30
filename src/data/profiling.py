"""
Data Profiling Module

Generates exploratory data analysis (EDA) reports.
Since ydata-profiling doesn't support Python 3.13,
we create our own profiling functions.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import get_paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_basic_stats(df: pd.DataFrame) -> dict:
    """
    Generate basic statistics for a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with statistics
    """
    stats = {
        "overview": {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing": int(df.isnull().sum().sum()),
        },
        "columns": {},
    }

    for col in df.columns:
        col_stats = {
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isnull().sum()),
            "missing_pct": round(df[col].isnull().sum() / len(df) * 100, 2),
            "unique": int(df[col].nunique()),
            "unique_pct": round(df[col].nunique() / len(df) * 100, 2),
        }

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update(
                {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": round(float(df[col].mean()), 2),
                    "median": float(df[col].median()),
                    "std": round(float(df[col].std()), 2),
                    "zeros": int((df[col] == 0).sum()),
                    "zeros_pct": round((df[col] == 0).sum() / len(df) * 100, 2),
                }
            )

            # Quartiles
            col_stats["q1"] = float(df[col].quantile(0.25))
            col_stats["q3"] = float(df[col].quantile(0.75))
            col_stats["iqr"] = col_stats["q3"] - col_stats["q1"]

        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_stats.update(
                {
                    "min": str(df[col].min()),
                    "max": str(df[col].max()),
                    "range_days": (df[col].max() - df[col].min()).days,
                }
            )

        # Categorical/Object columns
        else:
            # Top 5 most frequent values
            top_values = df[col].value_counts().head(5).to_dict()
            col_stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        stats["columns"][col] = col_stats

    return stats


def generate_correlation_matrix(df: pd.DataFrame) -> dict:
    """
    Generate correlation matrix for numeric columns.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with correlation data
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        return {"message": "Not enough numeric columns for correlation"}

    # Calculate correlation
    corr_matrix = numeric_df.corr()

    # Convert to nested dict for JSON serialization
    corr_dict = {}
    for col in corr_matrix.columns:
        corr_dict[col] = {
            other_col: round(corr_matrix.loc[col, other_col], 3)
            for other_col in corr_matrix.columns
        }

    # Find highly correlated pairs (excluding self-correlation)
    high_corr = []
    for i, col1 in enumerate(corr_matrix.columns):
        for col2 in corr_matrix.columns[i + 1 :]:
            corr_val = corr_matrix.loc[col1, col2]
            if abs(corr_val) > 0.7:  # Threshold for "high" correlation
                high_corr.append({"col1": col1, "col2": col2, "correlation": round(corr_val, 3)})

    return {"matrix": corr_dict, "high_correlations": high_corr}


def generate_target_analysis(df: pd.DataFrame, target_col: str = "target_demand") -> dict:
    """
    Generate analysis focused on the target variable.

    Args:
        df: Input DataFrame
        target_col: Name of target column

    Returns:
        Dictionary with target analysis
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found"}

    target = df[target_col]

    analysis = {
        "basic_stats": {
            "mean": round(float(target.mean()), 2),
            "median": float(target.median()),
            "std": round(float(target.std()), 2),
            "min": float(target.min()),
            "max": float(target.max()),
            "skewness": round(float(target.skew()), 2),
            "kurtosis": round(float(target.kurtosis()), 2),
        },
        "distribution": {
            "q10": float(target.quantile(0.1)),
            "q25": float(target.quantile(0.25)),
            "q50": float(target.quantile(0.5)),
            "q75": float(target.quantile(0.75)),
            "q90": float(target.quantile(0.9)),
        },
        "value_ranges": {
            "0-10": int((target <= 10).sum()),
            "11-20": int(((target > 10) & (target <= 20)).sum()),
            "21-30": int(((target > 20) & (target <= 30)).sum()),
            "31-40": int(((target > 30) & (target <= 40)).sum()),
            "40+": int((target > 40).sum()),
        },
    }

    return analysis


def generate_profile_report(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    target_col: str = "target_demand",
) -> dict:
    """
    Generate a complete profile report for the DataFrame.

    Args:
        df: Input DataFrame
        output_path: Where to save the report (JSON). If None, doesn't save.
        target_col: Name of target column

    Returns:
        Complete profile report as dictionary
    """
    logger.info("Generating data profile report...")

    report = {
        "basic_stats": generate_basic_stats(df),
        "correlations": generate_correlation_matrix(df),
        "target_analysis": generate_target_analysis(df, target_col),
    }

    # Save to file if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to: {output_path}")

    logger.info("Profile report generated successfully")
    return report


def print_profile_summary(report: dict) -> None:
    """
    Print a human-readable summary of the profile report.

    Args:
        report: Profile report dictionary
    """
    print("\n" + "=" * 60)
    print("DATA PROFILE SUMMARY")
    print("=" * 60)

    overview = report["basic_stats"]["overview"]
    print("\n📊 Overview:")
    print(f"   Rows: {overview['rows']:,}")
    print(f"   Columns: {overview['columns']}")
    print(f"   Memory: {overview['memory_usage_mb']} MB")
    print(f"   Duplicates: {overview['duplicate_rows']}")
    print(f"   Missing values: {overview['total_missing']}")

    print("\n📈 Target Variable (target_demand):")
    target = report["target_analysis"]["basic_stats"]
    print(f"   Mean: {target['mean']}")
    print(f"   Std: {target['std']}")
    print(f"   Range: {target['min']} - {target['max']}")

    print("\n🔗 High Correlations (|r| > 0.7):")
    high_corr = report["correlations"].get("high_correlations", [])
    if high_corr:
        for item in high_corr:
            print(f"   {item['col1']} ↔ {item['col2']}: {item['correlation']}")
    else:
        print("   No high correlations found")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from src.data.ingestion import load_raw_data

    # Load data
    df = load_raw_data()

    # Generate report
    paths = get_paths()
    report_path = paths["outputs"] / "profiling_reports" / "raw_data_profile.json"

    report = generate_profile_report(df, output_path=report_path)

    # Print summary
    print_profile_summary(report)
