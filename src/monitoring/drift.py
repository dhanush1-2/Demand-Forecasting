"""
Data Drift Detection Module

Uses Evidently AI to detect data and prediction drift.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric,
)

from src.utils.logger import get_logger
from src.utils.config import get_paths

logger = get_logger(__name__)


class DriftDetector:
    """
    Detects data drift between reference and current datasets.
    
    Uses Evidently AI for statistical drift detection.
    """
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize with reference (training) data.
        
        Args:
            reference_data: The training dataset to compare against.
        """
        self.reference_data = reference_data
        self.paths = get_paths()
        self.reports_dir = self.paths["outputs"] / "monitoring_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Define column mapping
        self.column_mapping = ColumnMapping(
            target="target_demand",
            numerical_features=self._get_numerical_features(),
            categorical_features=self._get_categorical_features()
        )
        
        logger.info(f"DriftDetector initialized with {len(reference_data)} reference samples")
    
    def _get_numerical_features(self) -> list:
        """Get numerical feature columns."""
        exclude = ["date", "target_demand"]
        numerical = self.reference_data.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()
        return [c for c in numerical if c not in exclude]
    
    def _get_categorical_features(self) -> list:
        """Get categorical feature columns."""
        return ["promotion_flag", "holiday_flag"]
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True
    ) -> dict:
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data: New data to check for drift.
            save_report: Whether to save HTML report.
            
        Returns:
            Dictionary with drift detection results.
        """
        logger.info(f"Checking drift for {len(current_data)} current samples")
        
        # Create drift report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract results
        results = report.as_dict()
        
        # Parse drift results
        drift_summary = self._parse_drift_results(results)
        
        # Save report if requested
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"drift_report_{timestamp}.html"
            report.save_html(str(report_path))
            drift_summary["report_path"] = str(report_path)
            logger.info(f"Drift report saved to: {report_path}")
        
        return drift_summary
    
    def _parse_drift_results(self, results: dict) -> dict:
        """Parse Evidently results into summary."""
        metrics = results.get("metrics", [])
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift_detected": False,
            "drift_share": 0.0,
            "drifted_features": [],
            "total_features": 0
        }
        
        for metric in metrics:
            metric_id = metric.get("metric", "")
            result = metric.get("result", {})
            
            if "DatasetDriftMetric" in metric_id:
                summary["dataset_drift_detected"] = result.get("dataset_drift", False)
                summary["drift_share"] = result.get("drift_share", 0.0)
                summary["total_features"] = result.get("number_of_columns", 0)
                summary["drifted_features_count"] = result.get("number_of_drifted_columns", 0)
        
        return summary
    
    def check_column_drift(
        self,
        current_data: pd.DataFrame,
        column: str
    ) -> dict:
        """
        Check drift for a specific column.
        
        Args:
            current_data: Current data to check.
            column: Column name to analyze.
            
        Returns:
            Drift results for the column.
        """
        report = Report(metrics=[
            ColumnDriftMetric(column_name=column)
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        results = report.as_dict()
        
        for metric in results.get("metrics", []):
            result = metric.get("result", {})
            return {
                "column": column,
                "drift_detected": result.get("drift_detected", False),
                "drift_score": result.get("drift_score", 0.0),
                "stattest_name": result.get("stattest_name", "unknown")
            }
        
        return {"column": column, "drift_detected": False}


def create_drift_detector_from_features() -> DriftDetector:
    """
    Create a DriftDetector using saved feature data.
    
    Returns:
        Initialized DriftDetector.
    """
    from src.features.store import load_features, create_train_test_split
    
    df = load_features()
    train_df, _ = create_train_test_split(df)
    
    return DriftDetector(reference_data=train_df)
