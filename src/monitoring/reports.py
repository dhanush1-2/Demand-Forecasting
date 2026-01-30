"""
Monitoring Reports Module

Generates comprehensive monitoring reports.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    RegressionPreset,
)
from evidently import ColumnMapping

from src.utils.logger import get_logger
from src.utils.config import get_paths

logger = get_logger(__name__)


class MonitoringReporter:
    """
    Generates various monitoring reports using Evidently.
    """
    
    def __init__(self):
        self.paths = get_paths()
        self.reports_dir = self.paths["outputs"] / "monitoring_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.column_mapping = ColumnMapping(
            target="target_demand",
            prediction="prediction"
        )
    
    def generate_data_quality_report(
        self,
        data: pd.DataFrame,
        save: bool = True
    ) -> dict:
        """
        Generate a data quality report.
        
        Args:
            data: Dataset to analyze.
            save: Whether to save HTML report.
            
        Returns:
            Quality metrics summary.
        """
        logger.info("Generating data quality report...")
        
        report = Report(metrics=[
            DataQualityPreset()
        ])
        
        report.run(
            reference_data=None,
            current_data=data
        )
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"data_quality_{timestamp}.html"
            report.save_html(str(report_path))
            logger.info(f"Data quality report saved: {report_path}")
        
        return report.as_dict()
    
    def generate_regression_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        save: bool = True
    ) -> dict:
        """
        Generate regression performance report.
        
        Both datasets must have 'target_demand' and 'prediction' columns.
        
        Args:
            reference_data: Reference (training) predictions.
            current_data: Current predictions to compare.
            save: Whether to save HTML report.
            
        Returns:
            Performance metrics.
        """
        logger.info("Generating regression performance report...")
        
        report = Report(metrics=[
            RegressionPreset()
        ])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"regression_report_{timestamp}.html"
            report.save_html(str(report_path))
            logger.info(f"Regression report saved: {report_path}")
        
        return report.as_dict()
    
    def generate_full_monitoring_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        save: bool = True
    ) -> dict:
        """
        Generate comprehensive monitoring report.
        
        Includes data drift and data quality analysis.
        
        Args:
            reference_data: Reference dataset.
            current_data: Current dataset.
            save: Whether to save HTML report.
            
        Returns:
            Full monitoring results.
        """
        logger.info("Generating full monitoring report...")
        
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data
        )
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"full_monitoring_{timestamp}.html"
            report.save_html(str(report_path))
            logger.info(f"Full monitoring report saved: {report_path}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "report_path": str(report_path) if save else None,
            "status": "completed"
        }
    
    def save_metrics_json(self, metrics: dict, name: str) -> Path:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary.
            name: Report name.
            
        Returns:
            Path to saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.reports_dir / f"{name}_{timestamp}.json"
        
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Metrics saved: {json_path}")
        return json_path
