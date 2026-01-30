"""
Run Monitoring Script

Generate drift and quality reports.

Usage:
    python scripts/run_monitoring.py
"""

import sys
from pathlib import Path

from src.features.store import create_train_test_split, load_features
from src.monitoring.drift import DriftDetector
from src.monitoring.reports import MonitoringReporter

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def main():
    print("=" * 60)
    print("MONITORING REPORT GENERATION")
    print("=" * 60)

    # Load data
    print("\n1. Loading features...")
    df = load_features()
    train_df, test_df = create_train_test_split(df)

    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")

    # Initialize drift detector
    print("\n2. Initializing drift detector...")
    detector = DriftDetector(reference_data=train_df)

    # Check drift between train and test
    print("\n3. Checking data drift...")
    drift_results = detector.detect_drift(current_data=test_df)

    print("\n   Drift Results:")
    print(f"   - Dataset drift detected: {drift_results['dataset_drift_detected']}")
    print(f"   - Drift share: {drift_results['drift_share']:.2%}")
    print(f"   - Report saved: {drift_results.get('report_path', 'N/A')}")

    # Generate data quality report
    print("\n4. Generating data quality report...")
    reporter = MonitoringReporter()
    # quality_report = reporter.generate_data_quality_report(df)
    print("   Data quality report generated.")

    # Generate full monitoring report
    print("\n5. Generating full monitoring report...")
    full_report = reporter.generate_full_monitoring_report(
        reference_data=train_df, current_data=test_df
    )
    print(f"   Full report saved: {full_report.get('report_path', 'N/A')}")

    print("\n" + "=" * 60)
    print("MONITORING COMPLETE")
    print("=" * 60)
    print("\nReports saved in: outputs/monitoring_reports/")
    print("Open the HTML files in a browser to view interactive reports.")


if __name__ == "__main__":
    main()
