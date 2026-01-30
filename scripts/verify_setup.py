#!/usr/bin/env python3
"""
Setup Verification Script

Run this script to verify that your project setup is correct.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def check_python_version():
    """Check Python version is 3.10+"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_directories():
    """Check all required directories exist"""
    print("\nChecking directories...")

    required_dirs = [
        "data/raw",
        "data/processed",
        "data/features",
        "src/data",
        "src/features",
        "src/models",
        "src/api/routes",
        "src/monitoring",
        "src/utils",
        "pipelines",
        "dashboard/pages",
        "tests",
        "configs",
        "models",
        "outputs",
        "logs",
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (missing)")
            all_exist = False

    return all_exist


def check_config_files():
    """Check configuration files exist"""
    print("\nChecking configuration files...")

    required_files = [
        "pyproject.toml",
        "configs/config.yaml",
        ".gitignore",
        ".pre-commit-config.yaml",
        ".env.example",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_exist = False

    return all_exist


def check_raw_data():
    """Check raw data file exists"""
    print("\nChecking raw data...")

    raw_data_path = project_root / "data" / "raw" / "demand_forecasting_dataset (1).csv"
    if raw_data_path.exists():
        # Get file size
        size_mb = raw_data_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Raw data file found ({size_mb:.2f} MB)")
        return True
    else:
        print(f"  ✗ Raw data file not found")
        print(f"    Expected: {raw_data_path}")
        return False


def check_imports():
    """Check if basic imports work"""
    print("\nChecking imports...")

    try:
        from src.utils.logger import get_logger
        print("  ✓ src.utils.logger")
    except ImportError as e:
        print(f"  ✗ src.utils.logger: {e}")
        return False

    try:
        from src.utils.config import get_config
        print("  ✓ src.utils.config")
    except ImportError as e:
        print(f"  ✗ src.utils.config: {e}")
        return False

    return True


def check_config_loading():
    """Check if config loads correctly"""
    print("\nChecking configuration loading...")

    try:
        from src.utils.config import get_config, get_paths

        config = get_config()
        print(f"  ✓ Config loaded successfully")
        print(f"    Project name: {config['project']['name']}")
        print(f"    Version: {config['project']['version']}")

        paths = get_paths()
        print(f"  ✓ Paths resolved successfully")
        print(f"    Base path: {paths['base']}")

        return True
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def check_logger():
    """Check if logger works"""
    print("\nChecking logger...")

    try:
        from src.utils.logger import get_logger

        logger = get_logger(__name__)
        logger.info("Test log message")
        print("  ✓ Logger working")
        return True
    except Exception as e:
        print(f"  ✗ Logger failed: {e}")
        return False


def main():
    """Run all checks"""
    print("=" * 60)
    print("DEMAND FORECASTING MLOPS - SETUP VERIFICATION")
    print("=" * 60)

    results = {
        "Python Version": check_python_version(),
        "Directories": check_directories(),
        "Config Files": check_config_files(),
        "Raw Data": check_raw_data(),
        "Imports": check_imports(),
        "Config Loading": check_config_loading(),
        "Logger": check_logger(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Install dependencies: poetry install")
        print("  2. Initialize DVC: dvc init")
        print("  3. Set up pre-commit: pre-commit install")
        print("=" * 60)
        return 0
    else:
        print("Some checks failed. Please fix the issues above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
