# Demand Forecasting MLOps

An end-to-end MLOps pipeline for retail demand forecasting.

## Features

- **Data Pipeline**: Automated data ingestion, validation, and transformation
- **Feature Engineering**: Time-based, lag, rolling, and interaction features
- **ML Models**: XGBoost and LightGBM with hyperparameter tuning
- **REST API**: FastAPI-based prediction service
- **Dashboard**: Interactive Streamlit dashboard
- **Monitoring**: Data drift detection with Evidently AI
- **Containerization**: Docker and Docker Compose
- **CI/CD**: GitHub Actions workflows

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11+ |
| Package Manager | Poetry |
| Data Processing | Pandas, Pandera |
| ML Models | XGBoost, LightGBM |
| Hyperparameter Tuning | Optuna |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Monitoring | Evidently AI |
| Testing | Pytest |
| Linting | Ruff, Black |
| Containerization | Docker |
| CI/CD | GitHub Actions |

## Quick Start

### Local Development

```bash
# Install dependencies
poetry install

# Run tests
make test

# Start API
make run-api

# Start Dashboard (in another terminal)
make run-dashboard
