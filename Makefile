.PHONY: help install test lint format run-api run-dashboard docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  install       Install dependencies with Poetry"
	@echo "  test          Run tests with pytest"
	@echo "  lint          Run linting with Ruff"
	@echo "  format        Format code with Black"
	@echo "  run-api       Start the FastAPI server"
	@echo "  run-dashboard Start the Streamlit dashboard"
	@echo "  docker-build  Build Docker images"
	@echo "  docker-up     Start Docker containers"
	@echo "  docker-down   Stop Docker containers"
	@echo "  clean         Clean up cache files"

install:
	poetry install

test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ -v --cov=src --cov-report=html

lint:
	poetry run ruff check src/ tests/

format:
	poetry run black src/ tests/
	poetry run ruff check --fix src/ tests/

run-api:
	poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	poetry run streamlit run dashboard/app.py

run-monitoring:
	poetry run python scripts/run_monitoring.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage coverage.xml
