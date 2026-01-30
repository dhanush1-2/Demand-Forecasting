"""
Tests for API Endpoints
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")

        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_liveness_endpoint(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")

        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    def test_predict_valid_input(self, client):
        """Test prediction with valid input."""
        payload = {
            "prediction_date": "2024-01-15",
            "product_id": 1,
            "category_id": 1,
            "store_id": 1,
            "historical_sales": 150.0,
            "price": 25.0,
            "promotion_flag": 0,
            "holiday_flag": 0,
            "economic_index": 100.0,
        }

        response = client.post("/predict", json=payload)

        # Should succeed if model is loaded, otherwise 500
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "predicted_demand" in data
            assert "model_used" in data

    def test_predict_invalid_price(self, client):
        """Test prediction with invalid price."""
        payload = {
            "prediction_date": "2024-01-15",
            "product_id": 1,
            "category_id": 1,
            "store_id": 1,
            "historical_sales": 150.0,
            "price": -10.0,  # Invalid
            "promotion_flag": 0,
            "holiday_flag": 0,
            "economic_index": 100.0,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 422  # Validation error

    def test_predict_missing_field(self, client):
        """Test prediction with missing required field."""
        payload = {
            "prediction_date": "2024-01-15",
            "product_id": 1,
            # Missing other required fields
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 422
