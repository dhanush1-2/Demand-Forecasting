"""
API Schemas Module

Pydantic models for request validation and response serialization.
"""

from datetime import date as DateType

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Schema for single prediction request."""

    prediction_date: DateType = Field(..., description="Date for prediction")
    product_id: int = Field(..., ge=1, description="Product identifier")
    category_id: int = Field(..., ge=1, description="Category identifier")
    store_id: int = Field(..., ge=1, description="Store identifier")
    historical_sales: float = Field(..., ge=0, description="Historical sales value")
    price: float = Field(..., gt=0, description="Product price")
    promotion_flag: int = Field(..., ge=0, le=1, description="Promotion active (0 or 1)")
    holiday_flag: int = Field(..., ge=0, le=1, description="Holiday indicator (0 or 1)")
    economic_index: float = Field(..., description="Economic index value")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction_date": "2024-01-15",
                    "product_id": 1,
                    "category_id": 1,
                    "store_id": 1,
                    "historical_sales": 150.5,
                    "price": 29.99,
                    "promotion_flag": 1,
                    "holiday_flag": 0,
                    "economic_index": 102.5,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    predicted_demand: float = Field(..., description="Predicted demand value")
    model_used: str = Field(..., description="Name of model used for prediction")
    confidence: str = Field(default="high", description="Prediction confidence level")


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""

    predictions: list[PredictionRequest] = Field(
        ..., min_length=1, max_length=1000, description="List of prediction requests"
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""

    predictions: list[float] = Field(..., description="List of predicted demands")
    count: int = Field(..., description="Number of predictions made")
    model_used: str = Field(..., description="Name of model used")


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class ModelInfoResponse(BaseModel):
    """Schema for model information response."""

    model_name: str = Field(..., description="Name of the loaded model")
    model_type: str = Field(..., description="Type of model (xgboost, lightgbm)")
    features_count: int = Field(..., description="Number of input features")
    trained_date: str = Field(..., description="When the model was trained")


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional error details")
