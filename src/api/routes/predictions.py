"""
Prediction Routes

Endpoints for making demand predictions.
"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_model_service
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.utils.logger import get_logger

router = APIRouter(prefix="/predict", tags=["Predictions"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Single Prediction",
    description="Make a single demand prediction.",
)
async def predict_single(request: PredictionRequest, model_service=Depends(get_model_service)):
    """
    Make a single demand prediction.

    Takes product, store, and contextual information to predict demand.
    """
    try:
        # Convert request to features
        features = model_service.prepare_features(request)

        # Make prediction
        prediction = model_service.predict(features)

        logger.info(f"Prediction made: {prediction:.2f} for product {request.product_id}")

        return PredictionResponse(
            predicted_demand=round(prediction, 2),
            model_used=model_service.model_name,
            confidence="high",
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Predictions",
    description="Make multiple predictions in a single request.",
)
async def predict_batch(request: BatchPredictionRequest, model_service=Depends(get_model_service)):
    """
    Make batch predictions for multiple inputs.

    More efficient than making multiple single requests.
    """
    try:
        # Prepare all features
        all_features = [
            model_service.prepare_features(pred_request) for pred_request in request.predictions
        ]

        # Make batch predictions
        predictions = model_service.predict_batch(all_features)

        logger.info(f"Batch prediction completed: {len(predictions)} predictions")

        return BatchPredictionResponse(
            predictions=[round(p, 2) for p in predictions],
            count=len(predictions),
            model_used=model_service.model_name,
        )

    except ValueError as e:
        logger.error(f"Batch validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")
