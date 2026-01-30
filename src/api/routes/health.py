"""
Health Check Routes

Endpoints for service health monitoring.
"""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_model_service
from src.api.schemas import HealthResponse, ModelInfoResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API service is running and model is loaded.",
)
async def health_check(model_service=Depends(get_model_service)):
    """
    Basic health check endpoint.

    Returns service status and model availability.
    """
    return HealthResponse(
        status="healthy", version="1.0.0", model_loaded=model_service.is_model_loaded()
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness Check",
    description="Check if the service is ready to handle requests.",
)
async def readiness_check(model_service=Depends(get_model_service)):
    """
    Readiness probe for Kubernetes/Docker.

    Returns 200 only if model is loaded and ready.
    """
    is_ready = model_service.is_model_loaded()

    return HealthResponse(
        status="ready" if is_ready else "not_ready",
        version="1.0.0",
        model_loaded=is_ready,
    )


@router.get("/live", summary="Liveness Check", description="Simple liveness probe.")
async def liveness_check():
    """
    Liveness probe - just confirms the service is running.
    """
    return {"status": "alive"}


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Get information about the loaded model.",
)
async def model_info(model_service=Depends(get_model_service)):
    """
    Returns details about the currently loaded model.
    """
    return model_service.get_model_info()
