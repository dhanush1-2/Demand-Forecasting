"""
FastAPI Application

Main entry point for the Demand Forecasting API.

Usage:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import get_model_service
from src.api.routes import health, predictions
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.
    """
    # Startup: Load model
    logger.info("Starting Demand Forecasting API...")
    try:
        model_service = get_model_service()
        logger.info(f"Model loaded: {model_service.model_name}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Demand Forecasting API",
    description="""
    ## Demand Forecasting REST API

    This API provides demand prediction capabilities using machine learning models.

    ### Features:
    - Single prediction endpoint
    - Batch predictions for efficiency
    - Health check endpoints for monitoring
    - Model information endpoint

    ### Models:
    - LightGBM (default, better accuracy)
    - XGBoost (fallback)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add response timing header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2)) + "ms"
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )


# Include routers
app.include_router(health.router)
app.include_router(predictions.router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Demand Forecasting API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    api_config = config.get("api", {})

    uvicorn.run(
        "src.api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=True,
    )
