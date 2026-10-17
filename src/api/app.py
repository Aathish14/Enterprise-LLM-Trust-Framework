"""
Main FastAPI application for the Enterprise LLM Trust Framework.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import evaluation, benchmarking, health
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Enterprise LLM Trust Framework API",
    description="API for evaluating, monitoring, and benchmarking Large Language Models",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(evaluation.router, prefix="/evaluate", tags=["evaluation"])
app.include_router(benchmarking.router, prefix="/benchmark", tags=["benchmarking"])


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting Enterprise LLM Trust Framework API")
    logger.info(f"Version: {settings.app_version if hasattr(settings, 'app_version') else '0.1.0'}")
    logger.info(f"Debug mode: {settings.debug}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down Enterprise LLM Trust Framework API")