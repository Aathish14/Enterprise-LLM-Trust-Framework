"""
Health check routes for the Enterprise LLM Trust Framework API.
"""

from fastapi import APIRouter
from ..schemas.responses import HealthResponse
from config.settings import settings
from config.logging import get_logger
import time

logger = get_logger(__name__)

router = APIRouter()

# Track start time for uptime calculation
_start_time = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Current health status of the service
    """
    logger.debug("Health check requested")
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",  # TODO: Get from settings or package metadata
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        uptime_seconds=time.time() - _start_time
    )