"""
API dependencies for the Enterprise LLM Trust Framework.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)

# Security
security = HTTPBearer(auto_error=False)


async def get_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """
    Extract API key from Authorization header.
    Returns None if no credentials provided (for endpoints that don't require auth).
    """
    if credentials:
        return credentials.credentials
    return None


def get_settings():
    """Dependency to get application settings."""
    from config.settings import settings
    return settings


# Provider-specific API key dependencies
async def get_openai_api_key(
    api_key: Optional[str] = Depends(get_api_key),
) -> str:
    """Get OpenAI API key from header or environment."""
    if api_key:
        return api_key
    if settings.OPENAI_API_KEY:
        return settings.OPENAI_API_KEY.get_secret_value() if hasattr(settings.OPENAI_API_KEY, 'get_secret_value') else str(settings.OPENAI_API_KEY)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="OpenAI API key required. Provide via Authorization header or OPENAI_API_KEY environment variable.",
    )


async def get_gemini_api_key(
    api_key: Optional[str] = Depends(get_api_key),
) -> str:
    """Get Gemini API key from header or environment."""
    if api_key:
        return api_key
    if settings.GEMINI_API_KEY:
        return settings.GEMINI_API_KEY.get_secret_value() if hasattr(settings.GEMINI_API_KEY, 'get_secret_value') else str(settings.GEMINI_API_KEY)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Gemini API key required. Provide via Authorization header or GEMINI_API_KEY environment variable.",
    )


async def get_anthropic_api_key(
    api_key: Optional[str] = Depends(get_api_key),
) -> str:
    """Get Anthropic API key from header or environment."""
    if api_key:
        return api_key
    if settings.ANTHROPIC_API_KEY:
        return settings.ANTHROPIC_API_KEY.get_secret_value() if hasattr(settings.ANTHROPIC_API_KEY, 'get_secret_value') else str(settings.ANTHROPIC_API_KEY)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Anthropic API key required. Provide via Authorization header or ANTHROPIC_API_KEY environment variable.",
    )


async def get_huggingface_api_key(
    api_key: Optional[str] = Depends(get_api_key),
) -> str:
    """Get HuggingFace API key from header or environment."""
    if api_key:
        return api_key
    if settings.HUGGINGFACE_API_KEY:
        return settings.HUGGINGFACE_API_KEY.get_secret_value() if hasattr(settings.HUGGINGFACE_API_KEY, 'get_secret_value') else str(settings.HUGGINGFACE_API_KEY)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="HuggingFace API key required. Provide via Authorization header or HUGGINGFACE_API_KEY environment variable.",
    )