"""
API request schemas for the Enterprise LLM Trust Framework.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    HUGGINGFACE = "huggingface"


class EvaluationRequest(BaseModel):
    """Request for evaluating a single LLM response."""
    prompt: str = Field(..., description="The input prompt to evaluate")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context for evaluation"
    )
    provider: LLMProvider = Field(
        LLMProvider.OPENAI, description="LLM provider to use"
    )
    model_name: Optional[str] = Field(
        default=None, description="Specific model name (optional)"
    )
    api_key: Optional[str] = Field(
        default=None, description="API key (optional, uses env var if not provided)"
    )
    max_iterations: int = Field(
        default=2, description="Maximum iterations for critique pipeline"
    )


class BenchmarkRequest(BaseModel):
    """Request for benchmarking multiple LLMs."""
    prompts: List[str] = Field(
        ..., description="List of prompts to evaluate"
    )
    providers: List[LLMProvider] = Field(
        ..., description="List of LLM providers to benchmark"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context for evaluation"
    )
    model_names: Optional[Dict[LLMProvider, str]] = Field(
        default=None, description="Specific model names for each provider"
    )
    api_keys: Optional[Dict[LLMProvider, str]] = Field(
        default=None, description="API keys for each provider"
    )
    max_iterations: int = Field(
        default=2, description="Maximum iterations for critique pipeline"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Framework version")
    timestamp: str = Field(..., description="Current timestamp")