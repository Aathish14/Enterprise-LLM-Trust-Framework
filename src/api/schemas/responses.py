"""
API response schemas for the Enterprise LLM Trust Framework.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class EvaluationResultDetail(BaseModel):
    """Individual evaluation metric result."""
    score: float = Field(..., description="Score between 0.0 and 1.0")
    confidence: float = Field(..., description="Confidence in the score")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )


class EvaluationResponse(BaseModel):
    """Response for a single evaluation."""
    prompt: str = Field(..., description="The input prompt")
    response: str = Field(..., description="Generated LLM response")
    trust_score: float = Field(
        ..., description="Overall trust score (0.0 to 1.0)"
    )
    confidence: float = Field(
        ..., description="Confidence in the trust score"
    )
    evaluation_results: Dict[str, EvaluationResultDetail] = Field(
        ..., description="Individual evaluation metric results"
    )
    processing_time: float = Field(
        ..., description="Processing time in seconds"
    )
    timestamp: str = Field(..., description="Timestamp of evaluation")


class BenchmarkResultSummary(BaseModel):
    """Summary of benchmark results for a single provider."""
    provider: str = Field(..., description="LLM provider name")
    model_name: str = Field(..., description="Model name used")
    average_trust_score: float = Field(
        ..., description="Average trust score across all prompts"
    )
    average_processing_time: float = Field(
        ..., description="Average processing time in seconds"
    )
    total_prompts: int = Field(
        ..., description="Total number of prompts evaluated"
    )


class BenchmarkResponse(BaseModel):
    """Response for a benchmark comparison."""
    benchmark_id: str = Field(..., description="Unique benchmark identifier")
    results: List[BenchmarkResultSummary] = Field(
        ..., description="Results for each provider"
    )
    total_processing_time: float = Field(
        ..., description="Total processing time in seconds"
    )
    timestamp: str = Field(..., description="Timestamp of benchmark")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Framework version")
    timestamp: str = Field(..., description="Current timestamp")
    uptime_seconds: Optional[float] = Field(
        default=None, description="Uptime in seconds (if available)"
    )