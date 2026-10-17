"""
Benchmarking routes for the Enterprise LLM Trust Framework API.
"""

import asyncio
from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from ..schemas.requests import BenchmarkRequest, LLMProvider
from ..schemas.responses import BenchmarkResponse, BenchmarkResultSummary
from ..dependencies import (
    get_openai_api_key,
    get_gemini_api_key,
    get_anthropic_api_key,
    get_huggingface_api_key,
    get_settings
)
from ...llm.openai_adapter import OpenAIAdapter
from ...llm.gemini_adapter import GeminiAdapter
from ...llm.claude_adapter import ClaudeAdapter
from ...llm.huggingface_adapter import HuggingFaceAdapter
from ...benchmarking.benchmark_engine import ComparativeBenchmarkEngine
from ...benchmarking.benchmark_engine import create_llm_benchmark_config
from ...core.trust_scoring import TrustScoringService
from config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def get_llm_adapter(
    provider: LLMProvider,
    api_key: str,
    model_name: Optional[str] = None
):
    """Factory function to get the appropriate LLM adapter."""
    adapter_map = {
        LLMProvider.OPENAI: lambda: OpenAIAdapter(
            api_key=api_key,
            model_name=model_name or "gpt-4"
        ),
        LLMProvider.GEMINI: lambda: GeminiAdapter(
            api_key=api_key,
            model_name=model_name or "gemini-pro"
        ),
        LLMProvider.CLAUDE: lambda: ClaudeAdapter(
            api_key=api_key,
            model_name=model_name or "claude-3-sonnet-20240229"
        ),
        LLMProvider.HUGGINGFACE: lambda: HuggingFaceAdapter(
            api_key=api_key,
            model_name=model_name or "meta-llama/Llama-2-7b-chat-hf"
        ),
    }
    
    if provider not in adapter_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported LLM provider: {provider}"
        )
    
    return adapter_map[provider]()


@router.post("/", response_model=BenchmarkResponse)
async def run_benchmark(
    request: BenchmarkRequest,
    settings = Depends(get_settings)
):
    """
    Run a benchmark comparison across multiple LLM providers.
    
    Args:
        request: Benchmark request containing prompts and provider configuration
        
    Returns:
        BenchmarkResponse: Benchmark results for each provider
    """
    logger.info(f"Benchmark request received for {len(request.providers)} providers")
    
    try:
        # Prepare LLM adapters
        llm_adapters = {}
        
        for provider in request.providers:
            # Get API key (from request or environment)
            api_key = None
            if request.api_keys and provider in request.api_keys:
                api_key = request.api_keys[provider]
            else:
                # Get API key based on provider
                if provider == LLMProvider.OPENAI:
                    api_key = get_openai_api_key()
                elif provider == LLMProvider.GEMINI:
                    api_key = get_gemini_api_key()
                elif provider == LLMProvider.CLAUDE:
                    api_key = get_anthropic_api_key()
                elif provider == LLMProvider.HUGGINGFACE:
                    api_key = get_huggingface_api_key()
            
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No API key provided for provider {provider}"
                )
            
            # Get model name
            model_name = None
            if request.model_names and provider in request.model_names:
                model_name = request.model_names[provider]
            
            # Create adapter
            llm_adapters[provider.value] = get_llm_adapter(provider, api_key, model_name)
        
        # Prepare evaluation services (using the first provider's adapter for service initialization)
        # In a more sophisticated implementation, we might want to create services per provider
        # For simplicity, we'll use the first provider's adapter
        first_provider = request.providers[0]
        first_adapter = llm_adapters[first_provider.value]
        
        # We need evaluation services that work with the adapters
        # For benchmarking, we'll create a simple evaluation setup
        # Note: This is a simplification - in reality, each provider might need its own services
        from ....evaluation.factual_correctness import FactualCorrectnessEvaluator as FactualCorrectnessService
        from ....evaluation.response_quality import ResponseQualityEvaluator as ResponseQualityService
        
        evaluation_services = [
            FactualCorrectnessService(first_adapter),
            ResponseQualityService(first_adapter)
        ]
        
        # Get trust scoring service
        trust_scoring_service = TrustScoringService()
        
        # Create benchmark configuration
        config = create_llm_benchmark_config(
            benchmark_id=f"benchmark_{int(__import__('time').time())}",
            name="API Benchmark",
            description="Benchmark initiated via API",
            prompts=request.prompts,
            llm_adapters=llm_adapters,
            evaluation_services=evaluation_services,
            trust_scoring_service=trust_scoring_service,
            max_iterations=request.max_iterations
        )
        
        # Run benchmark
        logger.info(f"Running benchmark with {len(request.prompts)} prompts")
        engine = ComparativeBenchmarkEngine(config)
        results = await engine.run_benchmark()
        
        # Convert to API response format
        benchmark_results = []
        total_processing_time = 0.0
        
        for provider_name, provider_results in results.items():
            if not provider_results:
                continue
                
            # Calculate averages
            avg_trust_score = sum(r.trust_score.overall_score for r in provider_results) / len(provider_results)
            avg_processing_time = sum(r.processing_time for r in provider_results) / len(provider_results)
            total_processing_time += sum(r.processing_time for r in provider_results)
            
            benchmark_results.append(BenchmarkResultSummary(
                provider=provider_name,
                model_name=llm_adapters[provider_name].model_name,
                average_trust_score=avg_trust_score,
                average_processing_time=avg_processing_time,
                total_prompts=len(provider_results)
            ))
        
        return BenchmarkResponse(
            benchmark_id=config.benchmark_id,
            results=benchmark_results,
            total_processing_time=total_processing_time,
            timestamp=__import__('datetime').datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark failed: {str(e)}"
        )