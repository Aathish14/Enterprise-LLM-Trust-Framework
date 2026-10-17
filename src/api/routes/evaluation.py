"""
Evaluation routes for the Enterprise LLM Trust Framework API.
"""

import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from ..schemas.requests import EvaluationRequest, LLMProvider
from ..schemas.responses import EvaluationResponse, EvaluationResultDetail
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
from ...core.pipeline import EvaluationPipeline
from ...core.interfaces import EvaluationService, TrustScore
from ...evaluation.factual_correctness import FactualCorrectnessEvaluator as FactualCorrectnessService
from ...evaluation.hallucination_detection import HallucinationDetector as HallucinationDetectionService
from ...evaluation.semantic_similarity import SemanticSimilarityEvaluator as SemanticSimilarityService
from ...evaluation.response_quality import ResponseQualityEvaluator as ResponseQualityService
from ...evaluation.toxicity_safety import ToxicitySafetyEvaluator as ToxicitySafetyService
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


def get_evaluation_services(llm_adapter) -> List[EvaluationService]:
    """Get the standard evaluation services."""
    return [
        FactualCorrectnessService(llm_adapter),
        HallucinationDetectionService(llm_adapter),
        SemanticSimilarityService(llm_adapter),
        ResponseQualityService(llm_adapter),
        ToxicitySafetyService(llm_adapter),
    ]


@router.post("/", response_model=EvaluationResponse)
async def evaluate_prompt(
    request: EvaluationRequest,
    settings = Depends(get_settings)
):
    """
    Evaluate a single LLM response across multiple dimensions.
    
    Args:
        request: Evaluation request containing prompt and configuration
        
    Returns:
        EvaluationResponse: Evaluation results including trust score and metrics
    """
    logger.info(f"Evaluation request received for provider: {request.provider}")
    
    try:
        # Get API key (from request or environment)
        api_key = None
        if request.api_key:
            api_key = request.api_key
        else:
            # Get API key based on provider
            if request.provider == LLMProvider.OPENAI:
                api_key = get_openai_api_key()
            elif request.provider == LLMProvider.GEMINI:
                api_key = get_gemini_api_key()
            elif request.provider == LLMProvider.CLAUDE:
                api_key = get_anthropic_api_key()
            elif request.provider == LLMProvider.HUGGINGFACE:
                api_key = get_huggingface_api_key()
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No API key provided for provider {request.provider}"
            )
        
        # Get LLM adapter
        llm_adapter = get_llm_adapter(request.provider, api_key, request.model_name)
        
        # Get evaluation services
        evaluation_services = get_evaluation_services(llm_adapter)
        
        # Get trust scoring service
        trust_scoring_service = TrustScoringService()
        
        # Create evaluation pipeline
        pipeline = EvaluationPipeline(
            llm_adapter=llm_adapter,
            evaluation_services=evaluation_services,
            trust_scoring_service=trust_scoring_service,
            max_iterations=request.max_iterations
        )
        
        # Run evaluation
        logger.info(f"Running evaluation for prompt: {request.prompt[:50]}...")
        result = await pipeline.run_evaluation(
            prompt=request.prompt,
            context=request.context or {}
        )
        
        # Convert to API response format
        evaluation_results = {}
        for metric_name, eval_result in result["evaluation_results"].items():
            evaluation_results[metric_name] = EvaluationResultDetail(
                score=eval_result["score"],
                confidence=eval_result["confidence"],
                details=eval_result["details"]
            )
        
        trust_score = result["trust_score"]
        
        return EvaluationResponse(
            prompt=request.prompt,
            response=result["final_response"],
            trust_score=trust_score["overall_score"],
            confidence=trust_score["confidence"],
            evaluation_results=evaluation_results,
            processing_time=result.get("processing_time", 0.0),
            timestamp=result.get("timestamp", "")
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )