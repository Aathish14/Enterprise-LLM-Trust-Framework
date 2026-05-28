"""
Unit tests for the evaluation pipeline.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.core.pipeline import EvaluationPipeline
from src.core.interfaces import (
    EvaluationService, 
    CritiqueService, 
    TrustScoringService, 
    LLMAdapter,
    EvaluationResult,
    TrustScore
)


@pytest.fixture
def mock_llm_adapter():
    """Create a mock LLM adapter."""
    adapter = Mock(spec=LLMAdapter)
    adapter.generate = AsyncMock(return_value="Test response")
    adapter.get_model_info.return_value = {"provider": "test", "model_name": "test-model"}
    return adapter


@pytest.fixture
def mock_evaluation_service():
    """Create a mock evaluation service."""
    service = Mock(spec=EvaluationService)
    service.evaluate = AsyncMock(return_value=EvaluationResult(
        score=0.8,
        confidence=0.9,
        details={"reasoning": "Good"},
        metric_name="test_metric"
    ))
    return service


@pytest.fixture
def mock_critique_service():
    """Create a mock critique service."""
    service = Mock(spec=CritiqueService)
    service.critique = AsyncMock(return_value={"needs_refinement": False, "feedback": "Good job"})
    return service


@pytest.fixture
def mock_trust_scoring_service():
    """Create a mock trust scoring service."""
    service = Mock(spec=TrustScoringService)
    service.compute_trust_score = AsyncMock(return_value=TrustScore(
        overall_score=0.75,
        dimension_scores={"test_metric": 0.8},
        confidence=0.85,
        details={"reasoning": "Good overall"}
    ))
    return service


@pytest.mark.asyncio
async def test_evaluation_pipeline_initialization(mock_llm_adapter, mock_evaluation_service):
    """Test that the EvaluationPipeline initializes correctly."""
    pipeline = EvaluationPipeline(
        llm_adapter=mock_llm_adapter,
        evaluation_services=[mock_evaluation_service]
    )
    
    assert pipeline is not None
    assert pipeline.llm_adapter == mock_llm_adapter
    assert len(pipeline.evaluation_services) == 1
    assert pipeline.critique_service is None
    assert pipeline.trust_scoring_service is None
    assert pipeline.max_iterations == 2


    @pytest.mark.asyncio
    async def test_evaluation_pipeline_run_evaluation_basic(
        mock_llm_adapter, 
        mock_evaluation_service
    ):
        """Test running a basic evaluation pipeline without critique or trust scoring."""
        pipeline = EvaluationPipeline(
            llm_adapter=mock_llm_adapter,
            evaluation_services=[mock_evaluation_service],
            critique_service=None,
            trust_scoring_service=None
        )
        
        result = await pipeline.run_evaluation("Test prompt")
        
        # Check that LLM was called with prompt and context
        mock_llm_adapter.generate.assert_called_once_with("Test prompt", prompt="Test prompt")
        
        # Check that evaluation service was called
        mock_evaluation_service.evaluate.assert_called_once()
        
        # Check result structure
        assert "prompt" in result
        assert "final_response" in result
        assert "evaluation_results" in result
        assert "metadata" in result
        assert result["prompt"] == "Test prompt"
        assert result["final_response"] == "Test response"
        assert len(result["evaluation_results"]) == 1
        assert result["evaluation_results"][0]["metric_name"] == "test_metric"
        assert result["evaluation_results"][0]["score"] == 0.8
        assert result["trust_score"] is None
        assert result["critique_feedback"] is None


@pytest.mark.asyncio
async def test_evaluation_pipeline_with_critique_no_refinement(
    mock_llm_adapter, 
    mock_evaluation_service,
    mock_critique_service
):
    """Test evaluation pipeline with critique service that doesn't recommend refinement."""
    pipeline = EvaluationPipeline(
        llm_adapter=mock_llm_adapter,
        evaluation_services=[mock_evaluation_service],
        critique_service=mock_critique_service,
        trust_scoring_service=None,
        max_iterations=2
    )
    
    result = await pipeline.run_evaluation("Test prompt")
    
    # Check that critique service was called
    mock_critique_service.critique.assert_called_once()
    
    # Check that LLM was only called once (no refinement)
    assert mock_llm_adapter.generate.call_count == 1
    
    # Check result includes critique feedback
    assert result["critique_feedback"] == {"needs_refinement": False, "feedback": "Good job"}


@pytest.mark.asyncio
async def test_evaluation_pipeline_with_trust_scoring(
    mock_llm_adapter, 
    mock_evaluation_service,
    mock_trust_scoring_service
):
    """Test evaluation pipeline with trust scoring service."""
    pipeline = EvaluationPipeline(
        llm_adapter=mock_llm_adapter,
        evaluation_services=[mock_evaluation_service],
        critique_service=None,
        trust_scoring_service=mock_trust_scoring_service
    )
    
    result = await pipeline.run_evaluation("Test prompt")
    
    # Check that trust scoring service was called
    mock_trust_scoring_service.compute_trust_score.assert_called_once()
    
    # Check result includes trust score
    assert result["trust_score"] is not None
    assert result["trust_score"]["overall_score"] == 0.75
    assert result["trust_score"]["dimension_scores"]["test_metric"] == 0.8
    assert result["trust_score"]["confidence"] == 0.85


@pytest.mark.asyncio
async def test_evaluation_pipeline_with_refinement(
    mock_llm_adapter, 
    mock_evaluation_service,
    mock_critique_service
):
    """Test evaluation pipeline with refinement loop."""
    # Configure mock to recommend refinement on first call, then accept on second
    mock_critique_service.critique.side_effect = [
        {"needs_refinement": True, "feedback": "Needs improvement"},
        {"needs_refinement": False, "feedback": "Much better"}
    ]
    
    # Configure LLM to return different responses for refinement
    mock_llm_adapter.generate.side_effect = [
        "Initial response",
        "Improved response"
    ]
    
    pipeline = EvaluationPipeline(
        llm_adapter=mock_llm_adapter,
        evaluation_services=[mock_evaluation_service],
        critique_service=mock_critique_service,
        trust_scoring_service=None,
        max_iterations=2
    )
    
    result = await pipeline.run_evaluation("Test prompt")
    
    # Check that LLM was called twice (initial + refinement)
    assert mock_llm_adapter.generate.call_count == 2
    
    # Check that critique service was called twice
    assert mock_critique_service.critique.call_count == 2
    
    # Check that final response is the improved one
    assert result["final_response"] == "Improved response"
    
    # Check that metadata shows 2 iterations
    assert result["metadata"]["iterations"] == 2


@pytest.mark.asyncio
async def test_evaluation_pipeline_should_refine_logic(
    mock_llm_adapter, 
    mock_evaluation_service,
    mock_critique_service
):
    """Test the _should_refine logic."""
    pipeline = EvaluationPipeline(
        llm_adapter=mock_llm_adapter,
        evaluation_services=[mock_evaluation_service],
        critique_service=mock_critique_service
    )
    
    # Test with critique saying needs refinement
    critique_feedback = {"needs_refinement": True}
    evaluation_results = [EvaluationResult(score=0.8, confidence=0.9, details={}, metric_name="test")]
    assert pipeline._should_refine(critique_feedback, evaluation_results) == True
    
    # Test with critique saying no refinement but low score
    critique_feedback = {"needs_refinement": False}
    evaluation_results = [EvaluationResult(score=0.5, confidence=0.9, details={}, metric_name="test")]  # Below 0.6 threshold
    assert pipeline._should_refine(critique_feedback, evaluation_results) == True
    
    # Test with no refinement needed and good score
    critique_feedback = {"needs_refinement": False}
    evaluation_results = [EvaluationResult(score=0.8, confidence=0.9, details={}, metric_name="test")]
    assert pipeline._should_refine(critique_feedback, evaluation_results) == False
    
    # Test with None critique
    assert pipeline._should_refine(None, evaluation_results) == False