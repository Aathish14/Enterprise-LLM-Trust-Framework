"""
Integration tests for the evaluation pipeline.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.core.pipeline import EvaluationPipeline
from src.core.interfaces import EvaluationResult, TrustScore
from src.evaluation.factual_correctness import FactualCorrectnessEvaluator
from src.evaluation.hallucination_detection import HallucinationDetector
from src.core.trust_scoring import WeightedTrustScoringService
from langchain_core.messages import AIMessage


@pytest.mark.asyncio
async def test_evaluation_pipeline_with_real_evaluators():
    """Test the evaluation pipeline with real (but mocked LLM) evaluators."""
    # Create a mock LLM adapter
    mock_llm = Mock()
    mock_llm.generate = AsyncMock(return_value="The capital of France is Paris.")
    mock_llm.get_model_info.return_value = {"provider": "mock", "model_name": "mock-model"}
    
    # Create real evaluation services (but they'll use the mock LLM)
    factual_evaluator = FactualCorrectnessEvaluator()
    hallucination_evaluator = HallucinationDetector()
    
    # Mock the chain in the evaluators to avoid actual API calls
    mock_factual_result = AIMessage(content="SCORE: 0.9\nCONFIDENCE: 0.85\nREASONING: Factually correct.")
    mock_hallucination_result = AIMessage(content="SCORE: 0.95\nCONFIDENCE: 0.9\nREASONING: No hallucinations detected.")
    
    # Replace the chain of each evaluator with an AsyncMock that returns the predefined result
    mock_factual_chain = AsyncMock()
    mock_factual_chain.ainvoke.return_value = mock_factual_result
    mock_hallucination_chain = AsyncMock()
    mock_hallucination_chain.ainvoke.return_value = mock_hallucination_result
    
    factual_evaluator.chain = mock_factual_chain
    hallucination_evaluator.chain = mock_hallucination_chain
    
    # Create trust scoring service
    trust_scorer = WeightedTrustScoringService()
    
    # Create pipeline
    pipeline = EvaluationPipeline(
        llm_adapter=mock_llm,
        evaluation_services=[factual_evaluator, hallucination_evaluator],
        trust_scoring_service=trust_scorer
    )
    
    # Run evaluation
    result = await pipeline.run_evaluation(
        prompt="What is the capital of France?",
        context={"expected_answer": "Paris"}
    )
    
    # Assertions
    assert result["prompt"] == "What is the capital of France?"
    assert result["final_response"] == "The capital of France is Paris."
    assert len(result["evaluation_results"]) == 2
    
    # Check evaluation results
    eval_names = [er["metric_name"] for er in result["evaluation_results"]]
    assert "factual_correctness" in eval_names
    assert "hallucination_likelihood" in eval_names
    
    # Find specific results
    factual_result = next(er for er in result["evaluation_results"] if er["metric_name"] == "factual_correctness")
    hallucination_result = next(er for er in result["evaluation_results"] if er["metric_name"] == "hallucination_likelihood")
    
    assert factual_result["score"] == 0.9
    assert factual_result["confidence"] == 0.85
    assert hallucination_result["score"] == 0.95
    assert hallucination_result["confidence"] == 0.9
    
    # Check trust score
    assert result["trust_score"] is not None
    assert result["trust_score"]["overall_score"] > 0.8  # Should be high given the scores
    assert "factual_correctness" in result["trust_score"]["dimension_scores"]
    assert "hallucination_likelihood" in result["trust_score"]["dimension_scores"]
    
    # Check metadata
    assert "processing_time_seconds" in result["metadata"]
    assert result["metadata"]["iterations"] == 1  # No refinement needed
    
    # Verify LLM was called
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_evaluation_pipeline_with_refinement():
    """Test the evaluation pipeline with refinement loop."""
    # Create a mock LLM adapter that returns different responses
    mock_llm = Mock()
    mock_llm.generate = AsyncMock(side_effect=[
        "Initial incomplete response",  # First call
        "Improved complete response"    # Second call (after refinement)
    ])
    mock_llm.get_model_info.return_value = {"provider": "mock", "model_name": "mock-model"}
    
    # Create evaluation services
    factual_evaluator = FactualCorrectnessEvaluator()
    
    # Create a mock critique service that recommends refinement first, then accepts
    from src.core.interfaces import CritiqueService
    mock_critique = Mock(spec=CritiqueService)
    mock_critique.critique = AsyncMock(side_effect=[
        {"needs_refinement": True, "feedback": "Response is incomplete, needs more details"},
        {"needs_refinement": False, "feedback": "Response is now complete and accurate"}
    ])
    
    # Mock the chain in the evaluator to avoid actual API calls
    mock_result = AIMessage(content="SCORE: 0.7\nCONFIDENCE: 0.8\nREASONING: Mostly correct but incomplete.")
    
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = mock_result
    factual_evaluator.chain = mock_chain
    
    # Create pipeline
    pipeline = EvaluationPipeline(
        llm_adapter=mock_llm,
        evaluation_services=[factual_evaluator],
        critique_service=mock_critique,
        max_iterations=2
    )
    
    # Run evaluation
    result = await pipeline.run_evaluation(
        prompt="Describe the water cycle in detail.",
        context={}
    )
    
    # Assertions
    assert result["final_response"] == "Improved complete response"
    assert mock_llm.generate.call_count == 2  # Called twice due to refinement
    assert mock_critique.critique.call_count == 2  # Called twice
    assert result["metadata"]["iterations"] == 2  # Two iterations
    assert result["critique_feedback"] == {"needs_refinement": False, "feedback": "Response is now complete and accurate"}
    
    # Check that we have evaluation results from both iterations
    # Note: In the current implementation, we only keep the final evaluation results
    # but we could extend this to track all iterations if needed