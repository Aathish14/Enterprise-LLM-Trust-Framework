"""
Unit tests for the factual correctness evaluation service.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.evaluation.factual_correctness import FactualCorrectnessEvaluator
from src.core.interfaces import EvaluationResult
from src.core.trust_scoring import WeightedTrustScoringService
from langchain_core.messages import AIMessage


@pytest.mark.asyncio
async def test_factual_correctness_evaluator_initialization():
    """Test that the FactualCorrectnessEvaluator initializes correctly."""
    evaluator = FactualCorrectnessEvaluator()
    assert evaluator is not None
    assert evaluator.llm is not None
    assert evaluator.prompt_template is not None


@pytest.mark.asyncio
async def test_factual_correctness_evaluator_evaluate():
    """Test the evaluate method of FactualCorrectnessEvaluator."""
    evaluator = FactualCorrectnessEvaluator()
    
    # Mock the chain to return a predictable result
    mock_chain = AsyncMock()
    mock_result = AIMessage(content="SCORE: 0.8\nCONFIDENCE: 0.9\nREASONING: The response contains mostly accurate information with minor inaccuracies.")
    mock_chain.ainvoke.return_value = mock_result
    evaluator.chain = mock_chain
    
    result = await evaluator.evaluate(
        response="The sky is blue and grass is green.",
        context={"expected_answer": "The sky is blue and grass is green."}
    )
    
    assert isinstance(result, EvaluationResult)
    assert result.metric_name == "factual_correctness"
    assert result.score == pytest.approx(0.8, rel=1e-3)
    assert result.confidence == pytest.approx(0.9, rel=1e-3)
    assert "reasoning" in result.details
    assert result.details["reasoning"] == "The response contains mostly accurate information with minor inaccuracies."


@pytest.mark.asyncio
async def test_factual_correctness_evaluator_evaluate_error_handling():
    """Test error handling in the evaluate method."""
    evaluator = FactualCorrectnessEvaluator()
    
    # Mock the chain to raise an exception
    mock_chain = AsyncMock()
    mock_chain.ainvoke.side_effect = Exception("Test error")
    evaluator.chain = mock_chain
    
    result = await evaluator.evaluate(
        response="Test response",
        context={}
    )
    
    assert isinstance(result, EvaluationResult)
    assert result.metric_name == "factual_correctness"
    assert result.score == pytest.approx(0.0, abs=1e-6)
    assert result.confidence == pytest.approx(0.0, abs=1e-6)
    assert "error" in result.details
    assert result.details["error"] == "Test error"


def test_parse_evaluation_result():
    """Test the _parse_evaluation_result method."""
    evaluator = FactualCorrectnessEvaluator()
    
    # Test normal parsing
    result = "SCORE: 0.75\nCONFIDENCE: 0.8\nREASONING: Good response with minor issues."
    score, confidence, reasoning = evaluator._parse_evaluation_result(result)
    assert score == pytest.approx(0.75, rel=1e-3)
    assert confidence == pytest.approx(0.8, rel=1e-3)
    assert reasoning == "Good response with minor issues."
    
    # Test parsing with extra whitespace
    result = "SCORE: 0.5\nCONFIDENCE: 0.6\nREASONING: Average response"
    score, confidence, reasoning = evaluator._parse_evaluation_result(result)
    assert score == pytest.approx(0.5, rel=1e-3)
    assert confidence == pytest.approx(0.6, rel=1e-3)
    assert reasoning == "Average response"
    
    # Test parsing with invalid scores (should clamp to valid range)
    result = "SCORE: 1.5\nCONFIDENCE: -0.2\nREASONING: Invalid scores"
    score, confidence, reasoning = evaluator._parse_evaluation_result(result)
    assert score == pytest.approx(1.0, rel=1e-3)  # Clamped to max
    assert confidence == pytest.approx(0.0, abs=1e-6)  # Clamped to min
    assert reasoning == "Invalid scores"
    
    # Test parsing failure case — parse failure returns 0.0/0.0, not neutral 0.5
    result = "Invalid format"
    score, confidence, reasoning = evaluator._parse_evaluation_result(result)
    assert score == pytest.approx(0.0, abs=1e-6)
    assert confidence == pytest.approx(0.0, abs=1e-6)
    assert reasoning == "Failed to parse evaluation result"


@pytest.mark.asyncio
async def test_evaluate_provider_failure_returns_zero_score():
    """Provider failure must not result in a neutral 0.5 score.

    When the LLM adapter raises an exception, evaluate() must return
    score=0.0 and confidence=0.0 with an error detail, never a neutral
    0.5 score that could be mistaken for a real measurement.
    """
    evaluator = FactualCorrectnessEvaluator()

    mock_adapter = Mock()
    mock_adapter.generate = AsyncMock(side_effect=Exception("API rate limit exceeded"))
    evaluator.llm_adapter = mock_adapter
    evaluator.llm = None
    evaluator.chain = None

    result = await evaluator.evaluate(
        response="Some response text",
        context={"expected_facts": "Some context"},
    )

    assert isinstance(result, EvaluationResult)
    assert result.metric_name == "factual_correctness"
    assert result.score == pytest.approx(0.0, abs=1e-6)
    assert result.confidence == pytest.approx(0.0, abs=1e-6)
    assert "error" in result.details


def test_weighted_trust_scoring_missing_metric_not_reported_as_measured():
    """A missing evaluation metric must not appear as a measured score.

    When a metric has no weight (not in TRUST_SCORE_WEIGHTS), it must not
    appear in dimension_scores at all — not even with score 0.0.
    """
    scorer = WeightedTrustScoringService()

    results = [
        EvaluationResult(
            score=0.8,
            confidence=0.9,
            details={},
            metric_name="factual_correctness",
        ),
        EvaluationResult(
            score=0.6,
            confidence=0.9,
            details={},
            metric_name="hallucination_likelihood",
        ),
    ]

    trust_score = asyncio.run(scorer.compute_trust_score(results))

    assert "factual_correctness" in trust_score.dimension_scores
    assert "hallucination_likelihood" not in trust_score.dimension_scores
    assert trust_score.details["metrics_missing"] == []


if __name__ == "__main__":
    pytest.main([__file__])