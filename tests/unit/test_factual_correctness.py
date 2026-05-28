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
    assert result.score == 0.8
    assert result.confidence == 0.9
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
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert "error" in result.details
    assert result.details["error"] == "Test error"


def test_parse_evaluation_result():
    """Test the _parse_evaluation_result method."""
    evaluator = FactualCorrectnessEvaluator()
    
    # Test normal parsing
    result = "SCORE: 0.75\nCONFIDENCE: 0.8\nREASONING: Good response with minor issues."
    score, confidence, reasoning = evaluator._parse_evaluation_result(result)
    assert score == 0.75
    assert confidence == 0.8
    assert reasoning == "Good response with minor issues."
    
    # Test parsing with extra whitespace
    result = "SCORE: 0.5\nCONFIDENCE: 0.6\nREASONING: Average response"
    score, confidence, reasoning = evaluator._parse_evaluation_result(result)
    assert score == 0.5
    assert confidence == 0.6
    assert reasoning == "Average response"
    
    # Test parsing with invalid scores (should clamp to valid range)
    result = "SCORE: 1.5\nCONFIDENCE: -0.2\nREASONING: Invalid scores"
    score, confidence, reasoning = evaluator._parse_evaluation_result(result)
    assert score == 1.0  # Clamped to max
    assert confidence == 0.0  # Clamped to min
    assert reasoning == "Invalid scores"
    
    # Test parsing failure case
    result = "Invalid format"
    score, confidence, reasoning = evaluator._parse_evaluation_result(result)
    assert score == 0.5  # Default
    assert confidence == 0.5  # Default
    assert reasoning == "Failed to parse evaluation result"