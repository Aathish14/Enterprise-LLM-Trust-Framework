"""
Unit tests for the factual correctness evaluation service.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock

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


class TestFactualCorrectnessSemanticScoring:
    """Regression tests for semantic-capable evaluation scoring.

    These tests verify that the evaluator scores based on semantic coverage of
    ground-truth concepts, not keyword overlap, and that it never treats a
    simulated or failed run as a genuine live result.
    """

    GROUND_TRUTH = (
        "Explainability, Robustness, Fairness, Privacy, Transparency, "
        "and Accountability."
    )

    def _make_evaluator(self, nim_response_text: str):
        """Create an evaluator whose LLM adapter returns the given text."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=nim_response_text)
        return FactualCorrectnessEvaluator(llm_adapter=mock_llm)

    @pytest.mark.asyncio
    async def test_full_coverage_different_wording_scores_high(self):
        """All concepts covered with synonyms should score >= 0.8."""
        ev = self._make_evaluator(
            "SCORE: 0.9\nCONFIDENCE: 0.95\n"
            "REASONING: 6 COVERED, 0 PARTIAL, 0 MISSING, 0 CONTRADICTED"
        )
        result = await ev.evaluate(
            "Our framework ensures interpretability, adversarial resilience, "
            "equitable outcomes, data protection, clear disclosures, and "
            "designated ownership.",
            self.GROUND_TRUTH,
        )
        assert result.score >= 0.8

    @pytest.mark.asyncio
    async def test_partial_coverage_missing_one_concept(self):
        """Missing one concept should score in the 0.4–0.7 range."""
        ev = self._make_evaluator(
            "SCORE: 0.6\nCONFIDENCE: 0.85\n"
            "REASONING: 4 COVERED, 1 PARTIAL, 1 MISSING, 0 CONTRADICTED"
        )
        result = await ev.evaluate(
            "We focus on explainability, robustness, fairness, "
            "transparency, and accountability.",
            self.GROUND_TRUTH,
        )
        assert 0.4 <= result.score <= 0.7

    @pytest.mark.asyncio
    async def test_unrelated_response_scores_near_zero(self):
        ev = self._make_evaluator(
            "SCORE: 0.0\nCONFIDENCE: 0.99\n"
            "REASONING: 0 COVERED, 0 PARTIAL, 6 MISSING, 0 CONTRADICTED"
        )
        result = await ev.evaluate(
            "The weather today is sunny with a high of 75 degrees.",
            self.GROUND_TRUTH,
        )
        assert result.score <= 0.1

    @pytest.mark.asyncio
    async def test_contradictory_response_scores_very_low(self):
        ev = self._make_evaluator(
            "SCORE: 0.1\nCONFIDENCE: 0.95\n"
            "REASONING: 0 COVERED, 0 PARTIAL, 3 MISSING, 3 CONTRADICTED"
        )
        result = await ev.evaluate(
            "AI governance does not require explainability or fairness. "
            "Privacy is irrelevant in enterprise settings.",
            self.GROUND_TRUTH,
        )
        assert result.score <= 0.2

    @pytest.mark.asyncio
    async def test_empty_response_scores_zero(self):
        ev = self._make_evaluator(
            "SCORE: 0.0\nCONFIDENCE: 1.0\n"
            "REASONING: 0 COVERED, 0 PARTIAL, 6 MISSING, 0 CONTRADICTED"
        )
        result = await ev.evaluate("", self.GROUND_TRUTH)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_malformed_evaluator_output_returns_failure(self):
        """Malformed LLM output must return score=0.0, not a plausible guess."""
        ev = self._make_evaluator(
            "I think the answer is pretty good overall."
        )
        result = await ev.evaluate("Some response", self.GROUND_TRUTH)
        assert result.score == 0.0
        assert "Failed to parse" in result.details.get("reasoning", "")

    @pytest.mark.asyncio
    async def test_out_of_range_score_is_clamped(self):
        ev = self._make_evaluator(
            "SCORE: 1.5\nCONFIDENCE: 0.9\nREASONING: Excellent"
        )
        result = await ev.evaluate("Some response", self.GROUND_TRUTH)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_provider_timeout_returns_failure(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = TimeoutError("NIM timeout")
        ev = FactualCorrectnessEvaluator(llm_adapter=mock_llm)
        result = await ev.evaluate("Some response", self.GROUND_TRUTH)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_simulated_result_never_marked_as_live(self):
        ev = self._make_evaluator(
            "SCORE: 0.8\nCONFIDENCE: 0.9\nREASONING: Simulated"
        )
        result = await ev.evaluate("Some response", self.GROUND_TRUTH)
        assert result.score == 0.8
        assert not result.details.get("simulated", False)
        assert result.details.get("source") != "live"


if __name__ == "__main__":
    pytest.main([__file__])