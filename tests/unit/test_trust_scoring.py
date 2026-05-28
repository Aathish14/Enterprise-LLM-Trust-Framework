"""
Unit tests for the trust scoring service.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.trust_scoring import WeightedTrustScoringService, AdaptiveTrustScoringService
from src.core.interfaces import EvaluationResult, TrustScore
from datetime import datetime


@pytest.fixture
def sample_evaluation_results():
    """Create sample evaluation results for testing."""
    return [
        EvaluationResult(
            score=0.8,
            confidence=0.9,
            details={"reasoning": "Good factual accuracy"},
            metric_name="factual_correctness"
        ),
        EvaluationResult(
            score=0.6,
            confidence=0.8,
            details={"reasoning": "Some hallucinations detected"},
            metric_name="hallucination_likelihood"
        ),
        EvaluationResult(
            score=0.9,
            confidence=0.95,
            details={"reasoning": "High semantic similarity"},
            metric_name="semantic_similarity"
        )
    ]


def test_weighted_trust_scoring_service_initialization():
    """Test that the WeightedTrustScoringService initializes correctly."""
    scorer = WeightedTrustScoringService()
    assert scorer is not None
    assert len(scorer.weights) > 0
    
    # Test with custom weights
    custom_weights = {"factual_correctness": 0.5, "hallucination_likelihood": 0.5}
    scorer = WeightedTrustScoringService(weights=custom_weights)
    assert scorer.weights == {"factual_correctness": 0.5, "hallucination_likelihood": 0.5}


def test_weighted_trust_scoring_service_compute_score(sample_evaluation_results):
    """Test computing trust score with weighted averaging."""
    scorer = WeightedTrustScoringService()
    
    # Run the scoring
    import asyncio
    trust_score = asyncio.run(scorer.compute_trust_score(sample_evaluation_results))
    
    assert isinstance(trust_score, TrustScore)
    assert 0.0 <= trust_score.overall_score <= 1.0
    assert 0.0 <= trust_score.confidence <= 1.0
    assert isinstance(trust_score.dimension_scores, dict)
    # Should have scores for all configured weights
    assert len(trust_score.dimension_scores) == len(scorer.weights)
    
    # Check that dimension scores match the input for provided metrics
    for result in sample_evaluation_results:
        assert result.metric_name in trust_score.dimension_scores
        assert trust_score.dimension_scores[result.metric_name] == result.score
    
    # Check that missing metrics have score 0.0
    for metric_name in scorer.weights:
        if metric_name not in [r.metric_name for r in sample_evaluation_results]:
            assert trust_score.dimension_scores[metric_name] == 0.0


def test_weighted_trust_scoring_service_with_missing_results():
    """Test trust scoring when some evaluation results are missing."""
    # Create results with only some of the expected metrics
    partial_results = [
        EvaluationResult(
            score=0.7,
            confidence=0.8,
            details={},
            metric_name="factual_correctness"
        ),
        EvaluationResult(
            score=0.8,
            confidence=0.9,
            details={},
            metric_name="response_quality"
        )
        # Missing hallucination_likelihood, semantic_similarity, etc.
    ]
    
    scorer = WeightedTrustScoringService()
    
    import asyncio
    trust_score = asyncio.run(scorer.compute_trust_score(partial_results))
    
    assert isinstance(trust_score, TrustScore)
    assert 0.0 <= trust_score.overall_score <= 1.0
    # Should have scores for all configured weights, with missing ones treated as 0
    assert len(trust_score.dimension_scores) == len(scorer.weights)


def test_adaptive_trust_scoring_service_initialization():
    """Test that the AdaptiveTrustScoringService initializes correctly."""
    scorer = AdaptiveTrustScoringService()
    assert scorer is not None
    assert scorer.current_weights == scorer.initial_weights
    assert isinstance(scorer.performance_history, list)


def test_adaptive_trust_scoring_service_record_performance():
    """Test recording performance for adaptive weighting."""
    scorer = AdaptiveTrustScoringService()
    
    # Create sample results
    results = [
        EvaluationResult(
            score=0.8,
            confidence=0.9,
            details={},
            metric_name="factual_correctness"
        )
    ]
    
    trust_score = TrustScore(
        overall_score=0.8,
        dimension_scores={"factual_correctness": 0.8},
        confidence=0.9,
        details={}
    )
    
    # Record performance
    scorer._record_performance(results, trust_score)
    
    assert len(scorer.performance_history) == 1
    assert scorer.performance_history[0]["trust_score"]["overall_score"] == 0.8
    assert len(scorer.performance_history[0]["evaluation_results"]) == 1


def test_adaptive_trust_scoring_service_compute_score_delegates():
    """Test that AdaptiveTrustScoringService delegates to WeightedTrustScoringService."""
    scorer = AdaptiveTrustScoringService()
    
    results = [
        EvaluationResult(
            score=0.75,
            confidence=0.85,
            details={},
            metric_name="factual_correctness"
        )
    ]
    
    # Mock the weighted scorer to verify delegation
    with patch.object(scorer, '_record_performance'):
        import asyncio
        trust_score = asyncio.run(scorer.compute_trust_score(results))
        
        assert isinstance(trust_score, TrustScore)
        assert trust_score.overall_score == 0.75