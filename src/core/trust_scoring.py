"""
Trust scoring service for the Enterprise LLM Trust Framework.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from core.interfaces import TrustScoringService, EvaluationResult, TrustScore
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


class WeightedTrustScoringService(TrustScoringService):
    """
    Computes composite trust scores using weighted average of evaluation dimensions.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the trust scoring service.
        
        Args:
            weights: Dictionary mapping metric names to weights (should sum to 1.0)
        """
        self.weights = weights or settings.TRUST_SCORE_WEIGHTS
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.info(f"WeightedTrustScoringService initialized with weights: {self.weights}")
    
    async def compute_trust_score(self, evaluation_results: List[EvaluationResult]) -> TrustScore:
        """
        Compute a composite trust score from multiple evaluation results.
        
        Args:
            evaluation_results: List of evaluation results from various metrics
            
        Returns:
            TrustScore containing the composite score and details
        """
        if not evaluation_results:
            logger.warning("No evaluation results provided for trust scoring")
            return TrustScore(
                overall_score=0.0,
                dimension_scores={},
                confidence=0.0,
                details={"error": "No evaluation results provided"}
            )
        
        # Map evaluation results by metric name
        results_by_metric = {result.metric_name: result for result in evaluation_results}
        
        # Calculate weighted score
        weighted_sum = 0.0
        total_weight = 0.0
        dimension_scores = {}
        confidences = []
        
        for metric_name, weight in self.weights.items():
            if metric_name in results_by_metric:
                result = results_by_metric[metric_name]
                weighted_sum += result.score * weight
                total_weight += weight
                dimension_scores[metric_name] = result.score
                confidences.append(result.confidence)
            else:
                logger.warning(f"Missing evaluation result for metric: {metric_name}")
                dimension_scores[metric_name] = 0.0
        
        # Calculate overall score (if we have weights for all metrics)
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Prepare details
        details = {
            "weights_used": self.weights,
            "metrics_evaluated": list(dimension_scores.keys()),
            "metrics_missing": [m for m in self.weights.keys() if m not in results_by_metric],
            "score_range": {
                "min": min(dimension_scores.values()) if dimension_scores else 0.0,
                "max": max(dimension_scores.values()) if dimension_scores else 0.0
            }
        }
        
        return TrustScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            confidence=avg_confidence,
            details=details
        )


class AdaptiveTrustScoringService(TrustScoringService):
    """
    Computes adaptive trust scores that adjust weights based on performance history.
    """
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the adaptive trust scoring service.
        
        Args:
            initial_weights: Initial weights for each metric
        """
        self.initial_weights = initial_weights or settings.TRUST_SCORE_WEIGHTS
        self.current_weights = self.initial_weights.copy()
        self.performance_history: List[Dict[str, Any]] = []
        
        logger.info(f"AdaptiveTrustScoringService initialized with weights: {self.current_weights}")
    
    async def compute_trust_score(self, evaluation_results: List[EvaluationResult]) -> TrustScore:
        """
        Compute a composite trust score from multiple evaluation results.
        
        Args:
            evaluation_results: List of evaluation results from various metrics
            
        Returns:
            TrustScore containing the composite score and details
        """
        # For now, delegate to weighted scoring (adaptive logic can be added later)
        weighted_scorer = WeightedTrustScoringService(self.current_weights)
        trust_score = await weighted_scorer.compute_trust_score(evaluation_results)
        
        # Record performance for future adaptation
        self._record_performance(evaluation_results, trust_score)
        
        return trust_score
    
    def _record_performance(self, evaluation_results: List[EvaluationResult], trust_score: TrustScore):
        """Record evaluation performance for future weight adaptation."""
        performance_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "evaluation_results": [
                {
                    "metric_name": result.metric_name,
                    "score": result.score,
                    "confidence": result.confidence
                }
                for result in evaluation_results
            ],
            "trust_score": {
                "overall_score": trust_score.overall_score,
                "dimension_scores": trust_score.dimension_scores,
                "confidence": trust_score.confidence
            }
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only last 100 records to prevent memory growth
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def adapt_weights(self):
        """
        Adapt weights based on historical performance.
        This is a placeholder for future implementation.
        """
        # TODO: Implement weight adaptation based on performance history
        # For example, increase weights for metrics that correlate strongly with human judgments
        logger.info("Weight adaptation not yet implemented")
        pass