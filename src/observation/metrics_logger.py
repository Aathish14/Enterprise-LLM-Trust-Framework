"""
Metrics logging for the Enterprise LLM Trust Framework.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import wraps

from config.logging import get_logger
from core.interfaces import EvaluationResult, TrustScore

logger = get_logger(__name__)


class MetricsLogger:
    """
    Handles logging of evaluation metrics, trust scores, and performance data.
    """
    
    def __init__(self):
        """Initialize the metrics logger."""
        self.evaluation_counts: Dict[str, int] = {}
        self.latency_history: List[float] = []
        logger.info("MetricsLogger initialized")
    
    def log_evaluation_result(self, 
                            result: EvaluationResult, 
                            run_id: Optional[str] = None,
                            iteration: int = 0):
        """
        Log an evaluation result.
        
        Args:
            result: EvaluationResult to log
            run_id: Optional experiment run ID
            iteration: Iteration number in the evaluation pipeline
        """
        try:
            # Prepare metrics for logging
            metrics = {
                f"evaluation_{result.metric_name}_score": result.score,
                f"evaluation_{result.metric_name}_confidence": result.confidence
            }
            
            # Add any numeric details as metrics
            for key, value in result.details.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[f"evaluation_{result.metric_name}_{key}"] = value
            
            # Log to experiment tracker if run_id provided
            if run_id:
                from .experiment_tracking import experiment_tracker
                experiment_tracker.log_metrics(run_id, metrics, step=iteration)
            
            # Update counters
            self.evaluation_counts[result.metric_name] = self.evaluation_counts.get(result.metric_name, 0) + 1
            
            logger.debug(f"Logged evaluation result for {result.metric_name}: score={result.score:.3f}, confidence={result.confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to log evaluation result: {e}")
    
    def log_trust_score(self, 
                       trust_score: TrustScore, 
                       run_id: Optional[str] = None,
                       iteration: int = 0):
        """
        Log a trust score.
        
        Args:
            trust_score: TrustScore to log
            run_id: Optional experiment run ID
            iteration: Iteration number in the evaluation pipeline
        """
        try:
            # Prepare metrics for logging
            metrics = {
                "trust_score_overall": trust_score.overall_score,
                "trust_score_confidence": trust_score.confidence
            }
            
            # Add dimension scores
            for dimension, score in trust_score.dimension_scores.items():
                metrics[f"trust_score_dimension_{dimension}"] = score
            
            # Log to experiment tracker if run_id provided
            if run_id:
                from .experiment_tracking import experiment_tracker
                experiment_tracker.log_metrics(run_id, metrics, step=iteration)
            
            logger.debug(f"Logged trust score: overall={trust_score.overall_score:.3f}, confidence={trust_score.confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to log trust score: {e}")
    
    def log_latency(self, 
                   latency_seconds: float, 
                   operation: str = "evaluation",
                   run_id: Optional[str] = None):
        """
        Log latency for an operation.
        
        Args:
            latency_seconds: Latency in seconds
            operation: Name of the operation
            run_id: Optional experiment run ID
        """
        try:
            # Add to history
            self.latency_history.append(latency_seconds)
            
            # Keep only last 1000 entries to prevent memory growth
            if len(self.latency_history) > 1000:
                self.latency_history = self.latency_history[-1000:]
            
            # Prepare metrics
            metrics = {
                f"latency_{operation}_seconds": latency_seconds,
                f"latency_{operation}_count": len(self.latency_history)
            }
            
            # Add summary statistics if we have enough data
            if len(self.latency_history) >= 5:
                import numpy as np
                metrics[f"latency_{operation}_mean"] = np.mean(self.latency_history[-100:])
                metrics[f"latency_{operation}_p50"] = np.percentile(self.latency_history[-100:], 50)
                metrics[f"latency_{operation}_p95"] = np.percentile(self.latency_history[-100:], 95)
                metrics[f"latency_{operation}_p99"] = np.percentile(self.latency_history[-100:], 99)
            
            # Log to experiment tracker if run_id provided
            if run_id:
                from .experiment_tracking import experiment_tracker
                experiment_tracker.log_metrics(run_id, metrics)
            
            logger.debug(f"Logged {operation} latency: {latency_seconds:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to log latency: {e}")
    
    def log_evaluation_pipeline_metrics(self, 
                                      pipeline_result: Dict[str, Any],
                                      run_id: Optional[str] = None):
        """
        Log metrics from a complete evaluation pipeline run.
        
        Args:
            pipeline_result: Result from the evaluation pipeline
            run_id: Optional experiment run ID
        """
        try:
            metrics = {}
            
            # Log pipeline-level metrics
            if "metadata" in pipeline_result:
                metadata = pipeline_result["metadata"]
                if "processing_time_seconds" in metadata:
                    metrics["pipeline_processing_time_seconds"] = metadata["processing_time_seconds"]
                if "iterations" in metadata:
                    metrics["pipeline_iterations"] = metadata["iterations"]
            
            # Log trust score if present
            if pipeline_result.get("trust_score"):
                ts = pipeline_result["trust_score"]
                metrics["pipeline_trust_score_overall"] = ts["overall_score"]
                metrics["pipeline_trust_score_confidence"] = ts["confidence"]
                
                # Log dimension scores
                for dim, score in ts["dimension_scores"].items():
                    metrics[f"pipeline_trust_score_dimension_{dim}"] = score
            
            # Log evaluation scores
            if pipeline_result.get("evaluation_results"):
                for eval_result in pipeline_result["evaluation_results"]:
                    metric_name = eval_result["metric_name"]
                    metrics[f"pipeline_evaluation_{metric_name}_score"] = eval_result["score"]
                    metrics[f"pipeline_evaluation_{metric_name}_confidence"] = eval_result["confidence"]
            
            # Log to experiment tracker if run_id provided
            if run_id:
                from .experiment_tracking import experiment_tracker
                experiment_tracker.log_metrics(run_id, metrics)
            
            logger.debug(f"Logged pipeline metrics for run {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to log evaluation pipeline metrics: {e}")
    
    def get_evaluation_counts(self) -> Dict[str, int]:
        """Get counts of evaluations performed by metric."""
        return self.evaluation_counts.copy()
    
    def get_latency_stats(self, operation: str = "evaluation") -> Dict[str, float]:
        """
        Get latency statistics for an operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Dictionary containing latency statistics
        """
        if not self.latency_history:
            return {}
        
        try:
            import numpy as np
            latency_key = f"latency_{operation}_seconds"
            # Filter latency history for this operation (simplified - in practice we'd store operation-specific histories)
            recent_latencies = self.latency_history[-100:] if len(self.latency_history) >= 100 else self.latency_history
            
            return {
                "count": len(recent_latencies),
                "mean": np.mean(recent_latencies),
                "median": np.percentile(recent_latencies, 50),
                "p95": np.percentile(recent_latencies, 95),
                "p99": np.percentile(recent_latencies, 99),
                "min": np.min(recent_latencies),
                "max": np.max(recent_latencies)
            }
        except Exception as e:
            logger.error(f"Failed to calculate latency stats: {e}")
            return {}


# Global metrics logger instance
metrics_logger = MetricsLogger()


def log_evaluation_time(func):
    """
    Decorator to log execution time of evaluation functions.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            latency = time.time() - start_time
            metrics_logger.log_latency(latency, func.__name__)
            return result
        except Exception as e:
            latency = time.time() - start_time
            metrics_logger.log_latency(latency, f"{func.__name__}_failed")
            raise e
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start_time
            metrics_logger.log_latency(latency, func.__name__)
            return result
        except Exception as e:
            latency = time.time() - start_time
            metrics_logger.log_latency(latency, f"{func.__name__}_failed")
            raise e
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper