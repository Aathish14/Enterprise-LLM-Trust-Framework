"""
Latency monitoring for the Enterprise LLM Trust Framework.
"""

import time
import logging
from typing import Callable, Any, Optional
from functools import wraps

from config.logging import get_logger
from .metrics_logger import metrics_logger

logger = get_logger(__name__)


class LatencyMonitor:
    """
    Monitors latency of operations and provides statistics.
    """
    
    def __init__(self, operation_name: str):
        """
        Initialize the latency monitor.
        
        Args:
            operation_name: Name of the operation to monitor
        """
        self.operation_name = operation_name
        self.latencies: list[float] = []
        logger.info(f"LatencyMonitor initialized for operation: {operation_name}")
    
    def time_call(self, func: Callable) -> Callable:
        """
        Decorator to time a function call.
        
        Args:
            func: Function to time
            
        Returns:
            Wrapped function that logs latency
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                self.record_latency(latency)
                return result
            except Exception as e:
                latency = time.time() - start_time
                self.record_latency(latency, success=False)
                raise e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                self.record_latency(latency)
                return result
            except Exception as e:
                latency = time.time() - start_time
                self.record_latency(latency, success=False)
                raise e
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def record_latency(self, latency_seconds: float, success: bool = True):
        """
        Record a latency measurement.
        
        Args:
            latency_seconds: Latency in seconds
            success: Whether the operation was successful
        """
        self.latencies.append(latency_seconds)
        
        # Log to metrics logger
        metrics_logger.log_latency(
            latency_seconds, 
            operation=self.operation_name,
            run_id=None  # In a real scenario, we might pass the current run ID
        )
        
        logger.debug(f"Recorded latency for {self.operation_name}: {latency_seconds:.3f}s (success={success})")
    
    def get_statistics(self) -> dict[str, float]:
        """
        Get latency statistics.
        
        Returns:
            Dictionary containing latency statistics
        """
        if not self.latencies:
            return {}
        
        try:
            import numpy as np
            latencies = np.array(self.latencies)
            
            return {
                "count": len(latencies),
                "mean": np.mean(latencies),
                "median": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "std": np.std(latencies)
            }
        except Exception as e:
            logger.error(f"Failed to compute latency statistics: {e}")
            return {}
    
    def reset(self):
        """Reset the latency history."""
        self.latencies.clear()
        logger.info(f"Latency history reset for {self.operation_name}")


# Global latency monitors for common operations
llm_call_latency = LatencyMonitor("llm_call")
evaluation_latency = LatencyMonitor("evaluation")
pipeline_latency = LatencyMonitor("evaluation_pipeline")