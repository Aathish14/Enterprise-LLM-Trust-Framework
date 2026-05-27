"""
Failure tracking and alerting for the Enterprise LLM Trust Framework.
"""

import logging
import traceback
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from functools import wraps

from config.logging import get_logger

logger = get_logger(__name__)


class FailureTracker:
    """
    Tracks failures and can trigger alerts.
    """
    
    def __init__(self, alert_threshold: int = 5):
        """
        Initialize the failure tracker.
        
        Args:
            alert_threshold: Number of failures before triggering an alert
        """
        self.alert_threshold = alert_threshold
        self.failures: List[Dict[str, Any]] = []
        self.failure_counts: Dict[str, int] = {}
        logger.info(f"FailureTracker initialized with alert threshold: {alert_threshold}")
    
    def record_failure(self, 
                      exception: Exception, 
                      context: Optional[Dict[str, Any]] = None,
                      component: str = "unknown"):
        """
        Record a failure.
        
        Args:
            exception: The exception that occurred
            context: Optional context about when the failure occurred
            component: Component where the failure occurred
        """
        failure_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": component,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.failures.append(failure_info)
        
        # Update failure count for this component
        self.failure_counts[component] = self.failure_counts.get(component, 0) + 1
        
        # Log the failure
        logger.error(f"Failure recorded in {component}: {type(exception).__name__}: {exception}")
        logger.debug(f"Failure details: {failure_info}")
        
        # Check if we should trigger an alert
        if self.failure_counts[component] >= self.alert_threshold:
            self._trigger_alert(component, self.failure_counts[component])
    
    def _trigger_alert(self, component: str, count: int):
        """
        Trigger an alert for excessive failures.
        
        Args:
            component: Component that has exceeded the failure threshold
            count: Number of failures
        """
        alert_message = (
            f"ALERT: Component '{component}' has experienced {count} failures, "
            f"exceeding the threshold of {self.alert_threshold}."
        )
        logger.critical(alert_message)
        # In a real system, we would send this alert to a notification system
        # (e.g., email, Slack, PagerDuty, etc.)
        # For now, we just log it critically.
    
    def get_failure_history(self, 
                           component: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get failure history.
        
        Args:
            component: Optional component to filter by
            limit: Maximum number of failures to return
            
        Returns:
            List of failure dictionaries
        """
        if component:
            filtered = [f for f in self.failures if f["component"] == component]
            return filtered[-limit:] if len(filtered) > limit else filtered
        else:
            return self.failures[-limit:] if len(self.failures) > limit else self.failures
    
    def get_failure_counts(self) -> Dict[str, int]:
        """Get failure counts by component."""
        return self.failure_counts.copy()
    
    def reset(self):
        """Reset the failure history and counts."""
        self.failures.clear()
        self.failure_counts.clear()
        logger.info("FailureTracker history and counts reset")


# Global failure tracker instance
failure_tracker = FailureTracker()


def track_failures(component: str = "unknown"):
    """
    Decorator to track failures in a function.
    
    Args:
        component: Component name for failure tracking
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                failure_tracker.record_failure(e, 
                                             context={"args": str(args), "kwargs": str(kwargs)},
                                             component=component)
                raise e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                failure_tracker.record_failure(e,
                                             context={"args": str(args), "kwargs": str(kwargs)},
                                             component=component)
                raise e
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator