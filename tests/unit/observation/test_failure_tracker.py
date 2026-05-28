"""
Unit tests for the failure tracking component.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from observation.failure_tracker import FailureTracker


@pytest.fixture
def failure_tracker():
    """Create a FailureTracker instance for testing."""
    return FailureTracker(alert_threshold=2)  # Lower threshold for easier testing


def test_failure_tracker_initialization(failure_tracker):
    """Test that the FailureTracker initializes correctly."""
    assert failure_tracker is not None
    assert failure_tracker.alert_threshold == 2
    assert isinstance(failure_tracker.failures, list)
    assert len(failure_tracker.failures) == 0
    assert isinstance(failure_tracker.failure_counts, dict)
    assert len(failure_tracker.failure_counts) == 0


def test_record_failure(failure_tracker):
    """Test recording a failure."""
    with patch('observation.failure_tracker.logger'):
        try:
            raise ValueError("This is a test error")
        except ValueError as e:
            failure_tracker.record_failure(
                exception=e,
                context={"key": "value"},
                component="test_component"
            )
        
        assert len(failure_tracker.failures) == 1
        failure = failure_tracker.failures[0]
        assert failure["component"] == "test_component"
        assert failure["exception_type"] == "ValueError"
        assert failure["exception_message"] == "This is a test error"
        assert failure["context"] == {"key": "value"}
        assert "timestamp" in failure
        
        # Check failure count was updated
        assert failure_tracker.failure_counts["test_component"] == 1


def test_record_failure_without_context(failure_tracker):
    """Test recording a failure without context."""
    with patch('observation.failure_tracker.logger'):
        try:
            raise RuntimeError("Another test error")
        except RuntimeError as e:
            failure_tracker.record_failure(
                exception=e,
                component="test_component"
            )
        
        assert len(failure_tracker.failures) == 1  # Only this failure in fresh instance
        failure = failure_tracker.failures[0]
        assert failure["context"] == {}


def test_get_failure_history(failure_tracker):
    """Test getting failure history."""
    with patch('observation.failure_tracker.logger'):
        # Record a few failures
        try:
            raise ValueError("Error 1")
        except ValueError as e:
            failure_tracker.record_failure(e, component="test_component")
        
        try:
            raise TypeError("Error 2")
        except TypeError as e:
            failure_tracker.record_failure(e, component="test_component")
        
        try:
            raise RuntimeError("Error 3")
        except RuntimeError as e:
            failure_tracker.record_failure(e, component="other_component")
        
        # Get all failures
        history = failure_tracker.get_failure_history()
        assert len(history) == 3
        
        # Get failures for specific component
        history = failure_tracker.get_failure_history(component="test_component")
        assert len(history) == 2
        assert all(f["component"] == "test_component" for f in history)
        
        # Get failures with limit
        history = failure_tracker.get_failure_history(limit=1)
        assert len(history) == 1


def test_get_failure_counts(failure_tracker):
    """Test getting failure counts."""
    with patch('observation.failure_tracker.logger'):
        # Initially no failures
        counts = failure_tracker.get_failure_counts()
        assert counts == {}
        
        # Record some failures
        try:
            raise ValueError("Error 1")
        except ValueError as e:
            failure_tracker.record_failure(e, component="test_component")
        
        try:
            raise ValueError("Error 2")
        except ValueError as e:
            failure_tracker.record_failure(e, component="test_component")
        
        try:
            raise RuntimeError("Error 3")
        except RuntimeError as e:
            failure_tracker.record_failure(e, component="other_component")
        
        # Check counts
        counts = failure_tracker.get_failure_counts()
        assert counts["test_component"] == 2
        assert counts["other_component"] == 1


def test_reset(failure_tracker):
    """Test resetting the failure tracker."""
    with patch('observation.failure_tracker.logger'):
        # Record a failure
        try:
            raise ValueError("Test error")
        except ValueError as e:
            failure_tracker.record_failure(e, component="test_component")
        
        # Verify failure was recorded
        assert len(failure_tracker.failures) == 1
        assert failure_tracker.failure_counts["test_component"] == 1
        
        # Reset
        failure_tracker.reset()
        
        # Verify everything was cleared
        assert len(failure_tracker.failures) == 0
        assert len(failure_tracker.failure_counts) == 0


def test_alert_triggering(failure_tracker):
    """Test that alerts are triggered when threshold is exceeded."""
    with patch('observation.failure_tracker.logger') as mock_logger:
        # Record failures up to the threshold
        for i in range(2):  # threshold is 2
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                failure_tracker.record_failure(e, component="test_component")
        
        # Check that critical log was called (alert triggered)
        mock_logger.critical.assert_called()
        alert_call_args = mock_logger.critical.call_args[0][0]
        assert "ALERT" in alert_call_args
        assert "test_component" in alert_call_args
        assert "2" in alert_call_args  # failure count
        assert "threshold of 2" in alert_call_args