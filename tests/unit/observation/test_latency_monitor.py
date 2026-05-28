"""
Unit tests for the latency monitoring component.
"""

import pytest
import asyncio
import sys
import os
import time
from unittest.mock import Mock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from observation.latency_monitor import LatencyMonitor


@pytest.fixture
def latency_monitor():
    """Create a LatencyMonitor instance for testing."""
    return LatencyMonitor("test_operation")


def test_latency_monitor_initialization(latency_monitor):
    """Test that the LatencyMonitor initializes correctly."""
    assert latency_monitor is not None
    assert latency_monitor.operation_name == "test_operation"
    assert isinstance(latency_monitor.latencies, list)
    assert len(latency_monitor.latencies) == 0


def test_record_latency(latency_monitor):
    """Test recording latency measurements."""
    # Record a few latencies
    latency_monitor.record_latency(0.1)
    latency_monitor.record_latency(0.2)
    latency_monitor.record_latency(0.15, success=False)
    
    assert len(latency_monitor.latencies) == 3
    assert latency_monitor.latencies == [0.1, 0.2, 0.15]


def test_time_call_sync(latency_monitor):
    """Test timing a synchronous function."""
    @latency_monitor.time_call
    def slow_function():
        time.sleep(0.01)  # Sleep for 10ms
        return "result"
    
    result = slow_function()
    
    assert result == "result"
    assert len(latency_monitor.latencies) == 1
    # Should be at least the sleep time (10ms)
    assert latency_monitor.latencies[0] >= 0.009  # Allow for small timing variations
    # Upper bound is less critical, just verify it recorded something reasonable
    assert latency_monitor.latencies[0] <= 0.1  # Should be much less than 100ms


@pytest.mark.asyncio
async def test_time_call_async(latency_monitor):
    """Test timing an asynchronous function."""
    @latency_monitor.time_call
    async def slow_async_function():
        await asyncio.sleep(0.01)  # Sleep for 10ms
        return "async_result"
    
    result = await slow_async_function()
    
    assert result == "async_result"
    assert len(latency_monitor.latencies) == 1
    # Should be at least the sleep time (10ms)
    assert latency_monitor.latencies[0] >= 0.009  # Allow for small timing variations
    # Upper bound is less critical, just verify it recorded something reasonable
    assert latency_monitor.latencies[0] <= 0.1  # Should be much less than 100ms


def test_time_call_with_exception(latency_monitor):
    """Test timing a function that raises an exception."""
    @latency_monitor.time_call
    def failing_function():
        time.sleep(0.005)  # Sleep for 5ms
        raise ValueError("Test error")
    
    with pytest.raises(ValueError, match="Test error"):
        failing_function()
    
    # Should still record latency even when function fails
    assert len(latency_monitor.latencies) == 1
    # Should be at least the sleep time (5ms)
    assert latency_monitor.latencies[0] >= 0.004  # Allow for small timing variations
    # Upper bound is less critical, just verify it recorded something reasonable
    assert latency_monitor.latencies[0] <= 0.05  # Should be much less than 50ms


def test_get_statistics(latency_monitor):
    """Test getting latency statistics."""
    # No latencies recorded yet
    stats = latency_monitor.get_statistics()
    assert stats == {}
    
    # Record some latencies
    latency_monitor.record_latency(0.1)
    latency_monitor.record_latency(0.2)
    latency_monitor.record_latency(0.15)
    latency_monitor.record_latency(0.3)
    latency_monitor.record_latency(0.05)
    
    stats = latency_monitor.get_statistics()
    
    assert isinstance(stats, dict)
    assert stats["count"] == 5
    assert abs(stats["mean"] - 0.16) < 0.001  # (0.1+0.2+0.15+0.3+0.05)/5 = 0.16
    assert stats["min"] == 0.05
    assert stats["max"] == 0.3
    assert stats["median"] == 0.15  # Middle value when sorted: [0.05, 0.1, 0.15, 0.2, 0.3]
    # For p95 with 5 values, it should be the second highest (0.3) or close to it
    assert stats["p95"] >= 0.25  # Adjusted expectation
    assert stats["p99"] >= 0.25  # Adjusted expectation
    assert stats["std"] >= 0  # Standard deviation should be non-negative


def test_reset(latency_monitor):
    """Test resetting the latency history."""
    # Record some latencies
    latency_monitor.record_latency(0.1)
    latency_monitor.record_latency(0.2)
    
    assert len(latency_monitor.latencies) == 2
    
    # Reset
    latency_monitor.reset()
    
    assert len(latency_monitor.latencies) == 0
    
    # Statistics should be empty after reset
    stats = latency_monitor.get_statistics()
    assert stats == {}


def test_global_latency_monitors():
    """Test that global latency monitors exist."""
    from observation.latency_monitor import llm_call_latency, evaluation_latency, pipeline_latency
    
    assert llm_call_latency is not None
    assert llm_call_latency.operation_name == "llm_call"
    
    assert evaluation_latency is not None
    assert evaluation_latency.operation_name == "evaluation"
    
    assert pipeline_latency is not None
    assert pipeline_latency.operation_name == "evaluation_pipeline"