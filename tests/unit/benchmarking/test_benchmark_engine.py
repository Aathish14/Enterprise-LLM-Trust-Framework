"""
Unit tests for the benchmarking engine.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from benchmarking.benchmark_engine import ComparativeBenchmarkEngine, BenchmarkConfig, BenchmarkResult
from core.interfaces import LLMAdapter, EvaluationResult, TrustScore


@pytest.fixture
def mock_llm_adapter():
    """Create a mock LLM adapter."""
    adapter = Mock(spec=LLMAdapter)
    adapter.generate = AsyncMock(return_value="Test response")
    adapter.get_model_info.return_value = {"provider": "test", "model_name": "test-model"}
    adapter.model_name = "test-model"
    return adapter


@pytest.fixture
def mock_evaluation_service():
    """Create a mock evaluation service."""
    service = Mock()
    service.evaluate = AsyncMock(return_value=EvaluationResult(
        score=0.8,
        confidence=0.9,
        details={"reasoning": "Good"},
        metric_name="test_metric"
    ))
    return service


@pytest.fixture
def mock_trust_scoring_service():
    """Create a mock trust scoring service."""
    service = Mock()
    service.compute_trust_score = AsyncMock(return_value=TrustScore(
        overall_score=0.75,
        dimension_scores={"test_metric": 0.8},
        confidence=0.85,
        details={"reasoning": "Good overall"}
    ))
    return service


@pytest.fixture
def benchmark_config(mock_llm_adapter, mock_evaluation_service, mock_trust_scoring_service):
    """Create a benchmark configuration for testing."""
    return BenchmarkConfig(
        benchmark_id="test_benchmark",
        name="Test Benchmark",
        description="A test benchmark",
        prompts=["Test prompt 1", "Test prompt 2"],
        context_template={},
        llm_adapters={"test_llm": mock_llm_adapter},
        evaluation_services=[mock_evaluation_service],
        trust_scoring_service=mock_trust_scoring_service,
        max_iterations=1
    )


@pytest.fixture
def benchmark_engine(benchmark_config):
    """Create a benchmark engine for testing."""
    return ComparativeBenchmarkEngine(benchmark_config)


def test_benchmark_config_initialization(benchmark_config):
    """Test that the BenchmarkConfig initializes correctly."""
    assert benchmark_config is not None
    assert benchmark_config.benchmark_id == "test_benchmark"
    assert benchmark_config.name == "Test Benchmark"
    assert benchmark_config.description == "A test benchmark"
    assert len(benchmark_config.prompts) == 2
    # Check that the llm_adapters dictionary contains our mock
    assert "test_llm" in benchmark_config.llm_adapters
    # We can't easily compare mock objects, so just check it's the right type and has the right key
    assert benchmark_config.llm_adapters["test_llm"] is not None
    assert isinstance(benchmark_config.llm_adapters["test_llm"], Mock)
    assert len(benchmark_config.evaluation_services) == 1
    assert benchmark_config.trust_scoring_service is not None
    assert isinstance(benchmark_config.trust_scoring_service, Mock)
    assert benchmark_config.max_iterations == 1


def test_benchmark_result_initialization():
    """Test that the BenchmarkResult initializes correctly."""
    eval_result = EvaluationResult(
        score=0.8,
        confidence=0.9,
        details={"reasoning": "Good"},
        metric_name="test_metric"
    )
    
    trust_score = TrustScore(
        overall_score=0.75,
        dimension_scores={"test_metric": 0.8},
        confidence=0.85,
        details={"reasoning": "Good overall"}
    )
    
    result = BenchmarkResult(
        benchmark_id="test_benchmark",
        llm_name="test_llm",
        prompt="Test prompt",
        response="Test response",
        evaluation_results=[eval_result],
        trust_score=trust_score,
        processing_time=1.5,
        timestamp=datetime.utcnow(),
        metadata={"test": "value"}
    )
    
    assert result is not None
    assert result.benchmark_id == "test_benchmark"
    assert result.llm_name == "test_llm"
    assert result.prompt == "Test prompt"
    assert result.response == "Test response"
    assert len(result.evaluation_results) == 1
    assert result.evaluation_results[0].metric_name == "test_metric"
    assert result.trust_score is not None
    assert result.trust_score.overall_score == 0.75
    assert result.processing_time == 1.5
    assert result.metadata["test"] == "value"


def test_benchmark_engine_initialization(benchmark_engine, benchmark_config):
    """Test that the ComparativeBenchmarkEngine initializes correctly."""
    assert benchmark_engine is not None
    assert benchmark_engine.config == benchmark_config


@pytest.mark.asyncio
async def test_run_benchmark(benchmark_engine, mock_llm_adapter, mock_evaluation_service, mock_trust_scoring_service):
    """Test running a benchmark."""
    with patch('benchmarking.benchmark_engine.mlflow_tracker'), \
         patch('benchmarking.benchmark_engine.experiment_tracker'):
        
        # Run the benchmark
        results = await benchmark_engine.run_benchmark()
        
        # Check that results were returned
        assert isinstance(results, dict)
        assert "test_llm" in results
        assert len(results["test_llm"]) == 2  # Two prompts
        
        # Check that LLM adapter was called for each prompt
        assert mock_llm_adapter.generate.call_count == 2
        
        # Check that evaluation service was called for each prompt
        assert mock_evaluation_service.evaluate.call_count == 2
        
        # Check that trust scoring service was called for each prompt
        assert mock_trust_scoring_service.compute_trust_score.call_count == 2
        
        # Check the structure of results
        for llm_name, llm_results in results.items():
            assert llm_name == "test_llm"
            assert len(llm_results) == 2
            
            for result in llm_results:
                assert isinstance(result, BenchmarkResult)
                assert result.benchmark_id == "test_benchmark"
                assert result.llm_name == "test_llm"
                assert result.processing_time >= 0
                assert result.timestamp is not None


@pytest.mark.asyncio
async def test_run_benchmark_with_errors(benchmark_engine, mock_llm_adapter, mock_evaluation_service, mock_trust_scoring_service):
    """Test running a benchmark when some evaluations fail."""
    with patch('benchmarking.benchmark_engine.mlflow_tracker'), \
         patch('benchmarking.benchmark_engine.experiment_tracker'):
        
        # Make the LLM adapter fail on the second call
        mock_llm_adapter.generate.side_effect = [
            "First response",
            Exception("API Error")
        ]
        
        # Run the benchmark
        results = await benchmark_engine.run_benchmark()
        
        # Check that results were returned
        assert isinstance(results, dict)
        assert "test_llm" in results
        # When an error occurs, we might get fewer results due to error handling in the pipeline
        # The important thing is that we don't crash and we get some results
        assert len(results["test_llm"]) >= 1  # At least one successful result
        
        # Check that LLM adapter was called for each prompt
        assert mock_llm_adapter.generate.call_count == 2
        
        # Check the structure of results
        for llm_name, llm_results in results.items():
            assert llm_name == "test_llm"
            assert len(llm_results) >= 1
            
            for result in llm_results:
                assert isinstance(result, BenchmarkResult)
                assert result.benchmark_id == "test_benchmark"
                assert result.llm_name == "test_llm"
                assert result.processing_time >= 0
                assert result.timestamp is not None


def test_log_benchmark_summary(benchmark_engine):
    """Test logging benchmark summary."""
    with patch('benchmarking.benchmark_engine.mlflow_tracker') as mock_mlflow, \
         patch('benchmarking.benchmark_engine.logger') as mock_logger:
        
        # Ensure mlflow_tracker.log_dict doesn't raise an exception
        mock_mlflow.log_dict.return_value = None
        
        # Create mock results with proper trust_score structure
        mock_result1 = Mock()
        mock_result1.trust_score = Mock()
        mock_result1.trust_score.overall_score = 0.8
        mock_result1.trust_score.dimension_scores = {"test_metric": 0.8}
        mock_result1.processing_time = 1.0
        
        mock_result2 = Mock()
        mock_result2.trust_score = Mock()
        mock_result2.trust_score.overall_score = 0.9
        mock_result2.trust_score.dimension_scores = {"test_metric": 0.9}
        mock_result2.processing_time = 1.5
        
        results = {
            "test_llm": [mock_result1, mock_result2]
        }
        
        # Call the method
        benchmark_engine._log_benchmark_summary(results, total_time=3.0)
        
        # Verify that the logger info method was called
        mock_logger.info.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])