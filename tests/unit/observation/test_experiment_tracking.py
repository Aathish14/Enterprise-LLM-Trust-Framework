"""
Unit tests for the experiment tracking component.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from observation.experiment_tracking import ExperimentTracker, ExperimentConfig, ExperimentRun


@pytest.fixture
def experiment_tracker():
    """Create an ExperimentTracker instance for testing."""
    return ExperimentTracker()


def test_experiment_tracker_initialization(experiment_tracker):
    """Test that the ExperimentTracker initializes correctly."""
    assert experiment_tracker is not None
    assert isinstance(experiment_tracker.active_runs, dict)
    assert len(experiment_tracker.active_runs) == 0


def test_create_experiment(experiment_tracker):
    """Test creating a new experiment."""
    with patch('observation.experiment_tracking.mlflow_tracker'):
        experiment_id = experiment_tracker.create_experiment(
            name="test_experiment",
            description="A test experiment",
            tags={"env": "test"},
            parameters={"learning_rate": 0.01}
        )
        
        assert isinstance(experiment_id, str)
        assert len(experiment_id) > 0
        
        # Verify MLflow calls were made
        from observation.experiment_tracking import mlflow_tracker
        mlflow_tracker.start_run.assert_called_once()
        mlflow_tracker.log_params.assert_called_once()
        mlflow_tracker.log_tags.assert_called_once()
        mlflow_tracker.log_dict.assert_called_once()
        # Note: mlflow_tracker.end_run is called automatically by the context manager


def test_start_run(experiment_tracker):
    """Test starting a new run within an experiment."""
    with patch('observation.experiment_tracking.mlflow_tracker'):
        # Mock the MLflow run ID return value
        from observation.experiment_tracking import mlflow_tracker
        mlflow_tracker.start_run.return_value = "mlflow_run_123"
        
        run_id = experiment_tracker.start_run(
            experiment_id="exp_123",
            run_name="test_run",
            parameters={"batch_size": 32},
            tags={"version": "1.0"}
        )
        
        assert isinstance(run_id, str)
        assert len(run_id) > 0
        assert run_id in experiment_tracker.active_runs
        
        run = experiment_tracker.active_runs[run_id]
        assert run.run_id == run_id
        assert run.experiment_id == "exp_123"
        assert run.name == "test_run"
        assert run.parameters == {"batch_size": 32}
        assert run.tags["experiment_id"] == "exp_123"
        assert run.tags["run_id"] == run_id
        assert run.tags["version"] == "1.0"
        assert isinstance(run.start_time, datetime)
        
        # Verify MLflow calls
        mlflow_tracker.start_run.assert_called_once()
        mlflow_tracker.log_params.assert_called_once_with({"batch_size": 32})


def test_end_run(experiment_tracker):
    """Test ending an experiment run."""
    with patch('observation.experiment_tracking.mlflow_tracker'):
        # First create a run
        run_id = experiment_tracker.start_run(
            experiment_id="exp_123",
            run_name="test_run"
        )
        
        # Verify it's active
        assert run_id in experiment_tracker.active_runs
        
        # End the run
        experiment_tracker.end_run(run_id, status="FINISHED")
        
        # Verify it's no longer active
        assert run_id not in experiment_tracker.active_runs
        
        # Verify MLflow calls
        from observation.experiment_tracking import mlflow_tracker
        mlflow_tracker.end_run.assert_called_once_with(status="FINISHED")


def test_log_parameters(experiment_tracker):
    """Test logging parameters to a run."""
    with patch('observation.experiment_tracking.mlflow_tracker'):
        # First create a run
        run_id = experiment_tracker.start_run(
            experiment_id="exp_123",
            run_name="test_run"
        )
        
        # Log parameters
        experiment_tracker.log_parameters(run_id, {"epochs": 10, "lr": 0.001})
        
        # Verify the run's parameters were updated
        run = experiment_tracker.active_runs[run_id]
        assert run.parameters == {"epochs": 10, "lr": 0.001}
        
        # Verify MLflow call
        from observation.experiment_tracking import mlflow_tracker
        mlflow_tracker.log_params.assert_called_once_with({"epochs": 10, "lr": 0.001})


def test_log_metrics(experiment_tracker):
    """Test logging metrics to a run."""
    with patch('observation.experiment_tracking.mlflow_tracker'):
        # First create a run
        run_id = experiment_tracker.start_run(
            experiment_id="exp_123",
            run_name="test_run"
        )
        
        # Log metrics
        experiment_tracker.log_metrics(run_id, {"accuracy": 0.95, "loss": 0.05}, step=1)
        
        # Verify the run's metrics were updated
        run = experiment_tracker.active_runs[run_id]
        assert run.metrics == {"accuracy": 0.95, "loss": 0.05}
        
        # Verify MLflow call
        from observation.experiment_tracking import mlflow_tracker
        mlflow_tracker.log_metrics.assert_called_once_with({"accuracy": 0.95, "loss": 0.05}, step=1)


def test_log_artifact(experiment_tracker):
    """Test logging an artifact to a run."""
    with patch('observation.experiment_tracking.mlflow_tracker'):
        # First create a run
        run_id = experiment_tracker.start_run(
            experiment_id="exp_123",
            run_name="test_run"
        )
        
        # Log artifact
        experiment_tracker.log_artifact(run_id, "/path/to/model.pkl", "models/model.pkl")
        
        # Verify the run's artifacts were updated
        run = experiment_tracker.active_runs[run_id]
        assert "models/model.pkl" in run.artifacts
        
        # Verify MLflow call
        from observation.experiment_tracking import mlflow_tracker
        mlflow_tracker.log_artifact.assert_called_once_with("/path/to/model.pkl", "models/model.pkl")


def test_set_tag(experiment_tracker):
    """Test setting a tag on a run."""
    with patch('observation.experiment_tracking.mlflow_tracker'):
        # First create a run
        run_id = experiment_tracker.start_run(
            experiment_id="exp_123",
            run_name="test_run"
        )
        
        # Set tag
        experiment_tracker.set_tag(run_id, "environment", "production")
        
        # Verify the run's tags were updated
        run = experiment_tracker.active_runs[run_id]
        assert run.tags["environment"] == "production"
        
        # Verify MLflow call
        from observation.experiment_tracking import mlflow_tracker
        mlflow_tracker.set_tag.assert_called_once_with("environment", "production")


def test_get_run(experiment_tracker):
    """Test getting information about a run."""
    with patch('observation.experiment_tracking.mlflow_tracker'):
        # First create a run
        run_id = experiment_tracker.start_run(
            experiment_id="exp_123",
            run_name="test_run"
        )
        
        # Get the run
        run = experiment_tracker.get_run(run_id)
        
        assert run is not None
        assert run.run_id == run_id
        assert run.experiment_id == "exp_123"
        assert run.name == "test_run"
        
        # Test getting non-existent run
        non_existent_run = experiment_tracker.get_run("non_existent_id")
        assert non_existent_run is None


def test_list_active_runs(experiment_tracker):
    """Test listing all active runs."""
    with patch('observation.experiment_tracking.mlflow_tracker'):
        # Initially no active runs
        assert len(experiment_tracker.list_active_runs()) == 0
        
        # Create a few runs
        run_id_1 = experiment_tracker.start_run(
            experiment_id="exp_123",
            run_name="test_run_1"
        )
        
        run_id_2 = experiment_tracker.start_run(
            experiment_id="exp_123",
            run_name="test_run_2"
        )
        
        # List active runs
        active_runs = experiment_tracker.list_active_runs()
        assert len(active_runs) == 2
        
        run_ids = [run.run_id for run in active_runs]
        assert run_id_1 in run_ids
        assert run_id_2 in run_ids
        
        # End one run
        experiment_tracker.end_run(run_id_1)
        
        # List active runs again
        active_runs = experiment_tracker.list_active_runs()
        assert len(active_runs) == 1
        assert active_runs[0].run_id == run_id_2


def test_finish_run(experiment_tracker):
    """Test finishing a run successfully."""
    with patch('observation.experiment_tracking.mlflow_tracker'):
        # First create a run
        run_id = experiment_tracker.start_run(
            experiment_id="exp_123",
            run_name="test_run"
        )
        
        # Verify it's active
        assert run_id in experiment_tracker.active_runs
        
        # Finish the run
        experiment_tracker.finish_run(run_id)
        
        # Verify it's no longer active
        assert run_id not in experiment_tracker.active_runs
        
        # Verify MLflow calls
        from observation.experiment_tracking import mlflow_tracker
        mlflow_tracker.end_run.assert_called_once_with(status="FINISHED")