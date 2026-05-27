"""
Experiment tracking capabilities for the Enterprise LLM Trust Framework.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

from .mlflow_integration import mlflow_tracker
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: str
    name: str
    description: str
    tags: Dict[str, str]
    parameters: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class ExperimentRun:
    """Represents a single experiment run."""
    run_id: str
    experiment_id: str
    name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    tags: Dict[str, str]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "RUNNING"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


class ExperimentTracker:
    """
    Handles experiment tracking for LLM evaluation experiments.
    """
    
    def __init__(self):
        """Initialize the experiment tracker."""
        self.active_runs: Dict[str, ExperimentRun] = {}
        logger.info("ExperimentTracker initialized")
    
    def create_experiment(self, 
                         name: str, 
                         description: str = "",
                         tags: Optional[Dict[str, str]] = None,
                         parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new experiment.
        
        Args:
            name: Name of the experiment
            description: Description of the experiment
            tags: Optional tags for the experiment
            parameters: Optional parameters for the experiment
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            tags=tags or {},
            parameters=parameters or {},
            created_at=datetime.utcnow()
        )
        
        # Log experiment creation as an MLflow tag/run
        try:
            # Start a run to log experiment metadata
            with mlflow_tracker.start_run(run_name=f"experiment_creation_{name}", 
                                        tags={"experiment_id": experiment_id, "type": "experiment_setup"}):
                mlflow_tracker.log_params({
                    "experiment_name": name,
                    "experiment_description": description,
                    **config.parameters
                })
                mlflow_tracker.log_tags(config.tags)
                mlflow_tracker.log_dict(config.to_dict(), "experiment_config.json")
                
            logger.info(f"Created experiment: {name} (ID: {experiment_id})")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    def start_run(self, 
                 experiment_id: str,
                 run_name: Optional[str] = None,
                 parameters: Optional[Dict[str, Any]] = None,
                 tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new run within an experiment.
        
        Args:
            experiment_id: ID of the experiment
            run_name: Optional name for the run
            parameters: Optional parameters for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())
        
        # Prepare tags
        run_tags = {
            "experiment_id": experiment_id,
            "run_id": run_id
        }
        if tags:
            run_tags.update(tags)
        
        # Start MLflow run
        mlflow_run_id = mlflow_tracker.start_run(
            run_name=run_name or f"run_{run_id[:8]}",
            tags=run_tags
        )
        
        # Create experiment run object
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            name=run_name or f"run_{run_id[:8]}",
            parameters=parameters or {},
            metrics={},
            artifacts=[],
            tags=run_tags,
            start_time=datetime.utcnow()
        )
        
        # Store active run
        self.active_runs[run_id] = run
        
        # Log parameters to MLflow
        if parameters:
            mlflow_tracker.log_params(parameters)
        
        logger.info(f"Started run {run_id} in experiment {experiment_id}")
        return run_id
    
    def end_run(self, run_id: str, status: str = "FINISHED"):
        """
        End an experiment run.
        
        Args:
            run_id: ID of the run to end
            status: Status of the run (FINISHED, FAILED, KILLED)
        """
        if run_id not in self.active_runs:
            logger.warning(f"Attempted to end unknown run: {run_id}")
            return
        
        run = self.active_runs[run_id]
        run.end_time = datetime.utcnow()
        run.status = status
        
        # End MLflow run
        mlflow_tracker.end_run(status=status)
        
        # Remove from active runs
        del self.active_runs[run_id]
        
        logger.info(f"Ended run {run_id} with status: {status}")
    
    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        """
        Log parameters to a run.
        
        Args:
            run_id: ID of the run
            parameters: Parameters to log
        """
        if run_id not in self.active_runs:
            logger.warning(f"Attempted to log parameters to unknown run: {run_id}")
            return
        
        run = self.active_runs[run_id]
        run.parameters.update(parameters)
        
        # Log to MLflow
        mlflow_tracker.log_params(parameters)
        
        logger.debug(f"Logged {len(parameters)} parameters to run {run_id}")
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to a run.
        
        Args:
            run_id: ID of the run
            metrics: Metrics to log
            step: Optional step number
        """
        if run_id not in self.active_runs:
            logger.warning(f"Attempted to log metrics to unknown run: {run_id}")
            return
        
        run = self.active_runs[run_id]
        run.metrics.update(metrics)
        
        # Log to MLflow
        mlflow_tracker.log_metrics(metrics, step=step)
        
        logger.debug(f"Logged {len(metrics)} metrics to run {run_id}")
    
    def log_artifact(self, run_id: str, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to a run.
        
        Args:
            run_id: ID of the run
            local_path: Local path to the artifact
            artifact_path: Optional path within the run's artifact directory
        """
        if run_id not in self.active_runs:
            logger.warning(f"Attempted to log artifact to unknown run: {run_id}")
            return
        
        run = self.active_runs[run_id]
        
        # Log to MLflow
        mlflow_tracker.log_artifact(local_path, artifact_path)
        
        # Track artifact
        artifact_name = artifact_path or local_path.split("/")[-1]
        run.artifacts.append(artifact_name)
        
        logger.debug(f"Logged artifact {local_path} to run {run_id}")
    
    def set_tag(self, run_id: str, key: str, value: str):
        """
        Set a tag on a run.
        
        Args:
            run_id: ID of the run
            key: Tag key
            value: Tag value
        """
        if run_id not in self.active_runs:
            logger.warning(f"Attempted to set tag on unknown run: {run_id}")
            return
        
        run = self.active_runs[run_id]
        run.tags[key] = value
        
        # Log to MLflow
        mlflow_tracker.set_tag(key, value)
        
        logger.debug(f"Set tag {key}={value} on run {run_id}")
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """
        Get information about a run.
        
        Args:
            run_id: ID of the run
            
        Returns:
            ExperimentRun object or None if not found
        """
        return self.active_runs.get(run_id)
    
    def list_active_runs(self) -> List[ExperimentRun]:
        """
        List all active runs.
        
        Returns:
            List of active ExperimentRun objects
        """
        return list(self.active_runs.values())
    
    def finish_run(self, run_id: str):
        """
        Finish a run successfully.
        
        Args:
            run_id: ID of the run to finish
        """
        self.end_run(run_id, status="FINISHED")


# Global experiment tracker instance
experiment_tracker = ExperimentTracker()