"""
MLflow integration for experiment tracking in the Enterprise LLM Trust Framework.
"""

import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os

from config.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)


class MLflowTracker:
    """
    Handles MLflow integration for experiment tracking and model registry.
    """
    
    def __init__(self, experiment_name: str = "llm_trust_framework"):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.client = None
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Sets up MLflow tracking."""
        try:
            # Set tracking URI from settings
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            
            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
            except mlflow.exceptions.MlflowException:
                # Experiment already exists
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
            
            # Set the experiment
            mlflow.set_experiment(experiment_id)
            
            # Initialize client
            self.client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
            
            logger.info(f"MLflow tracking initialized with URI: {settings.MLFLOW_TRACKING_URI}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow tracking: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to apply to the run
            
        Returns:
            Run ID of the started run
        """
        try:
            # Start MLflow run
            run = mlflow.start_run(run_name=run_name, tags=tags)
            run_id = run.info.run_id
            
            logger.info(f"Started MLflow run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.
        
        Args:
            status: Status of the run (FINISHED, FAILED, KILLED)
        """
        try:
            mlflow.end_run(status=status)
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current MLflow run.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            # Convert all values to strings as MLflow requires string parameters
            string_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(string_params)
            logger.debug(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Failed to log parameters to MLflow: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the current MLflow run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to the current MLflow run.
        
        Args:
            local_path: Path to the local artifact
            artifact_path: Optional path within the run's artifact directory
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact {local_path} to MLflow")
        except Exception as e:
            logger.error(f"Failed to log artifact to MLflow: {e}")
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """
        Log a dictionary as a JSON artifact to the current MLflow run.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Name of the artifact file
        """
        try:
            # Write dictionary to temporary JSON file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dictionary, f, indent=2)
                temp_path = f.name
            
            # Log the temporary file as an artifact
            self.log_artifact(temp_path, artifact_file)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            logger.debug(f"Logged dictionary as artifact {artifact_file} to MLflow")
        except Exception as e:
            logger.error(f"Failed to log dictionary to MLflow: {e}")
    
    def set_tag(self, key: str, value: str):
        """
        Set a tag on the current MLflow run.
        
        Args:
            key: Tag key
            value: Tag value
        """
        try:
            mlflow.set_tag(key, value)
            logger.debug(f"Set MLflow tag: {key}={value}")
        except Exception as e:
            logger.error(f"Failed to set MLflow tag: {e}")
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get information about the current experiment.
        
        Returns:
            Dictionary containing experiment information
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                return {
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "artifact_location": experiment.artifact_location,
                    "lifecycle_stage": experiment.lifecycle_stage
                }
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to get experiment info: {e}")
            return {}
    
    def search_runs(self, 
                   filter_string: Optional[str] = None,
                   run_view_type: int = 1,
                   max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Search for runs in the current experiment.
        
        Args:
            filter_string: Filter string for searching runs
            run_view_type: Type of runs to view (ACTIVE_ONLY, DELETED_ONLY, ALL)
            max_results: Maximum number of results to return
            
        Returns:
            List of run information dictionaries
        """
        try:
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                filter_string=filter_string,
                run_view_type=run_view_type,
                max_results=max_results
            )
            
            # Convert to list of dictionaries
            runs_list = runs.to_dict('records') if not runs.empty else []
            logger.info(f"Found {len(runs_list)} runs matching criteria")
            return runs_list
            
        except Exception as e:
            logger.error(f"Failed to search MLflow runs: {e}")
            return []


# Global MLflow tracker instance
mlflow_tracker = MLflowTracker()