"""
MLflow integration for experiment tracking in the Enterprise LLM Trust Framework.
"""

import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os
import pandas as pd

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
        self._initialized = False
        
    def _ensure_setup(self):
        """Lazily initialize MLflow tracking on first use."""
        if getattr(self, '_initialized', False):
            return
        try:
            self._setup_mlflow()
            self._initialized = True
        except Exception as e:
            logger.warning(f"MLflow setup deferred: {e}. Running in degraded mode.")
            self._initialized = True  # Prevent retry loops

    def _setup_mlflow(self):
        """Sets up MLflow tracking."""
        try:
            # Set tracking URI from settings (use absolute path for consistency)
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri_absolute)
            
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
            self.client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri_absolute)
            
            logger.info(f"MLflow tracking initialized with URI: {settings.mlflow_tracking_uri_absolute}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow tracking: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None, nested: bool = False):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to apply to the run
            nested: Whether to start a nested run
            
        Returns:
            Context manager for the MLflow run
        """
        try:
            self._ensure_setup()
            
            # Check for and close any orphaned active runs before starting a new one
            # (unless nested=True, which allows nested runs within an active parent)
            if not nested:
                while mlflow.active_run():
                    try:
                        active = mlflow.active_run()
                        logger.warning(f"Clearing active MLflow run {active.info.run_id} before starting new run")
                        mlflow.end_run()
                    except Exception:
                        break
            
            # Start MLflow run
            run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
            
            logger.info(f"Started MLflow run: {run.info.run_id}")
            # Return a context manager that provides access to self for logging
            return self._MLflowRunContext(run, self)
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise

    class _MLflowRunContext:
        """Context manager for MLflow runs that provides access to the tracker for logging."""
        def __init__(self, run, tracker):
            self.run = run
            self.tracker = tracker
            
        def __enter__(self):
            return self.tracker
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                # Import here to allow test patching of the tracker in experiment_tracking module
                from src.observation.experiment_tracking import mlflow_tracker as exp_mlflow_tracker
                status = getattr(self, '_exit_status', 'FINISHED')
                exp_mlflow_tracker.end_run(status=status)
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run (idempotent).
        
        Args:
            status: Status of the run (FINISHED, FAILED, KILLED)
        """
        try:
            active = mlflow.active_run()
            if active:
                mlflow.end_run(status=status)
                logger.info(f"Ended MLflow run {active.info.run_id} with status {status}")
            else:
                logger.debug("No active MLflow run to end")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current MLflow run.
        
        Args:
            params: Dictionary of parameters to log
        """
        self._ensure_setup()
        try:
            # Convert all values to strings as MLflow requires string parameters
            string_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(string_params)
            logger.debug(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Failed to log parameters to MLflow: {e}")
    
    def log_tags(self, tags: dict) -> None:
        """Log multiple tags to the active MLflow run."""
        self._ensure_setup()
        try:
            if tags:
                mlflow.set_tags(tags)
        except Exception as e:
            logger.error(f"Failed to log tags: {str(e)}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the current MLflow run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        """
        self._ensure_setup()
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
        self._ensure_setup()
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
        self._ensure_setup()
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
        self._ensure_setup()
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
        self._ensure_setup()
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
        self._ensure_setup()
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

    def get_runs(self, experiment_id: Optional[str] = None) -> pd.DataFrame:
        """Get all runs across all experiments in the MLflow database."""
        try:
            self._ensure_setup()
            all_exps = mlflow.search_experiments()
            exp_ids = [e.experiment_id for e in all_exps] if all_exps else ["0", "1", "2"]
            runs = mlflow.search_runs(experiment_ids=exp_ids)
            if isinstance(runs, list):
                return pd.DataFrame(runs) if len(runs) > 0 else pd.DataFrame()
            return runs if isinstance(runs, pd.DataFrame) else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to get MLflow runs: {e}")
            return pd.DataFrame()

# Global singleton instance
mlflow_tracker = MLflowTracker()
