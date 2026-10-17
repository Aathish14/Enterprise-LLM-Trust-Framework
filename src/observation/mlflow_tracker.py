"""
MLflow experiment tracking for the Enterprise LLM Trust Framework.

Provides :class:`MLflowTracker` with an always-absolute SQLite tracking URI
(resolving to the project-root ``mlflow.db``) and a cross-experiment
``get_runs()`` that reads every evaluation run regardless of experiment.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.exceptions
import mlflow.tracking
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

import pandas as pd

from config.logging import get_logger

logger = get_logger(__name__)


def get_absolute_sqlite_uri() -> str:
    """Return an absolute ``sqlite:///`` URI for the project-root ``mlflow.db``.

    Backslashes are converted to forward slashes so the URI is valid on
    Windows regardless of the current working directory.
    """
    return "sqlite:///" + os.path.abspath("mlflow.db").replace(chr(92), "/")


def _normalize_tracking_uri(uri: Optional[str]) -> str:
    """Upgrade a relative MLflow tracking URI to an absolute project-root URI."""
    if not uri:
        return get_absolute_sqlite_uri()
    uri = uri.strip()
    if uri.startswith("sqlite:///"):
        # Already absolute-sqlite; if the path portion is relative, make it absolute.
        path_part = uri[len("sqlite:///"):]
        if not os.path.isabs(path_part):
            return get_absolute_sqlite_uri()
        return uri
    if uri.startswith("sqlite://"):
        # sqlite://<host>/<path> or sqlite:relative -> normalize to project root.
        return get_absolute_sqlite_uri()
    # Non-sqlite URI: return as-is.
    return uri


class MLflowTracker:
    """
    Handles MLflow integration for experiment tracking and model registry.

    The tracker always connects to an absolute SQLite tracking URI pointing at
    the project-root ``mlflow.db``, so runs are never lost to a working-directory
    relative path.
    """

    def __init__(
        self,
        experiment_name: str = "llm_trust_framework",
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize the MLflow tracker.

        Args:
            experiment_name: Default MLflow experiment name.
            tracking_uri: Optional tracking URI. Relative sqlite URIs are
                auto-upgraded to the absolute project-root URI.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = _normalize_tracking_uri(tracking_uri)
        self.client: Optional[MlflowClient] = None
        self._initialized = False

    def _ensure_setup(self) -> None:
        """Lazily initialize MLflow tracking on first use."""
        if getattr(self, "_initialized", False):
            return
        try:
            self._setup_mlflow()
            self._initialized = True
        except Exception as e:  # noqa: BLE001 - degraded mode is intentional
            logger.warning(
                "MLflow setup deferred: %s. Running in degraded mode.", e
            )
            self._initialized = True  # Prevent retry loops

    def _setup_mlflow(self) -> None:
        """Sets up MLflow tracking against the absolute project-root URI."""
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get the default experiment
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(
                "Created MLflow experiment: %s (ID: %s)",
                self.experiment_name,
                experiment_id,
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id if experiment else "0"
            logger.info(
                "Using existing MLflow experiment: %s (ID: %s)",
                self.experiment_name,
                experiment_id,
            )

        mlflow.set_experiment(experiment_id)

        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        logger.info(
            "MLflow tracking initialized with URI: %s", self.tracking_uri
        )

    # ------------------------------------------------------------------ #
    # Run lifecycle
    # ------------------------------------------------------------------ #
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """Start a new MLflow run and return a context manager bound to self."""
        try:
            self._ensure_setup()

            if not nested:
                while mlflow.active_run():
                    active = mlflow.active_run()
                    logger.warning(
                        "Clearing active MLflow run %s before starting new run",
                        active.info.run_id,
                    )
                    try:
                        mlflow.end_run()
                    except Exception:  # noqa: BLE001
                        break

            run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
            logger.info("Started MLflow run: %s", run.info.run_id)
            return self._MLflowRunContext(run, self)
        except Exception as e:
            logger.error("Failed to start MLflow run: %s", e)
            raise

    class _MLflowRunContext:
        """Context manager for MLflow runs that provides access to the tracker."""

        def __init__(self, run, tracker: "MLflowTracker"):
            self.run = run
            self.tracker = tracker

        def __enter__(self):
            return self.tracker

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                # Import here to allow test patching of the tracker in the
                # experiment_tracking module.
                from src.observation.experiment_tracking import (  # noqa: WPS433
                    mlflow_tracker as exp_mlflow_tracker,
                )

                status = getattr(self, "_exit_status", "FINISHED")
                exp_mlflow_tracker.end_run(status=status)
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to end MLflow run: %s", e)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run (idempotent)."""
        try:
            active = mlflow.active_run()
            if active:
                mlflow.end_run(status=status)
                logger.info(
                    "Ended MLflow run %s with status %s",
                    active.info.run_id,
                    status,
                )
            else:
                logger.debug("No active MLflow run to end")
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to end MLflow run: %s", e)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def log_params(self, params: Dict[str, Any]) -> None:
        self._ensure_setup()
        try:
            string_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(string_params)
            logger.debug("Logged %d parameters to MLflow", len(params))
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to log parameters to MLflow: %s", e)

    def log_tags(self, tags: dict) -> None:
        """Log multiple tags to the active MLflow run."""
        self._ensure_setup()
        try:
            if tags:
                mlflow.set_tags(tags)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to log tags: %s", str(e))

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        self._ensure_setup()
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug("Logged %d metrics to MLflow", len(metrics))
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to log metrics to MLflow: %s", e)

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        self._ensure_setup()
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug("Logged artifact %s to MLflow", local_path)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to log artifact to MLflow: %s", e)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact to the current MLflow run."""
        self._ensure_setup()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(dictionary, f, indent=2)
                temp_path = f.name
            self.log_artifact(temp_path, artifact_file)
            os.unlink(temp_path)
            logger.debug(
                "Logged dictionary as artifact %s to MLflow", artifact_file
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to log dictionary to MLflow: %s", e)

    def set_tag(self, key: str, value: str) -> None:
        self._ensure_setup()
        try:
            mlflow.set_tag(key, value)
            logger.debug("Set MLflow tag: %s=%s", key, value)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to set MLflow tag: %s", e)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get_experiment_info(self) -> Dict[str, Any]:
        self._ensure_setup()
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                return {
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "artifact_location": experiment.artifact_location,
                    "lifecycle_stage": experiment.lifecycle_stage,
                }
            return {}
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to get experiment info: %s", e)
            return {}

    def search_runs(
        self,
        filter_string: Optional[str] = None,
        run_view_type: int = ViewType.ACTIVE_ONLY,
        max_results: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Search for runs in the current experiment."""
        self._ensure_setup()
        try:
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                filter_string=filter_string,
                run_view_type=run_view_type,
                max_results=max_results,
            )
            runs_list = runs.to_dict("records") if not runs.empty else []
            logger.info("Found %d runs matching criteria", len(runs_list))
            return runs_list
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to search MLflow runs: %s", e)
            return []

    def get_runs(
        self,
        experiment_names: Optional[List[str]] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Get runs as a :class:`pandas.DataFrame`.

        When ``experiment_names`` is empty or ``None``, all experiments are
        searched via ``mlflow.search_runs(search_all_experiments=True)`` so
        cross-experiment data is never missed. Returns a safe empty
        :class:`pandas.DataFrame` on any error.
        """
        self._ensure_setup()
        try:
            if experiment_names:
                runs = mlflow.search_runs(
                    experiment_names=list(experiment_names),
                    max_results=limit,
                )
            else:
                runs = mlflow.search_runs(
                    search_all_experiments=True,
                    max_results=limit,
                )
            if runs is None or runs.empty:
                return pd.DataFrame()
            return runs
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to get MLflow runs: %s", e)
            return pd.DataFrame()


# Global singleton instance (absolute project-root tracking URI).
mlflow_tracker = MLflowTracker()



