import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.fixture(autouse=True, scope="session")
def mock_env_credentials():
    """Inject mock API keys so tests never fail on missing environment variables."""
    os.environ["OPENAI_API_KEY"] = "sk-mock-key-12345"
    os.environ["NVIDIA_API_KEY"] = "nvapi-mock-key-12345"
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-mock-key-12345"
    os.environ["GOOGLE_API_KEY"] = "mock-google-key-12345"

@pytest.fixture(autouse=True)
def mock_llm_network_calls(monkeypatch):
    """Mock external LLM network calls for 100% offline unit test execution."""
    async def mock_ainvoke(*args, **kwargs):
        mock_msg = MagicMock()
        mock_msg.content = "SCORE: 0.85\nCONFIDENCE: 0.90\nREASONING: Factually accurate."
        return mock_msg

    try:
        from langchain_openai import ChatOpenAI
        monkeypatch.setattr(ChatOpenAI, "ainvoke", mock_ainvoke)
    except Exception:
        pass

@pytest.fixture(autouse=True, scope="session")
def mock_mlflow_tracker():
    """Globally mock MLflow tracker to prevent DB initialization in unit tests."""
    with patch("src.observation.mlflow_integration.mlflow_tracker") as mock_tracker, \
         patch("src.observation.experiment_tracking.mlflow_tracker") as mock_tracker2, \
         patch("src.benchmarking.benchmark_engine.mlflow_tracker") as mock_tracker3, \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.create_experiment"), \
         patch("mlflow.get_experiment_by_name"), \
         patch("mlflow.start_run"), \
         patch("mlflow.end_run"), \
         patch("mlflow.log_params"), \
         patch("mlflow.log_metrics"), \
         patch("mlflow.log_artifact"), \
         patch("mlflow.set_tag"), \
         patch("mlflow.search_runs") as mock_search_runs:
        
        # Configure mock tracker
        mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=mock_tracker)
        mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_tracker.get_runs.return_value = []
        mock_tracker.experiment_name = "llm_trust_framework"
        
        mock_tracker2.start_run.return_value.__enter__ = MagicMock(return_value=mock_tracker2)
        mock_tracker2.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_tracker2.get_runs.return_value = []
        
        mock_tracker3.start_run.return_value.__enter__ = MagicMock(return_value=mock_tracker3)
        mock_tracker3.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_tracker3.get_runs.return_value = []
        
        # Mock search_runs to return empty DataFrame
        import pandas as pd
        mock_search_runs.return_value = pd.DataFrame()
        
        yield mock_tracker

@pytest.fixture
def approx():
    """Provide pytest.approx for floating-point assertions."""
    return pytest.approx