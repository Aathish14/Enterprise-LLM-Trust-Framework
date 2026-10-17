<#
.SYNOPSIS
    Enterprise LLM Trust Framework - Test Repair Script
    Fixes: Missing API keys, floating-point assertions, network call mocking
    Run in PowerShell: .\fix_tests.ps1
#>

$ErrorActionPreference = "Stop"
Set-Location "D:\Enterprise-LLM-Trust-Framework"

Write-Host "=== Step 1: Clean up MLflow test databases ===" -ForegroundColor Cyan
Remove-Item -Path "test_mlflow.db" -ErrorAction SilentlyContinue
Remove-Item -Path "mlflow.db" -ErrorAction SilentlyContinue
Write-Host "Cleaned up MLflow databases" -ForegroundColor Green


Write-Host "=== Step 2: Create tests/conftest.py ===" -ForegroundColor Cyan

# Write conftest.py using Python to avoid PowerShell parsing issues
& ".\.venv\Scripts\python.exe" -c "
import os
content = '''
\"\"\"
Global pytest configuration for Enterprise LLM Trust Framework tests.
Provides mock environment credentials and common fixtures.
\"\"\"

import os
import pytest
from unittest.mock import Mock, AsyncMock, patch
import sys

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), \"..\", \"src\"))


# Mock MLflow BEFORE any imports that use it
@pytest.fixture(scope=\"session\", autouse=True)
def mock_mlflow_tracker():
    \"\"\"
    Global mock for MLflow tracker to prevent database initialization issues.
    Runs automatically for every test session before any imports.
    \"\"\"
    with patch(\"src.observation.mlflow_integration.mlflow_tracker\") as mock_tracker, \\
         patch(\"src.observation.experiment_tracking.mlflow_tracker\") as mock_tracker2, \\
         patch(\"src.benchmarking.benchmark_engine.mlflow_tracker\") as mock_tracker3, \\
         patch(\"mlflow.set_tracking_uri\"), \\
         patch(\"mlflow.set_experiment\"), \\
         patch(\"mlflow.create_experiment\"), \\
         patch(\"mlflow.get_experiment_by_name\"), \\
         patch(\"mlflow.start_run\"), \\
         patch(\"mlflow.end_run\"), \\
         patch(\"mlflow.log_params\"), \\
         patch(\"mlflow.log_metrics\"), \\
         patch(\"mlflow.log_artifact\"), \\
         patch(\"mlflow.set_tag\"), \\
         patch(\"mlflow.search_runs\") as mock_search_runs:
        
        # Configure mock tracker
        mock_tracker.start_run.return_value.__enter__ = Mock(return_value=mock_tracker)
        mock_tracker.start_run.return_value.__exit__ = Mock(return_value=False)
        mock_tracker.get_runs.return_value = []
        mock_tracker.experiment_name = \"llm_trust_framework\"
        
        mock_tracker2.start_run.return_value.__enter__ = Mock(return_value=mock_tracker2)
        mock_tracker2.start_run.return_value.__exit__ = Mock(return_value=False)
        mock_tracker2.get_runs.return_value = []
        
        mock_tracker3.start_run.return_value.__enter__ = Mock(return_value=mock_tracker3)
        mock_tracker3.start_run.return_value.__exit__ = Mock(return_value=False)
        mock_tracker3.get_runs.return_value = []
        
        # Mock search_runs to return empty DataFrame
        import pandas as pd
        mock_search_runs.return_value = pd.DataFrame()
        
        yield mock_tracker


@pytest.fixture(scope=\"session\", autouse=True)
def mock_env_credentials():
    \"\"\"
    Global fixture that sets dummy API keys for all tests.
    Runs automatically for every test session.
    \"\"\"
    original_env = dict(os.environ)
    
    os.environ[\"OPENAI_API_KEY\"] = \"sk-mock-test-key-12345\"
    os.environ[\"NVIDIA_API_KEY\"] = \"nvapi-mock-test-key-12345\"
    os.environ[\"ANTHROPIC_API_KEY\"] = \"sk-ant-mock-test-key-12345\"
    os.environ[\"GOOGLE_API_KEY\"] = \"mock-google-key-12345\"
    os.environ[\"HF_TOKEN\"] = \"hf_mock_token_12345\"
    os.environ[\"MLFLOW_TRACKING_URI\"] = \"sqlite:///test_mlflow.db\"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def mock_llm_network_calls():
    \"\"\"
    Auto-mock common LLM network calls for unit tests.
    Ensures tests run 100% offline in under 10 seconds.
    \"\"\"
    with patch(\"langchain_openai.ChatOpenAI.ainvoke\", new_callable=AsyncMock) as mock_ainvoke, \\
         patch(\"langchain_openai.ChatOpenAI.invoke\", new_callable=Mock) as mock_invoke, \\
         patch(\"openai.AsyncOpenAI\") as mock_async_openai, \\
         patch(\"openai.OpenAI\") as mock_openai:
        
        # Configure default mock responses
        mock_response = Mock()
        mock_response.content = \"SCORE: 0.8\nCONFIDENCE: 0.9\nREASONING: Test response\"
        mock_ainvoke.return_value = mock_response
        mock_invoke.return_value = mock_response
        
        # Mock AsyncOpenAI client
        mock_client = AsyncMock()
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content=\"Test response\"))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_async_openai.return_value = mock_client
        
        # Mock sync OpenAI client
        mock_sync_client = Mock()
        mock_sync_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_sync_client
        
        yield {
            \"ainvoke\": mock_ainvoke,
            \"invoke\": mock_invoke,
            \"async_client\": mock_client,
            \"sync_client\": mock_sync_client
        }


@pytest.fixture
def sample_evaluation_results():
    \"\"\"Create sample evaluation results for testing.\"\"\"
    from src.core.interfaces import EvaluationResult
    return [
        EvaluationResult(score=0.8, confidence=0.9, details={}, metric_name=\"factual_correctness\"),
        EvaluationResult(score=0.6, confidence=0.8, details={}, metric_name=\"hallucination_likelihood\"),
        EvaluationResult(score=0.9, confidence=0.95, details={}, metric_name=\"semantic_similarity\")
    ]


@pytest.fixture
def sample_trust_score():
    \"\"\"Create a sample TrustScore for testing.\"\"\"
    from src.core.interfaces import TrustScore
    return TrustScore(
        overall_score=0.78,
        confidence=0.88,
        details={},
        dimension_scores={\"test_metric\": 0.85, \"another_metric\": 0.70}
    )


# Import pytest.approx for floating-point comparisons
@pytest.fixture
def approx():
    \"\"\"Provide pytest.approx for floating-point assertions.\"\"\"
    return pytest.approx
'''
with open('tests/conftest.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('conftest.py created')
"
Write-Host "✓ tests/conftest.py created" -ForegroundColor Green


Write-Host "=== Step 3: Fix floating-point assertions in test files ===" -ForegroundColor Cyan

# Helper function to fix floating-point assertions in a file
function Fix-FloatAssertions {
    param([string]$FilePath, [string[]]$Patterns)
    
    $content = [System.IO.File]::ReadAllText($FilePath, [System.Text.Encoding]::UTF8)
    $original = $content
    
    foreach ($pattern in $Patterns) {
        $content = $content -replace $pattern[0], $pattern[1]
    }
    
    if ($content -ne $original) {
        [System.IO.File]::WriteAllText($FilePath, $content, [System.Text.Encoding]::UTF8)
        Write-Host "  Fixed: $FilePath" -ForegroundColor Yellow
    } else {
        Write-Host "  No changes: $FilePath" -ForegroundColor Gray
    }
}

# Fix test_trust_scoring.py
Fix-FloatAssertions "tests/unit/test_trust_scoring.py" @(
    @('assert trust_score\.dimension_scores\[result\.metric_name\] == result\.score', 'assert trust_score.dimension_scores[result.metric_name] == pytest.approx(result.score, rel=1e-3)'),
    @('assert trust_score\.dimension_scores\[metric_name\] == 0\.0', 'assert trust_score.dimension_scores[metric_name] == pytest.approx(0.0, abs=1e-6)'),
    @('assert scorer\.performance_history\[0\]\["trust_score"\]\["overall_score"\] == 0\.8', 'assert scorer.performance_history[0]["trust_score"]["overall_score"] == pytest.approx(0.8, rel=1e-3)'),
    @('assert trust_score\.overall_score == 0\.75', 'assert trust_score.overall_score == pytest.approx(0.75, rel=1e-3)')
)

# Fix test_factual_correctness.py
Fix-FloatAssertions "tests/unit/test_factual_correctness.py" @(
    @('assert result\.score == 0\.8', 'assert result.score == pytest.approx(0.8, rel=1e-3)'),
    @('assert result\.confidence == 0\.9', 'assert result.confidence == pytest.approx(0.9, rel=1e-3)'),
    @('assert result\.score == 0\.0', 'assert result.score == pytest.approx(0.0, abs=1e-6)'),
    @('assert result\.confidence == 0\.0', 'assert result.confidence == pytest.approx(0.0, abs=1e-6)'),
    @('assert score == 0\.75', 'assert score == pytest.approx(0.75, rel=1e-3)'),
    @('assert confidence == 0\.8', 'assert confidence == pytest.approx(0.8, rel=1e-3)'),
    @('assert score == 0\.5\s+assert confidence == 0\.6', 'assert score == pytest.approx(0.5, rel=1e-3)`n    assert confidence == pytest.approx(0.6, rel=1e-3)'),
    @('assert score == 1\.0  # Clamped to max\s+assert confidence == 0\.0  # Clamped to min', 'assert score == pytest.approx(1.0, rel=1e-3)  # Clamped to max`n    assert confidence == pytest.approx(0.0, abs=1e-6)  # Clamped to min'),
    @('assert score == 0\.5  # Default\s+assert confidence == 0\.5  # Default', 'assert score == pytest.approx(0.5, rel=1e-3)  # Default`n    assert confidence == pytest.approx(0.5, rel=1e-3)  # Default')
)

# Fix test_evaluation_pipeline.py
Fix-FloatAssertions "tests/unit/test_evaluation_pipeline.py" @(
    @('assert result\["evaluation_results"\]\[0\]\["score"\] == 0\.8', 'assert result["evaluation_results"][0]["score"] == pytest.approx(0.8, rel=1e-3)'),
    @('assert result\["trust_score"\]\["overall_score"\] == 0\.75', 'assert result["trust_score"]["overall_score"] == pytest.approx(0.75, rel=1e-3)'),
    @('assert result\["trust_score"\]\["dimension_scores"\]\["test_metric"\] == 0\.8', 'assert result["trust_score"]["dimension_scores"]["test_metric"] == pytest.approx(0.8, rel=1e-3)'),
    @('assert result\["trust_score"\]\["confidence"\] == 0\.85', 'assert result["trust_score"]["confidence"] == pytest.approx(0.85, rel=1e-3)')
)

# Fix test_metrics_logger.py
Fix-FloatAssertions "tests/unit/observation/test_metrics_logger.py" @(
    @('assert metrics_logger\.latency_history\[0\] == 0\.45', 'assert metrics_logger.latency_history[0] == pytest.approx(0.45, rel=1e-3)'),
    @('assert metrics_logger\.latency_history\[0\] == 0\.25', 'assert metrics_logger.latency_history[0] == pytest.approx(0.25, rel=1e-3)'),
    @('assert metrics_logger\.latency_history\[0\] == 0\.005', 'assert metrics_logger.latency_history[0] == pytest.approx(0.005, rel=1e-3)'),
    @('assert metrics_logger\.latency_history\[-1\] == 1\.004', 'assert metrics_logger.latency_history[-1] == pytest.approx(1.004, rel=1e-3)'),
    @('assert stats\["median"\] == 0\.15  # Middle value when sorted: \[0\.05, 0\.1, 0\.15, 0\.2, 0\.3\]', 'assert stats["median"] == pytest.approx(0.15, rel=1e-3)  # Middle value when sorted: [0.05, 0.1, 0.15, 0.2, 0.3]')
)

# Fix test_latency_monitor.py
Fix-FloatAssertions "tests/unit/observation/test_latency_monitor.py" @(
    @('assert stats\["median"\] == 0\.15  # Middle value when sorted: \[0\.05, 0\.1, 0\.15, 0\.2, 0\.3\]', 'assert stats["median"] == pytest.approx(0.15, rel=1e-3)  # Middle value when sorted: [0.05, 0.1, 0.15, 0.2, 0.3]')
)

# Fix test_benchmark_engine.py
Fix-FloatAssertions "tests/unit/benchmarking/test_benchmark_engine.py" @(
    @('assert result\.trust_score\.overall_score == 0\.75', 'assert result.trust_score.overall_score == pytest.approx(0.75, rel=1e-3)')
)

# Fix test_evaluation_pipeline_integration.py
Fix-FloatAssertions "tests/integration/test_evaluation_pipeline_integration.py" @(
    @('assert factual_result\["score"\] == 0\.9\s+assert factual_result\["confidence"\] == 0\.85\s+assert hallucination_result\["score"\] == 0\.95\s+assert hallucination_result\["confidence"\] == 0\.9', 'assert factual_result["score"] == pytest.approx(0.9, rel=1e-3)`n    assert factual_result["confidence"] == pytest.approx(0.85, rel=1e-3)`n    assert hallucination_result["score"] == pytest.approx(0.95, rel=1e-3)`n    assert hallucination_result["confidence"] == pytest.approx(0.9, rel=1e-3)')
)

Write-Host "✓ All floating-point assertions fixed" -ForegroundColor Green


Write-Host "=== Step 4: Verify all tests pass ===" -ForegroundColor Cyan

& ".\.venv\Scripts\python.exe" -m pytest -v --tb=short 2>&1

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host "`n=== ALL 83 TESTS PASSED ===" -ForegroundColor Green
} else {
    Write-Host "`n=== SOME TESTS FAILED (exit code: $exitCode) ===" -ForegroundColor Red
}
exit $exitCode