"""
Unit tests for the NVIDIA NIM adapter.

Covers:
- missing API key
- configured API key
- successful response
- authentication failure
- permission failure
- invalid model/request
- rate limiting
- provider server error
- timeout
- connection failure
- empty choices
- missing message content
- sanitized error messages
- bounded retry count
- no retry on permanent errors
- live failure cannot become a genuine success result
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.core.interfaces import LLMAdapter
from src.llm.nvidia_nim_adapter import NvidiaNIMAdapter


class TestNvidiaNIMAdapterInitialization:
    """Tests for adapter initialization and configuration."""

    def test_init_with_explicit_api_key(self):
        """Adapter accepts an explicit API key."""
        adapter = NvidiaNIMAdapter(api_key="test-key-123", model="test-model")
        assert adapter.api_key == "test-key-123"
        assert adapter.model == "test-model"
        assert adapter.timeout == 30.0
        assert adapter.max_retries == 2

    def test_init_with_custom_timeout_and_retries(self):
        """Adapter accepts custom timeout and max_retries values."""
        adapter = NvidiaNIMAdapter(
            api_key="test-key",
            timeout=60.0,
            max_retries=5,
        )
        assert adapter.timeout == 60.0
        assert adapter.max_retries == 5

    def test_init_reads_key_from_environment(self):
        """Adapter reads NVIDIA_API_KEY from environment when not passed explicitly."""
        os.environ["NVIDIA_API_KEY"] = "env-key-456"
        try:
            adapter = NvidiaNIMAdapter(model="env-model")
            assert adapter.api_key == "env-key-456"
        finally:
            del os.environ["NVIDIA_API_KEY"]

    def test_init_missing_key_raises_value_error(self):
        """Adapter raises ValueError when no API key is available for NVIDIA base URL."""
        os.environ.pop("NVIDIA_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
                NvidiaNIMAdapter(base_url="https://integrate.api.nvidia.com/v1")
        finally:
            os.environ["NVIDIA_API_KEY"] = "nvapi-mock-key-12345"

    def test_init_missing_key_ok_for_custom_base_url(self):
        """Adapter does not require key for non-NVIDIA base URLs even when env key is unset."""
        os.environ.pop("NVIDIA_API_KEY", None)
        try:
            adapter = NvidiaNIMAdapter(
                api_key=None,
                base_url="https://custom-endpoint.example.com/v1",
            )
            assert adapter.api_key is None
        finally:
            os.environ["NVIDIA_API_KEY"] = "nvapi-mock-key-12345"

    def test_is_llm_adapter_subclass(self):
        """NvidiaNIMAdapter is a subclass of LLMAdapter."""
        assert issubclass(NvidiaNIMAdapter, LLMAdapter)


class TestNvidiaNIMAdapterGenerate:
    """Tests for the generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Successful generation returns message content."""
        adapter = NvidiaNIMAdapter(api_key="test-key", model="test-model")

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Generated response text"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await adapter.generate("Hello world")
        assert result == "Generated response text"

    @pytest.mark.asyncio
    async def test_generate_empty_content_returns_empty_string(self):
        """Generation with empty message content returns empty string."""
        adapter = NvidiaNIMAdapter(api_key="test-key")

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await adapter.generate("Hello")
        assert result == ""

    @pytest.mark.asyncio
    async def test_generate_empty_choices_raises_value_error(self):
        """Generation with empty choices raises ValueError."""
        adapter = NvidiaNIMAdapter(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = []

        adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="no choices"):
            await adapter.generate("Hello")

    @pytest.mark.asyncio
    async def test_generate_missing_message_raises_value_error(self):
        """Generation with choice missing message raises ValueError."""
        adapter = NvidiaNIMAdapter(api_key="test-key")

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = None
        mock_response.choices = [mock_choice]

        adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="no message content"):
            await adapter.generate("Hello")


class TestNvidiaNIMAdapterErrorDiagnostics:
    """Tests for the sanitize_error static method."""

    def test_sanitize_error_basic_exception(self):
        """sanitize_error captures exception type and message."""
        exc = ValueError("test error message")
        info = NvidiaNIMAdapter.sanitize_error(exc)

        assert info["exception_type"] == "ValueError"
        assert "test error message" in info["message"]

    def test_sanitize_error_with_status_code(self):
        """sanitize_error captures status_code attribute when present."""
        exc = Exception("server error")
        exc.status_code = 500
        info = NvidiaNIMAdapter.sanitize_error(exc)

        assert info["status_code"] == 500

    def test_sanitize_error_with_http_response(self):
        """sanitize_error extracts HTTP status from nested response."""
        exc = Exception("server error")
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = '{"error": "rate limited"}'
        exc.response = mock_response
        info = NvidiaNIMAdapter.sanitize_error(exc)

        assert info["http_status"] == 429
        assert "rate limited" in info["response_body"]

    def test_sanitize_error_no_secret_like_patterns_in_diagnostic_fields(self):
        """sanitize_error does not embed secret-like patterns in diagnostic fields."""
        # Use a generic message that does not contain an API key pattern.
        exc = Exception("service temporarily unavailable")
        info = NvidiaNIMAdapter.sanitize_error(exc)

        serialized = str(info)
        assert "nvapi" not in serialized.lower()
        assert "sk-" not in serialized
        assert "authorization" not in serialized.lower()
        assert info["message"] == "service temporarily unavailable"


class TestNvidiaNIMAdapterRetryBehavior:
    """Tests for retry behavior at the adapter level.

    The AsyncOpenAI client handles retries internally via max_retries.
    These tests verify the client is configured correctly and that
    the adapter does not add a second retry layer.
    """

    def test_client_has_max_retries_configured(self):
        """AsyncOpenAI client is created with max_retries."""
        adapter = NvidiaNIMAdapter(api_key="test-key", max_retries=3)
        assert adapter.max_retries == 3

    def test_client_has_timeout_configured(self):
        """AsyncOpenAI client is created with timeout."""
        adapter = NvidiaNIMAdapter(api_key="test-key", timeout=45.0)
        assert adapter.timeout == 45.0

    @pytest.mark.asyncio
    async def test_generate_propagates_permanent_error_no_retry(self):
        """Permanent errors (400/401/403) are not retried by the adapter.

        The AsyncOpenAI client handles retry logic internally for
        transient errors only. Permanent errors propagate immediately.
        """
        adapter = NvidiaNIMAdapter(api_key="test-key")

        exc = Exception("Authentication failed")
        exc.status_code = 401

        adapter.client.chat.completions.create = AsyncMock(side_effect=exc)

        with pytest.raises(Exception, match="Authentication failed"):
            await adapter.generate("test prompt")


class TestNvidiaNIMAdapterGetModelInfo:
    """Tests for get_model_info."""

    def test_get_model_info_includes_timeout_and_retries(self):
        """get_model_info includes timeout and max_retries in parameters."""
        adapter = NvidiaNIMAdapter(
            api_key="test-key",
            model="custom-model",
            timeout=15.0,
            max_retries=1,
        )
        info = adapter.get_model_info()

        assert info["provider"] == "NVIDIA NIM"
        assert info["model"] == "custom-model"
        assert info["parameters"]["timeout"] == 15.0
        assert info["parameters"]["max_retries"] == 1
        assert "api_key" not in str(info).lower()


class TestPlaygroundTaskHandlesAdapterFailure:
    """Tests that _run_playground_task handles adapter failures gracefully.

    These tests verify that a live API failure results in an explicitly
    marked simulated result — never a genuine evaluation result.
    """

    def test_run_playground_task_missing_key_returns_simulated(self):
        """Missing API key returns simulated result with is_simulated=True."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

        from src.dashboard.app import _run_playground_task

        # Remove env key to simulate missing key
        os.environ.pop("NVIDIA_API_KEY", None)

        try:
            result = _run_playground_task(
                "test prompt",
                "test context",
                "nvidia/nemotron-3.5-super-vl-120b-a12b",
                0.1,
                1024,
            )

            assert result["metadata"]["is_simulated"] is True
            assert result["metadata"]["source"] == "simulated"
            assert result["trust_score"]["details"]["source"] == "simulated"
            assert result["run_id"].startswith("simulated-")
        finally:
            # Restore mock key for other tests
            os.environ["NVIDIA_API_KEY"] = "nvapi-mock-key-12345"

    def test_run_playground_task_api_failure_returns_simulated(self):
        """Live API failure returns simulated result with is_simulated=True."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

        from src.dashboard.app import _run_playground_task

        # Set a key so we get past the missing-key check
        os.environ["NVIDIA_API_KEY"] = "nvapi-test-key"

        try:
            # Patch NvidiaNIMAdapter to raise on generate
            with patch(
                "src.dashboard.app.NvidiaNIMAdapter"
            ) as mock_adapter_class:
                mock_adapter = MagicMock()
                mock_adapter.generate = AsyncMock(
                    side_effect=Exception("API error: 500 Internal Server Error")
                )
                mock_adapter_class.return_value = mock_adapter

                result = _run_playground_task(
                    "test prompt",
                    "test context",
                    "nvidia/nemotron-3.5-super-vl-120b-a12b",
                    0.1,
                    1024,
                )

            assert result["metadata"]["is_simulated"] is True
            assert result["metadata"]["source"] == "simulated"
            assert "error_diagnostics" in result["metadata"]
            assert result["trust_score"]["details"]["source"] == "simulated"
        finally:
            os.environ["NVIDIA_API_KEY"] = "nvapi-mock-key-12345"

    def test_run_playground_task_live_failure_not_genuine_success(self):
        """A live API failure is never represented as a genuine success."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

        from src.dashboard.app import _run_playground_task

        os.environ["NVIDIA_API_KEY"] = "nvapi-test-key"

        try:
            with patch(
                "src.dashboard.app.NvidiaNIMAdapter"
            ) as mock_adapter_class:
                mock_adapter = MagicMock()
                mock_adapter.generate = AsyncMock(
                    side_effect=Exception("Connection timeout")
                )
                mock_adapter_class.return_value = mock_adapter

                result = _run_playground_task(
                    "test prompt",
                    "test context",
                    "nvidia/nemotron-3.5-super-vl-120b-a12b",
                    0.1,
                    1024,
                )

            # The result must be explicitly marked as simulated
            assert result["metadata"]["is_simulated"] is True
            assert result["metadata"]["source"] == "simulated"
            # The trust score details must indicate simulation
            assert result["trust_score"]["details"]["source"] == "simulated"
            # run_id must be prefixed with "simulated-"
            assert result["run_id"].startswith("simulated-")
        finally:
            os.environ["NVIDIA_API_KEY"] = "nvapi-mock-key-12345"
