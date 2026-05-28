"""
Unit tests for the OpenAI LLM adapter.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from llm.openai_adapter import OpenAIAdapter
from core.interfaces import LLMAdapter


@pytest.fixture
def openai_adapter():
    """Create an OpenAIAdapter instance for testing."""
    with patch('llm.openai_adapter.AsyncOpenAI'):
        adapter = OpenAIAdapter(model_name="gpt-3.5-turbo", api_key="test-key")
        return adapter


@pytest.mark.asyncio
async def test_openai_adapter_initialization(openai_adapter):
    """Test that the OpenAIAdapter initializes correctly."""
    assert openai_adapter is not None
    assert openai_adapter.model_name == "gpt-3.5-turbo"
    # The config should be empty since api_key is passed separately to AsyncOpenAI
    assert openai_adapter.config == {}


@pytest.mark.asyncio
async def test_openai_adapter_generate(openai_adapter):
    """Test the generate method of OpenAIAdapter."""
    # Mock the AsyncOpenAI client
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response
    
    openai_adapter.client = mock_client
    
    result = await openai_adapter.generate("Test prompt", temperature=0.7)
    
    assert result == "Test response"
    # Check that the call was made with the correct parameters
    # Note: max_tokens and stream are only passed if explicitly set
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]["model"] == "gpt-3.5-turbo"
    assert call_args[1]["messages"] == [{"role": "user", "content": "Test prompt"}]
    assert call_args[1]["temperature"] == 0.7
    # max_tokens and stream should not be in the call if not specified
    assert "max_tokens" not in call_args[1] or call_args[1]["max_tokens"] is None
    assert "stream" not in call_args[1] or call_args[1]["stream"] is False


@pytest.mark.asyncio
async def test_openai_adapter_get_model_info(openai_adapter):
    """Test the get_model_info method of OpenAIAdapter."""
    info = openai_adapter.get_model_info()
    
    assert isinstance(info, dict)
    assert info["provider"] == "openai"
    assert info["model_name"] == "gpt-3.5-turbo"
    assert info["adapter_type"] == "OpenAIAdapter"
    # Check that config is present (should be empty for this fixture)
    assert "config" in info