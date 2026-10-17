import logging
import os
from typing import Any, Dict, Optional
from openai import AsyncOpenAI
from src.core.interfaces import LLMAdapter

logger = logging.getLogger(__name__)


class NvidiaNIMAdapter(LLMAdapter):
    """
    Adapter for NVIDIA NIM APIs implementing src.core.interfaces.LLMAdapter.
    Uses AsyncOpenAI client for fully non-blocking execution in EvaluationPipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/nemotron-3-ultra-550b-a55b",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        max_retries: int = 2,
        **kwargs
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_params = kwargs

        if not self.api_key and "integrate.api.nvidia.com" in self.base_url:
            raise ValueError("NVIDIA_API_KEY must be provided or set in environment variable.")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Asynchronous generate completion using NVIDIA NIM endpoint."""
        temp = kwargs.get("temperature", self.temperature)
        max_t = kwargs.get("max_tokens", self.max_tokens)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_t
        )

        if not response or not response.choices:
            raise ValueError("NVIDIA NIM returned no choices in response.")

        choice = response.choices[0]
        if not choice or not choice.message:
            raise ValueError("NVIDIA NIM returned a choice with no message content.")

        return choice.message.content or ""

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the current NIM endpoint config."""
        return {
            "provider": "NVIDIA NIM",
            "model": self.model,
            "base_url": self.base_url,
            "parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                **self.extra_params
            }
        }

    @staticmethod
    def sanitize_error(exc: Exception) -> Dict[str, Any]:
        """Extract sanitized diagnostic information from an exception.

        Never returns API keys, authorization headers, or secret-bearing data.
        """
        info: Dict[str, Any] = {
            "exception_type": type(exc).__name__,
            "message": str(exc)[:200] if str(exc) else "",
        }

        # Extract optional attributes only if present — never assume they exist.
        for attr in ("status_code", "code", "message", "type", "request_id"):
            value = getattr(exc, attr, None)
            if value is not None:
                info[attr] = value

        # Inspect nested response/body attributes for HTTP status codes.
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if status is not None:
                info["http_status"] = status
            body = getattr(response, "text", None)
            if body is not None:
                info["response_body"] = str(body)[:200]

        return info
