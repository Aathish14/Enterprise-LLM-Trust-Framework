import os
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from src.core.interfaces import LLMAdapter

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
        **kwargs
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

        if not self.api_key and "integrate.api.nvidia.com" in self.base_url:
            raise ValueError("NVIDIA_API_KEY must be provided or set in environment variable.")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate completion using NVIDIA NIM endpoint."""
        temp = kwargs.get("temperature", self.temperature)
        max_t = kwargs.get("max_tokens", self.max_tokens)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_t
        )
        return response.choices[0].message.content or ""

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the current NIM endpoint config."""
        return {
            "provider": "NVIDIA NIM",
            "model": self.model,
            "base_url": self.base_url,
            "parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **self.extra_params
            }
        }
