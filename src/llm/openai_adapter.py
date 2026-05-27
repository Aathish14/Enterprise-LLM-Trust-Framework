"""
OpenAI LLM adapter for the Enterprise LLM Trust Framework.
"""

import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from .base_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseLLMAdapter):
    """
    Adapter for OpenAI LLMs (GPT-3.5, GPT-4, etc.).
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 organization: Optional[str] = None,
                 **kwargs):
        """
        Initialize the OpenAI adapter.
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (if None, will use environment variable)
            organization: OpenAI organization ID (optional)
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization
        )
        
        logger.info(f"Initialized OpenAI adapter for model: {model_name}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the OpenAI LLM.
        
        Args:
            prompt: Input prompt for the LLM
            **kwargs: Additional parameters for generation (temperature, max_tokens, etc.)
            
        Returns:
            Generated response string
        """
        try:
            # Merge instance config with call-specific kwargs
            params = self._merge_kwargs(**kwargs)
            
            # Remove parameters that aren't valid for the OpenAI API
            valid_params = {
                "temperature", "max_tokens", "top_p", "frequency_penalty", 
                "presence_penalty", "stop", "n", "stream", "logit_bias", 
                "logprobs", "echo", "user"
            }
            filtered_params = {k: v for k, v in params.items() if k in valid_params}
            
            # Make the API call
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **filtered_params
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content
            
            logger.debug(f"Generated response of length {len(generated_text)} from {self.model_name}")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the OpenAI model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "adapter_type": "OpenAIAdapter",
            "config": self.config
        }