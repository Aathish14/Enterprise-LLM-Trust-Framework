"""
Anthropic Claude LLM adapter for the Enterprise LLM Trust Framework.
"""

import logging
from typing import Dict, Any, Optional
import anthropic
from .base_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class ClaudeAdapter(BaseLLMAdapter):
    """
    Adapter for Anthropic Claude LLMs.
    """
    
    def __init__(self, 
                 model_name: str = "claude-2",
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Claude adapter.
        
        Args:
            model_name: Name of the Claude model to use
            api_key: Anthropic API key (if None, will use environment variable)
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Initialize Anthropic client
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key
        )
        
        logger.info(f"Initialized Claude adapter for model: {model_name}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the Claude LLM.
        
        Args:
            prompt: Input prompt for the LLM
            **kwargs: Additional parameters for generation (temperature, max_tokens, etc.)
            
        Returns:
            Generated response string
        """
        try:
            # Merge instance config with call-specific kwargs
            params = self._merge_kwargs(**kwargs)
            
            # Prepare the message
            message = anthropic.types.MessageCreateParams(
                model=self.model_name,
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7),
                system=params.get("system", ""),  # Optional system prompt
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Remove None values
            message = {k: v for k, v in message.items() if v is not None}
            
            # Make the API call
            response = await self.client.messages.create(**message)
            
            # Extract the generated text
            generated_text = response.content[0].text
            
            logger.debug(f"Generated response of length {len(generated_text)} from {self.model_name}")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response from Claude: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Claude model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "anthropic",
            "model_name": self.model_name,
            "adapter_type": "ClaudeAdapter",
            "config": self.config
        }