"""
Google Gemini LLM adapter for the Enterprise LLM Trust Framework.
"""

import logging
from typing import Dict, Any, Optional
import google.generativeai as genai
from .base_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseLLMAdapter):
    """
    Adapter for Google Gemini LLMs.
    """
    
    def __init__(self, 
                 model_name: str = "gemini-pro",
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Gemini adapter.
        
        Args:
            model_name: Name of the Gemini model to use
            api_key: Gemini API key (if None, will use environment variable)
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Configure Gemini API
        if api_key:
            genai.configure(api_key=api_key)
        # If no api_key provided, it will use the GOOGLE_API_KEY environment variable
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"Initialized Gemini adapter for model: {model_name}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the Gemini LLM.
        
        Args:
            prompt: Input prompt for the LLM
            **kwargs: Additional parameters for generation (temperature, max_output_tokens, etc.)
            
        Returns:
            Generated response string
        """
        try:
            # Merge instance config with call-specific kwargs
            params = self._merge_kwargs(**kwargs)
            
            # Map common parameter names to Gemini-specific ones
            gemini_params = {}
            if "temperature" in params:
                gemini_params["temperature"] = params["temperature"]
            if "max_tokens" in params:
                gemini_params["max_output_tokens"] = params["max_tokens"]
            if "top_p" in params:
                gemini_params["top_p"] = params["top_p"]
            
            # Generate content asynchronously
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(**gemini_params)
            )
            
            # Extract the generated text
            generated_text = response.text
            
            logger.debug(f"Generated response of length {len(generated_text)} from {self.model_name}")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Gemini model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "google",
            "model_name": self.model_name,
            "adapter_type": "GeminiAdapter",
            "config": self.config
        }