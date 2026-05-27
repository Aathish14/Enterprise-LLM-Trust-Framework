"""
Base LLM adapter for the Enterprise LLM Trust Framework.
"""

import abc
import logging
from typing import Dict, Any, Optional
from core.interfaces import LLMAdapter

logger = logging.getLogger(__name__)


class BaseLLMAdapter(LLMAdapter, abc.ABC):
    """
    Base class for LLM provider adapters.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM adapter.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        logger.info(f"Initialized {self.__class__.__name__} for model: {model_name}")
    
    @abc.abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt for the LLM
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated response string
        """
        pass
    
    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.
        
        Returns:
            Dictionary containing model information
        """
        pass
    
    def _merge_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Merge instance config with call-specific kwargs.
        
        Args:
            **kwargs: Call-specific parameters
            
        Returns:
            Merged parameters dictionary
        """
        merged = self.config.copy()
        merged.update(kwargs)
        return merged