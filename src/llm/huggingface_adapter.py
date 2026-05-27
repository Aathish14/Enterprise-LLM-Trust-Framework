"""
HuggingFace LLM adapter for the Enterprise LLM Trust Framework.
"""

import logging
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .base_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class HuggingFaceAdapter(BaseLLMAdapter):
    """
    Adapter for HuggingFace LLMs.
    """
    
    def __init__(self, 
                 model_name: str = "gpt2",
                 device: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 **kwargs):
        """
        Initialize the HuggingFace adapter.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on (e.g., "cpu", "cuda", "cuda:0")
            torch_dtype: Torch dtype for the model (e.g., torch.float16)
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Set torch dtype
        if torch_dtype is None and device.startswith("cuda"):
            torch_dtype = torch.float16
        self.torch_dtype = torch_dtype
        
        # Load model and tokenizer
        logger.info(f"Loading HuggingFace model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None
        )
        
        # Create pipeline for easier generation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device.startswith("cuda") else -1,
            torch_dtype=torch_dtype
        )
        
        logger.info(f"Initialized HuggingFace adapter for model: {model_name}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the HuggingFace LLM.
        
        Args:
            prompt: Input prompt for the LLM
            **kwargs: Additional parameters for generation (max_length, temperature, etc.)
            
        Returns:
            Generated response string
        """
        try:
            # Merge instance config with call-specific kwargs
            params = self._merge_kwargs(**kwargs)
            
            # Set default generation parameters
            generation_kwargs = {
                "max_length": params.get("max_length", 100),
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 0.9),
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Override with any provided parameters
            for key in ["max_length", "temperature", "top_p", "do_sample", "num_return_sequences"]:
                if key in params:
                    generation_kwargs[key] = params[key]
            
            # Generate text
            outputs = self.generator(
                prompt,
                **generation_kwargs
            )
            
            # Extract the generated text (remove the prompt)
            generated_text = outputs[0]["generated_text"]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            logger.debug(f"Generated response of length {len(generated_text)} from {self.model_name}")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response from HuggingFace: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the HuggingFace model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "huggingface",
            "model_name": self.model_name,
            "adapter_type": "HuggingFaceAdapter",
            "device": self.device,
            "torch_dtype": str(self.torch_dtype) if self.torch_dtype else None,
            "config": self.config
        }