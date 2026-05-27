"""
Secrets management for the Enterprise LLM Trust Framework.
"""

import os
from typing import Optional, Dict, Any
from .settings import settings


class SecretsManager:
    """Manages application secrets and sensitive configuration."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieves a secret value from environment variables or cache.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Secret value or default if not found
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        # Get from environment
        value = os.getenv(key, default)
        
        # Cache non-None values
        if value is not None:
            self._cache[key] = value
            
        return value
    
    def get_openai_api_key(self) -> Optional[str]:
        """Gets OpenAI API key."""
        return self.get_secret("OPENAI_API_KEY")
    
    def get_gemini_api_key(self) -> Optional[str]:
        """Gets Gemini API key."""
        return self.get_secret("GEMINI_API_KEY")
    
    def get_anthropic_api_key(self) -> Optional[str]:
        """Gets Anthropic API key."""
        return self.get_secret("ANTHROPIC_API_KEY")
    
    def get_huggingface_api_key(self) -> Optional[str]:
        """Gets Hugging Face API key."""
        return self.get_secret("HUGGINGFACE_API_KEY")
    
    def get_wandb_api_key(self) -> Optional[str]:
        """Gets Weights & Biases API key."""
        return self.get_secret("WANDB_API_KEY")
    
    def validate_required_secrets(self) -> Dict[str, bool]:
        """
        Validates that required secrets are present.
        
        Returns:
            Dictionary mapping secret names to validation status
        """
        required_secrets = [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY", 
            "ANTHROPIC_API_KEY",
            "HUGGINGFACE_API_KEY",
            "WANDB_API_KEY"
        ]
        
        validation_results = {}
        for secret in required_secrets:
            validation_results[secret] = self.get_secret(secret) is not None
            
        return validation_results
    
    def is_configured(self) -> bool:
        """
        Checks if all required secrets are configured.
        
        Returns:
            True if all required secrets are present, False otherwise
        """
        results = self.validate_required_secrets()
        return all(results.values())


# Global secrets manager instance
secrets_manager = SecretsManager()