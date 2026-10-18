import os
"""
Configuration settings for the Enterprise LLM Trust Framework.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    debug: bool = False
    app_name: str = "Enterprise LLM Trust Framework"
    version: str = "0.1.0"
    # NVIDIA NIM Configuration
    nvidia_api_key: Optional[str] = None
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    """Application settings."""
    
    # Application
    APP_NAME: str = "Enterprise LLM Trust Framework"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: os.urandom(32).hex())
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: Optional[str] = None
    
    # LLM Provider API Keys (loaded from environment)
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    
    # Observability
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    
    @property
    def mlflow_tracking_uri_absolute(self) -> str:
        """Get absolute MLflow tracking URI for consistent database access."""
        return "sqlite:///" + os.path.abspath("mlflow.db").replace("\\", "/")
    WANDB_PROJECT: str = "llm-trust-framework"
    WANDB_API_KEY: Optional[str] = None
    
    # Evaluation Settings
    EVALUATION_BATCH_SIZE: int = 32
    MAX_TOKENS_PER_REQUEST: int = 4096
    TEMPERATURE: float = 0.1
    
    # Trust Scoring Weights
    # Only factual_correctness has an evaluator wired to LLMAdapter (NVIDIA NIM compatible).
    # The other evaluators use ChatOpenAI directly and cannot work with NVIDIA NIM.
    TRUST_SCORE_WEIGHTS: dict = {
        "factual_correctness": 1.0,
    }
    
    class Config:
        extra = "ignore"
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()