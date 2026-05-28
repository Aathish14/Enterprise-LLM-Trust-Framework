"""
Configuration settings for the Enterprise LLM Trust Framework.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
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
    WANDB_PROJECT: str = "llm-trust-framework"
    WANDB_API_KEY: Optional[str] = None
    
    # Evaluation Settings
    EVALUATION_BATCH_SIZE: int = 32
    MAX_TOKENS_PER_REQUEST: int = 4096
    TEMPERATURE: float = 0.1
    
    # Trust Scoring Weights
    TRUST_SCORE_WEIGHTS: dict = {
        "factual_correctness": 0.2,
        "hallucination_likelihood": 0.15,
        "semantic_similarity": 0.15,
        "context_relevance": 0.1,
        "logical_consistency": 0.1,
        "grammar_fluency": 0.1,
        "toxicity_safety": 0.1,
        "response_completeness": 0.1
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()