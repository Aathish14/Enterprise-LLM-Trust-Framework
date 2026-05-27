"""
Core interfaces for the Enterprise LLM Trust Framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EvaluationResult:
    """Result of an evaluation metric."""
    score: float
    confidence: float
    details: Dict[str, Any]
    metric_name: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class TrustScore:
    """Composite trust score from multiple evaluation dimensions."""
    overall_score: float
    dimension_scores: Dict[str, float]
    confidence: float
    details: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class EvaluationService(ABC):
    """Abstract base class for evaluation services."""
    
    @abstractmethod
    async def evaluate(self, response: str, context: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate a response based on the specific metric.
        
        Args:
            response: The LLM-generated response to evaluate
            context: Additional context including prompt, expected answer, etc.
            
        Returns:
            EvaluationResult containing the score and details
        """
        pass


class CritiqueService(ABC):
    """Abstract base class for critique services."""
    
    @abstractmethod
    async def critique(self, response: str, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Critique a response based on evaluation results.
        
        Args:
            response: The LLM-generated response to critique
            evaluation_results: List of evaluation results from various metrics
            
        Returns:
            Critique feedback and suggestions for improvement
        """
        pass


class TrustScoringService(ABC):
    """Abstract base class for trust scoring services."""
    
    @abstractmethod
    async def compute_trust_score(self, evaluation_results: List[EvaluationResult]) -> TrustScore:
        """
        Compute a composite trust score from multiple evaluation results.
        
        Args:
            evaluation_results: List of evaluation results from various metrics
            
        Returns:
            TrustScore containing the composite score and details
        """
        pass


class LLMAdapter(ABC):
    """Abstract base class for LLM provider adapters."""
    
    @abstractmethod
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
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.
        
        Returns:
            Dictionary containing model information
        """
        pass