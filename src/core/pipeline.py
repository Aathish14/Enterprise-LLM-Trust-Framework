"""
Evaluation pipeline orchestrator for the Enterprise LLM Trust Framework.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .interfaces import (
    EvaluationService, 
    CritiqueService, 
    TrustScoringService, 
    LLMAdapter,
    EvaluationResult,
    TrustScore
)
from config.logging import get_logger

logger = get_logger(__name__)


class EvaluationPipeline:
    """
    Orchestrates the multi-stage evaluation pipeline:
    Generation → Evaluation → Critique → Correction → Re-evaluation
    """
    
    def __init__(
        self,
        llm_adapter: LLMAdapter,
        evaluation_services: List[EvaluationService],
        critique_service: Optional[CritiqueService] = None,
        trust_scoring_service: Optional[TrustScoringService] = None,
        max_iterations: int = 2
    ):
        self.llm_adapter = llm_adapter
        self.evaluation_services = evaluation_services
        self.critique_service = critique_service
        self.trust_scoring_service = trust_scoring_service
        self.max_iterations = max_iterations
        
        logger.info(f"EvaluationPipeline initialized with {len(evaluation_services)} evaluation services")
    
    async def run_evaluation(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            prompt: Input prompt for LLM generation
            context: Additional context for evaluation
            
        Returns:
            Dictionary containing generation, evaluations, critique, trust score, and metadata
        """
        if context is None:
            context = {}
            
        context["prompt"] = prompt
        start_time = datetime.utcnow()
        
        logger.info(f"Starting evaluation pipeline for prompt: {prompt[:100]}...")
        
        # Stage 1: Generation
        logger.info("Stage 1: Generation")
        response = await self.llm_adapter.generate(prompt, **context)
        context["initial_response"] = response
        
        # Stage 2: Evaluation
        logger.info("Stage 2: Evaluation")
        evaluation_results = await self._run_evaluations(response, context)
        
        # Stage 3: Critique & Refinement Loop
        logger.info("Stage 3: Critique & Refinement")
        final_response = response
        critique_feedback = None
        
        for iteration in range(self.max_iterations):
            logger.info(f"Critique iteration {iteration + 1}/{self.max_iterations}")
            
            # Get critique feedback
            if self.critique_service:
                critique_feedback = await self.critique_service.critique(
                    final_response, evaluation_results
                )
                context["critique_feedback"] = critique_feedback
            
            # Check if refinement is needed
            needs_refinement = self._should_refine(critique_feedback, evaluation_results)
            
            if not needs_refinement or iteration == self.max_iterations - 1:
                break
                
            # Generate refined response
            logger.info("Generating refined response based on critique")
            refinement_prompt = self._build_refinement_prompt(
                prompt, final_response, critique_feedback, context
            )
            final_response = await self.llm_adapter.generate(refinement_prompt, **context)
            context[f"response_iteration_{iteration + 1}"] = final_response
            
            # Re-evaluate refined response
            logger.info("Re-evaluating refined response")
            evaluation_results = await self._run_evaluations(final_response, context)
        
        # Stage 4: Trust Scoring
        logger.info("Stage 4: Trust Scoring")
        trust_score = None
        if self.trust_scoring_service:
            trust_score = await self.trust_scoring_service.compute_trust_score(evaluation_results)
        
        # Prepare final result
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        result = {
            "prompt": prompt,
            "final_response": final_response,
            "initial_response": response,
            "evaluation_results": [
                {
                    "metric_name": result.metric_name,
                    "score": result.score,
                    "confidence": result.confidence,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in evaluation_results
            ],
            "trust_score": {
                "overall_score": trust_score.overall_score,
                "dimension_scores": trust_score.dimension_scores,
                "confidence": trust_score.confidence,
                "details": trust_score.details,
                "timestamp": trust_score.timestamp.isoformat()
            } if trust_score else None,
            "critique_feedback": critique_feedback,
            "metadata": {
                "processing_time_seconds": processing_time,
                "iterations": iteration + 1 if 'iteration' in locals() else 1,
                "timestamp": end_time.isoformat()
            }
        }
        
        logger.info(f"Evaluation pipeline completed in {processing_time:.2f}s")
        return result
    
    async def _run_evaluations(
        self, 
        response: str, 
        context: Dict[str, Any]
    ) -> List[EvaluationResult]:
        """Run all evaluation services on the response."""
        tasks = [
            service.evaluate(response, context)
            for service in self.evaluation_services
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        evaluation_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Evaluation service {self.evaluation_services[i].__class__.__name__} failed: {result}")
                # Create a failed evaluation result
                evaluation_results.append(EvaluationResult(
                    score=0.0,
                    confidence=0.0,
                    details={"error": str(result), "service": self.evaluation_services[i].__class__.__name__},
                    metric_name=self.evaluation_services[i].__class__.__name__
                ))
            else:
                evaluation_results.append(result)
                
        return evaluation_results
    
    def _should_refine(
        self, 
        critique_feedback: Optional[Dict[str, Any]], 
        evaluation_results: List[EvaluationResult]
    ) -> bool:
        """Determine if refinement is needed based on critique and evaluations."""
        if not critique_feedback:
            return False
            
        # Check if critique suggests refinement is needed
        if critique_feedback.get("needs_refinement", False):
            return True
            
        # Check if any evaluation scores are below threshold
        threshold = 0.6  # Configurable threshold
        for result in evaluation_results:
            if result.score < threshold:
                return True
                
        return False
    
    def _build_refinement_prompt(
        self, 
        original_prompt: str,
        current_response: str,
        critique_feedback: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build a refinement prompt based on critique feedback."""
        refinement_parts = [
            f"Original prompt: {original_prompt}",
            f"Current response: {current_response}",
            "",
            "Please improve the above response based on the following feedback:",
        ]
        
        if isinstance(critique_feedback, dict):
            for key, value in critique_feedback.items():
                if key != "needs_refinement":
                    refinement_parts.append(f"- {key}: {value}")
        else:
            refinement_parts.append(f"- {critique_feedback}")
            
        refinement_parts.extend([
            "",
            "Provide an improved response that addresses these concerns while staying true to the original prompt."
        ])
        
        return "\n".join(refinement_parts)