"""
Factual correctness evaluation service.
"""

import asyncio
import logging
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

from core.interfaces import EvaluationService, EvaluationResult
from config.logging import get_logger

logger = get_logger(__name__)


class FactualCorrectnessEvaluator(EvaluationService):
    """
    Evaluates factual correctness of LLM responses.
    
    Uses LLM-as-a-judge approach to assess factual accuracy.
    """
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["response", "context"],
            template="""
You are an expert fact-checker. Evaluate the factual correctness of the following response.

Response to evaluate:
{response}

Context (including expected facts or reference information):
{context}

Please assess the factual correctness on a scale from 0.0 to 1.0, where:
- 0.0 = Completely factually incorrect
- 0.5 = Partially correct with some factual errors
- 1.0 = Completely factually accurate

Provide your evaluation in the following format:
SCORE: [score between 0.0 and 1.0]
CONFIDENCE: [confidence in your evaluation between 0.0 and 1.0]
REASONING: [brief explanation of your score]
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    async def evaluate(self, response: str, context: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate factual correctness of a response.
        
        Args:
            response: The LLM-generated response to evaluate
            context: Additional context including prompt, expected answer, etc.
            
        Returns:
            EvaluationResult containing the score and details
        """
        try:
            # Prepare context string
            context_str = str(context)
            
            # Run evaluation
            result = await self.chain.arun(
                response=response,
                context=context_str
            )
            
            # Parse result
            score, confidence, reasoning = self._parse_evaluation_result(result)
            
            return EvaluationResult(
                score=score,
                confidence=confidence,
                details={
                    "reasoning": reasoning,
                    "raw_result": result
                },
                metric_name="factual_correctness"
            )
        except Exception as e:
            logger.error(f"Error in factual correctness evaluation: {e}")
            return EvaluationResult(
                score=0.0,
                confidence=0.0,
                details={"error": str(e)},
                metric_name="factual_correctness"
            )
    
    def _parse_evaluation_result(self, result: str) -> tuple[float, float, str]:
        """Parse the LLM evaluation result."""
        score = 0.5  # Default score
        confidence = 0.5  # Default confidence
        reasoning = "Failed to parse evaluation result"
        
        try:
            lines = result.strip().split('\n')
            for line in lines:
                if line.startswith('SCORE:'):
                    score_str = line.split(':', 1)[1].strip()
                    score = max(0.0, min(1.0, float(score_str)))
                elif line.startswith('CONFIDENCE:'):
                    confidence_str = line.split(':', 1)[1].strip()
                    confidence = max(0.0, min(1.0, float(confidence_str)))
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
        except Exception as e:
            logger.warning(f"Could not parse evaluation result: {e}")
            
        return score, confidence, reasoning