"""
Hallucination detection evaluation service.
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


class HallucinationDetector(EvaluationService):
    """
    Detects hallucinations in LLM responses.
    
    Uses LLM-as-a-judge approach to assess likelihood of hallucination.
    Lower scores indicate higher hallucination likelihood.
    """
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["response", "context"],
            template="""
You are an expert hallucination detector. Evaluate the likelihood of hallucination in the following response.

Response to evaluate:
{response}

Context (including source documents, facts, or reference information that should ground the response):
{context}

Please assess the hallucination likelihood on a scale from 0.0 to 1.0, where:
- 0.0 = Very likely hallucinated (mostly or completely fabricated)
- 0.5 = Somewhat likely hallucinated (contains noticeable fabricated elements)
- 1.0 = Very unlikely hallucinated (factually grounded in provided context)

Provide your evaluation in the following format:
SCORE: [score between 0.0 and 1.0]
CONFIDENCE: [confidence in your evaluation between 0.0 and 1.0]
REASONING: [brief explanation of your score]
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    async def evaluate(self, response: str, context: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate hallucination likelihood of a response.
        
        Args:
            response: The LLM-generated response to evaluate
            context: Additional context including source documents or expected facts
            
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
                metric_name="hallucination_likelihood"
            )
        except Exception as e:
            logger.error(f"Error in hallucination detection evaluation: {e}")
            return EvaluationResult(
                score=0.0,
                confidence=0.0,
                details={"error": str(e)},
                metric_name="hallucination_likelihood"
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