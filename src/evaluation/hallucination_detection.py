"""
Hallucination detection evaluation service.
"""

import asyncio
import logging
import time
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.core.interfaces import EvaluationService, EvaluationResult
from config.logging import get_logger

logger = get_logger(__name__)


async def _retry_llm_call(call_func, max_retries: int = 2, base_delay: float = 0.5):
    """Retry an LLM API call with exponential backoff on failure."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await call_func()
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"NVIDIA NIM API error ({exc}). Retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})."
                )
                await asyncio.sleep(delay)
            else:
                logger.warning(
                    f"NVIDIA NIM API error ({exc}). Falling back to simulated metric scores."
                )
    raise last_error


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
        self.chain = self.prompt_template | self.llm
    
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

            # Run evaluation with retry
            result = await _retry_llm_call(
                lambda: self.chain.ainvoke({
                    "response": response,
                    "context": context_str
                }),
                max_retries=2,
                base_delay=0.5,
            )
            
            # Extract text content from result if it's an AIMessage
            if hasattr(result, 'content'):
                result_text = result.content
            else:
                result_text = str(result)
            
            # Parse result
            score, confidence, reasoning = self._parse_evaluation_result(result_text)
            
            return EvaluationResult(
                score=score,
                confidence=confidence,
                details={
                    "reasoning": reasoning,
                    "raw_result": result_text
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
        """Parse the LLM evaluation result.

        Returns score=0.0 and confidence=0.0 when the LLM response cannot be
        parsed, so that a parse failure is not mistaken for a neutral 0.5 score.
        """
        score = 0.0
        confidence = 0.0
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