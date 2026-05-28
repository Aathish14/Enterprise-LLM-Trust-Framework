"""
Response quality evaluation service.
"""

import asyncio
import logging
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.core.interfaces import EvaluationService, EvaluationResult
from config.logging import get_logger

logger = get_logger(__name__)


class ResponseQualityEvaluator(EvaluationService):
    """
    Evaluates overall response quality including grammar, fluency, completeness, and relevance.
    
    Uses LLM-as-a-judge approach to assess multiple quality dimensions.
    """
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["response", "context"],
            template="""
You are an expert in evaluating response quality. Assess the overall quality of the following response considering:
- Grammar and fluency
- Completeness of answer
- Relevance to the prompt
- Clarity and coherence

Response to evaluate:
{response}

Context (including the original prompt and any specific quality criteria):
{context}

Please assess the overall response quality on a scale from 0.0 to 1.0, where:
- 0.0 = Very poor quality (unintelligible, completely irrelevant, major grammar issues)
- 0.5 = Moderate quality (somewhat understandable, partially relevant, some quality issues)
- 1.0 = Excellent quality (clear, fluent, completely relevant, high-quality response)

Provide your evaluation in the following format:
SCORE: [score between 0.0 and 1.0]
CONFIDENCE: [confidence in your evaluation between 0.0 and 1.0]
REASONING: [brief explanation of your score]
GRAMMAR_FLUENCY: [score for grammar and fluency between 0.0 and 1.0]
COMPLETENESS: [score for completeness between 0.0 and 1.0]
RELEVANCE: [score for relevance between 0.0 and 1.0]
CLARITY: [score for clarity and coherence between 0.0 and 1.0]
"""
        )
        self.chain = self.prompt_template | self.llm
    
    async def evaluate(self, response: str, context: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate overall response quality.
        
        Args:
            response: The LLM-generated response to evaluate
            context: Additional context including prompt and quality criteria
            
        Returns:
            EvaluationResult containing the score and details
        """
        try:
            # Prepare context string
            context_str = str(context)
            
            # Run evaluation
            result = await self.chain.ainvoke({
                "response": response,
                "context": context_str
            })
            
            # Extract text content from result if it's an AIMessage
            if hasattr(result, 'content'):
                result_text = result.content
            else:
                result_text = str(result)
            
            # Parse result
            score, confidence, grammar_fluency, completeness, relevance, clarity, reasoning = self._parse_evaluation_result(result_text)
            
            return EvaluationResult(
                score=score,
                confidence=confidence,
                details={
                    "reasoning": reasoning,
                    "raw_result": result_text,
                    "grammar_fluency": grammar_fluency,
                    "completeness": completeness,
                    "relevance": relevance,
                    "clarity": clarity
                },
                metric_name="response_quality"
            )
        except Exception as e:
            logger.error(f"Error in response quality evaluation: {e}")
            return EvaluationResult(
                score=0.0,
                confidence=0.0,
                details={"error": str(e)},
                metric_name="response_quality"
            )
    
    def _parse_evaluation_result(self, result: str) -> tuple[float, float, float, float, float, float, str]:
        """Parse the LLM evaluation result."""
        score = 0.5  # Default score
        confidence = 0.5  # Default confidence
        grammar_fluency = 0.5
        completeness = 0.5
        relevance = 0.5
        clarity = 0.5
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
                elif line.startswith('GRAMMAR_FLUENCY:'):
                    gf_str = line.split(':', 1)[1].strip()
                    grammar_fluency = max(0.0, min(1.0, float(gf_str)))
                elif line.startswith('COMPLETENESS:'):
                    c_str = line.split(':', 1)[1].strip()
                    completeness = max(0.0, min(1.0, float(c_str)))
                elif line.startswith('RELEVANCE:'):
                    r_str = line.split(':', 1)[1].strip()
                    relevance = max(0.0, min(1.0, float(r_str)))
                elif line.startswith('CLARITY:'):
                    cl_str = line.split(':', 1)[1].strip()
                    clarity = max(0.0, min(1.0, float(cl_str)))
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
        except Exception as e:
            logger.warning(f"Could not parse evaluation result: {e}")
            
        return score, confidence, grammar_fluency, completeness, relevance, clarity, reasoning