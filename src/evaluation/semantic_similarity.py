"""
Semantic similarity evaluation service.
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


class SemanticSimilarityEvaluator(EvaluationService):
    """
    Evaluates semantic similarity between LLM response and expected answer.
    
    Uses LLM-as-a-judge approach to assess semantic similarity.
    """
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["response", "expected_answer", "context"],
            template="""
You are an expert in semantic analysis. Evaluate the semantic similarity between the LLM response and the expected answer.

LLM Response:
{response}

Expected Answer:
{expected_answer}

Additional Context:
{context}

Please assess the semantic similarity on a scale from 0.0 to 1.0, where:
- 0.0 = Completely dissimilar in meaning
- 0.5 = Somewhat similar but with significant differences in meaning
- 1.0 = Essentially equivalent in meaning (paraphrase or same information)

Provide your evaluation in the following format:
SCORE: [score between 0.0 and 1.0]
CONFIDENCE: [confidence in your evaluation between 0.0 and 1.0]
REASONING: [brief explanation of your score]
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    async def evaluate(self, response: str, context: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate semantic similarity between response and expected answer.
        
        Args:
            response: The LLM-generated response to evaluate
            context: Additional context including expected answer, prompt, etc.
            
        Returns:
            EvaluationResult containing the score and details
        """
        try:
            # Extract expected answer from context
            expected_answer = context.get("expected_answer", "")
            if not expected_answer:
                logger.warning("No expected answer provided in context for semantic similarity evaluation")
                return EvaluationResult(
                    score=0.0,
                    confidence=0.0,
                    details={"error": "No expected answer provided"},
                    metric_name="semantic_similarity"
                )
            
            # Prepare context string (excluding expected_answer to avoid duplication)
            context_for_prompt = {k: v for k, v in context.items() if k != "expected_answer"}
            context_str = str(context_for_prompt)
            
            # Run evaluation
            result = await self.chain.arun(
                response=response,
                expected_answer=expected_answer,
                context=context_str
            )
            
            # Parse result
            score, confidence, reasoning = self._parse_evaluation_result(result)
            
            return EvaluationResult(
                score=score,
                confidence=confidence,
                details={
                    "reasoning": reasoning,
                    "raw_result": result,
                    "expected_answer": expected_answer
                },
                metric_name="semantic_similarity"
            )
        except Exception as e:
            logger.error(f"Error in semantic similarity evaluation: {e}")
            return EvaluationResult(
                score=0.0,
                confidence=0.0,
                details={"error": str(e)},
                metric_name="semantic_similarity"
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