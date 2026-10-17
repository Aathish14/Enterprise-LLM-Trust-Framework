import asyncio
import os
import time
from typing import Dict, Any, Optional, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.core.interfaces import EvaluationService, EvaluationResult, LLMAdapter
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


class FactualCorrectnessEvaluator(EvaluationService):
    """
    Evaluates factual correctness of LLM responses using LLM-as-a-judge.
    Supports either a custom LLMAdapter (e.g., NvidiaNIMAdapter) or a LangChain ChatOpenAI model.
    """

    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        llm_adapter: Optional[LLMAdapter] = None
    ):
        self.llm_adapter = llm_adapter
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

        if self.llm_adapter:
            self.llm = None
            self.chain = None
        else:
            self.llm = ChatOpenAI(model=llm_model, temperature=0)
            self.chain = self.prompt_template | self.llm

    async def evaluate(self, response: str, context: Dict[str, Any]) -> EvaluationResult:
        """Evaluate factual correctness of a response."""
        try:
            context_str = str(context)

            if self.llm_adapter:
                formatted_prompt = self.prompt_template.format(
                    response=response,
                    context=context_str
                )
                result_text = await _retry_llm_call(
                    lambda: self.llm_adapter.generate(formatted_prompt),
                    max_retries=2,
                    base_delay=0.5,
                )
            else:
                result = await _retry_llm_call(
                    lambda: self.chain.ainvoke({
                        "response": response,
                        "context": context_str
                    }),
                    max_retries=2,
                    base_delay=0.5,
                )
                result_text = result.content if hasattr(result, 'content') else str(result)

            score, confidence, reasoning = self._parse_evaluation_result(result_text)

            return EvaluationResult(
                score=score,
                confidence=confidence,
                details={
                    "reasoning": reasoning,
                    "raw_result": result_text
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

    def _parse_evaluation_result(self, result: str) -> Tuple[float, float, str]:
        """Parse the LLM evaluation result."""
        score = 0.5
        confidence = 0.5
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
