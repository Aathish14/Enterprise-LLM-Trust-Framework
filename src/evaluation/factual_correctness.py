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
            template="""\
You are a strict factual-correctness evaluator for enterprise AI governance content.

GROUND TRUTH CONCEPTS: {context}
LLM RESPONSE TO EVALUATE: {response}

INSTRUCTIONS:
1. Identify each distinct concept in the ground truth.
2. For each concept, classify the LLM response's coverage as:
   - COVERED: The response explicitly addresses this concept with substantive detail.
   - PARTIAL: The response touches on this concept but lacks depth, uses vague
     language, or addresses only a sub-aspect.
   - MISSING: The response does not substantively address this concept at all.
   - CONTRADICTED: The response states something that conflicts with this concept.

3. Match by MEANING, not exact keywords. Examples of valid semantic matches:
   - "bias mitigation" or "equitable outcomes" → covers "Fairness"
   - "model documentation, interpretability techniques, SHAP, LIME" → covers "Explainability"
   - "clear disclosures to stakeholders" → covers "Transparency"
   - "designated ownership, governance council, escalation" → covers "Accountability"
   - "incident response, resilience, graceful degradation" → partially covers "Robustness"
   - "data protection, PII handling, consent" → covers "Privacy"

4. Do NOT award credit for fluency, length, formatting, or general relevance alone.
   A response that is well-written but omits key concepts must be scored accordingly.

5. Additional valid details beyond the ground truth do NOT penalize the score.

SCORING RUBRIC:
  1.0  — All concepts COVERED
  0.8  — All concepts COVERED or PARTIAL, none MISSING or CONTRADICTED
  0.6  — Most concepts COVERED, 1–2 PARTIAL or MISSING
  0.4  — Half or fewer concepts COVERED, several MISSING
  0.2  — Few concepts COVERED, most MISSING or CONTRADICTED
  0.0  — No concepts COVERED, or response is entirely unrelated

Rules:
- Match by MEANING, not exact keywords. "Bias mitigation" → covers "Fairness".
  "Model documentation and interpretability techniques" → covers "Explainability".
  "Clear disclosures to stakeholders" → covers "Transparency".
  "Designated ownership, governance council, escalation" → covers "Accountability".
  "Incident response, resilience, graceful degradation" → partially covers "Robustness".
  "Data protection, PII handling, consent" → covers "Privacy".

6. Do NOT award credit for fluency, length, formatting, or general relevance alone.
   A response that is well-written but omits key concepts must be scored accordingly.

7. Additional valid details beyond the ground truth do NOT penalize the score.

8. If a concept like "Privacy" is not substantively addressed, mark it MISSING
   even if the response is otherwise excellent.

Reply in EXACTLY this format with no other text before or after:
SCORE: <float between 0.0 and 1.0>
CONFIDENCE: <float between 0.0 and 1.0>
REASONING: <one sentence listing the number of COVERED, PARTIAL, MISSING, and CONTRADICTED concepts>
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
        """Parse the LLM evaluation result.

        Returns score=0.0 and confidence=0.0 when the LLM response cannot be
        parsed, so that a parse failure is not mistaken for a neutral 0.5 score.
        """
        import math

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

        # Clamp non-finite values (NaN, Infinity) to safe defaults
        if not math.isfinite(score):
            score = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0

        return score, confidence, reasoning
