"""
Comparative benchmark engine for the Enterprise LLM Trust Framework.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from core.interfaces import LLMAdapter, EvaluationResult, TrustScore
from core.pipeline import EvaluationPipeline
from observation.mlflow_integration import mlflow_tracker
from observation.experiment_tracking import experiment_tracker
from observation.metrics_logger import metrics_logger
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    benchmark_id: str
    name: str
    description: str
    prompts: List[str]
    context_template: Dict[str, Any]
    llm_adapters: Dict[str, LLMAdapter]
    evaluation_services: List[Any]  # List of EvaluationService instances
    trust_scoring_service: Any  # TrustScoringService instance
    max_iterations: int = 2
    tags: Optional[Dict[str, str]] = None


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    benchmark_id: str
    llm_name: str
    prompt: str
    response: str
    evaluation_results: List[EvaluationResult]
    trust_score: TrustScore
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any]


class ComparativeBenchmarkEngine:
    """
    Engine for benchmarking multiple LLMs using the same evaluation criteria.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark engine.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        logger.info(f"Initialized ComparativeBenchmarkEngine: {config.name}")
    
    async def run_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """
        Run the comparative benchmark across all configured LLMs.
        
        Returns:
            Dictionary mapping LLM names to lists of benchmark results
        """
        benchmark_start_time = time.time()
        logger.info(f"Starting benchmark: {self.config.name}")
        
        # Start MLflow run for the entire benchmark
        with mlflow_tracker.start_run(
            run_name=f"benchmark_{self.config.name}_{int(time.time())}",
            tags={
                "benchmark_id": self.config.benchmark_id,
                "benchmark_name": self.config.name,
                ** (self.config.tags or {})
            }
        ) as benchmark_run:
            # Log benchmark configuration
            mlflow_tracker.log_params({
                "benchmark_name": self.config.name,
                "benchmark_description": self.config.description,
                "num_prompts": len(self.config.prompts),
                "num_llms": len(self.config.llm_adapters),
                "max_iterations": self.config.max_iterations
            })
            
            # Run evaluation pipeline for each LLM on each prompt
            results = {}
            
            for llm_name, llm_adapter in self.config.llm_adapters.items():
                logger.info(f"Benchmarking LLM: {llm_name}")
                results[llm_name] = []
                
                # Start experiment tracking for this LLM
                experiment_id = experiment_tracker.create_experiment(
                    name=f"{self.config.name}_{llm_name}",
                    description=f"Benchmark experiment for {llm_name} in {self.config.name}",
                    tags={
                        "benchmark_id": self.config.benchmark_id,
                        "llm_name": llm_name,
                        "benchmark_name": self.config.name
                    },
                    parameters={
                        "llm_model": getattr(llm_adapter, 'model_name', str(llm_adapter)),
                        "llm_adapter_type": llm_adapter.__class__.__name__
                    }
                )
                
                for i, prompt in enumerate(self.config.prompts):
                    prompt_start_time = time.time()
                    logger.info(f"Evaluating prompt {i+1}/{len(self.config.prompts)} for {llm_name}")
                    
                    # Start run for this prompt evaluation
                    run_name = f"prompt_{i+1}_{llm_name}"
                    run_id = experiment_tracker.start_run(
                        experiment_id=experiment_id,
                        run_name=run_name,
                        parameters={
                            "prompt_index": i,
                            "prompt_length": len(prompt),
                            "llm_name": llm_name
                        },
                        tags={
                            "prompt_index": str(i),
                            "llm_name": llm_name
                        }
                    )
                    
                    try:
                        # Create evaluation pipeline for this LLM
                        pipeline = EvaluationPipeline(
                            llm_adapter=llm_adapter,
                            evaluation_services=self.config.evaluation_services,
                            trust_scoring_service=self.config.trust_scoring_service,
                            max_iterations=self.config.max_iterations
                        )
                        
                        # Prepare context
                        context = self.config.context_template.copy()
                        context["prompt_index"] = i
                        context["prompt"] = prompt
                        
                        # Run evaluation pipeline
                        pipeline_result = await pipeline.run_evaluation(prompt, context)
                        
                        prompt_processing_time = time.time() - prompt_start_time
                        
                        # Convert pipeline result to BenchmarkResult
                        benchmark_result = BenchmarkResult(
                            benchmark_id=self.config.benchmark_id,
                            llm_name=llm_name,
                            prompt=prompt,
                            response=pipeline_result["final_response"],
                            evaluation_results=[
                                EvaluationResult(
                                    score=er["score"],
                                    confidence=er["confidence"],
                                    details=er["details"],
                                    metric_name=er["metric_name"]
                                )
                                for er in pipeline_result["evaluation_results"]
                            ],
                            trust_score=TrustScore(
                                overall_score=pipeline_result["trust_score"]["overall_score"],
                                dimension_scores=pipeline_result["trust_score"]["dimension_scores"],
                                confidence=pipeline_result["trust_score"]["confidence"],
                                details=pipeline_result["trust_score"]["details"]
                            ) if pipeline_result["trust_score"] else None,
                            processing_time=prompt_processing_time,
                            timestamp=datetime.utcnow(),
                            metadata=pipeline_result.get("metadata", {})
                        )
                        
                        results[llm_name].append(benchmark_result)
                        
                        # Log metrics to experiment tracking
                        experiment_tracker.log_metrics(
                            run_id, 
                            {
                                "processing_time": prompt_processing_time,
                                "trust_score": pipeline_result["trust_score"]["overall_score"] if pipeline_result["trust_score"] else 0.0,
                                "prompt_length": len(prompt),
                                "response_length": len(pipeline_result["final_response"])
                            }
                        )
                        
                        # Log detailed results as artifact
                        experiment_tracker.log_dict(
                            asdict(benchmark_result),
                            f"result_prompt_{i+1}.json"
                        )
                        
                        # End run successfully
                        experiment_tracker.finish_run(run_id)
                        
                    except Exception as e:
                        logger.error(f"Error benchmarking {llm_name} on prompt {i+1}: {e}")
                        # Mark run as failed
                        if 'run_id' in locals():
                            experiment_tracker.end_run(run_id, status="FAILED")
                        # Continue with next prompt
                        continue
                
                # Finish experiment for this LLM
                # (experiments are automatically finished when all runs are done)
            
            # Calculate and log benchmark summary statistics
            benchmark_total_time = time.time() - benchmark_start_time
            self._log_benchmark_summary(results, benchmark_total_time)
            
            # Log summary statistics to MLflow
            mlflow_tracker.log_metrics({
                "benchmark_total_time_seconds": benchmark_total_time,
                "avg_processing_time_per_prompt": benchmark_total_time / (len(self.config.prompts) * len(self.config.llm_adapters))
            })
            
            logger.info(f"Benchmark completed in {benchmark_total_time:.2f}s")
            return results
    
    def _log_benchmark_summary(self, 
                              results: Dict[str, List[BenchmarkResult]], 
                              total_time: float):
        """
        Log summary statistics for the benchmark.
        
        Args:
            results: Benchmark results by LLM name
            total_time: Total benchmark time in seconds
        """
        try:
            summary = {
                "benchmark_id": self.config.benchmark_id,
                "benchmark_name": self.config.name,
                "total_time_seconds": total_time,
                "llms_tested": list(results.keys()),
                "num_prompts": len(self.config.prompts),
                "llm_summaries": {}
            }
            
            for llm_name, llm_results in results.items():
                if not llm_results:
                    continue
                
                # Calculate averages
                trust_scores = [r.trust_score.overall_score for r in llm_results if r.trust_score]
                processing_times = [r.processing_time for r in llm_results]
                
                llm_summary = {
                    "num_prompts_evaluated": len(llm_results),
                    "avg_trust_score": sum(trust_scores) / len(trust_scores) if trust_scores else 0.0,
                    "avg_processing_time_seconds": sum(processing_times) / len(processing_times) if processing_times else 0.0,
                    "total_processing_time_seconds": sum(processing_times)
                }
                
                # Add dimension scores if available
                if llm_results[0].trust_score and llm_results[0].trust_score.dimension_scores:
                    dimension_averages = {}
                    for dimension in llm_results[0].trust_score.dimension_scores.keys():
                        scores = [r.trust_score.dimension_scores[dimension] for r in llm_results if r.trust_score and dimension in r.trust_score.dimension_scores]
                        dimension_averages[f"avg_{dimension}"] = sum(scores) / len(scores) if scores else 0.0
                    llm_summary.update(dimension_averages)
                
                summary["llm_summaries"][llm_name] = llm_summary
            
            # Log summary as artifact
            mlflow_tracker.log_dict(summary, "benchmark_summary.json")
            
            logger.info(f"Benchmark summary: {summary}")
            
        except Exception as e:
            logger.error(f"Failed to log benchmark summary: {e}")


# Factory functions for creating benchmark configurations
def create_llm_benchmark_config(
    benchmark_id: str,
    name: str,
    description: str,
    prompts: List[str],
    llm_adapters: Dict[str, LLMAdapter],
    evaluation_services: List[Any],
    trust_scoring_service: Any,
    context_template: Optional[Dict[str, Any]] = None,
    max_iterations: int = 2,
    tags: Optional[Dict[str, str]] = None
) -> BenchmarkConfig:
    """
    Factory function to create a benchmark configuration.
    
    Args:
        benchmark_id: Unique identifier for the benchmark
        name: Name of the benchmark
        description: Description of the benchmark
        prompts: List of prompts to evaluate
        llm_adapters: Dictionary mapping LLM names to adapter instances
        evaluation_services: List of evaluation service instances
        trust_scoring_service: Trust scoring service instance
        context_template: Template for context to pass to evaluations
        max_iterations: Maximum iterations for the evaluation pipeline
        tags: Optional tags for the benchmark
        
    Returns:
        BenchmarkConfig instance
    """
    if context_template is None:
        context_template = {}
    
    return BenchmarkConfig(
        benchmark_id=benchmark_id,
        name=name,
        description=description,
        prompts=prompts,
        context_template=context_template,
        llm_adapters=llm_adapters,
        evaluation_services=evaluation_services,
        trust_scoring_service=trust_scoring_service,
        max_iterations=max_iterations,
        tags=tags
    )