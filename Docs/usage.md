# Enterprise LLM Trust Framework - Usage Guide

This guide provides instructions on how to use the Enterprise LLM Trust Framework for evaluating, monitoring, and benchmarking Large Language Models.

## Table of Contents
1. [Installation](#installation)
2. [Basic Evaluation](#basic-evaluation)
3. [Benchmarking Multiple LLMs](#benchmarking-multiple-llms)
4. [Using the Dashboard](#using-the-dashboard)
5. [Configuration](#configuration)
6. [Extending the Framework](#extending-the-framework)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Aathish14/Enterprise-LLM-Trust-Framework.git
   cd Enterprise-LLM-Trust-Framework
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys for OpenAI, Gemini, Claude, etc.
   ```

## Basic Evaluation

The framework provides an evaluation pipeline that runs LLMs through multiple evaluation dimensions and computes a trust score.

### Example: Evaluating a Single Prompt

```python
import asyncio
from src.core.pipeline import EvaluationPipeline
from src.llm.openai_adapter import OpenAIAdapter
from src.evaluation.factual_correctness import FactualCorrectnessService
from src.evaluation.hallucination_detection import HallucinationDetectionService
from src.evaluation.response_quality import ResponseQualityService
from src.core.trust_scoring import TrustScoringService

async def main():
    # Initialize LLM adapter
    llm_adapter = OpenAIAdapter(api_key="your-openai-api-key")
    
    # Initialize evaluation services
    evaluation_services = [
        FactualCorrectnessService(llm_adapter),
        HallucinationDetectionService(llm_adapter),
        ResponseQualityService(llm_adapter)
    ]
    
    # Initialize trust scoring service
    trust_scoring_service = TrustScoringService()
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(
        llm_adapter=llm_adapter,
        evaluation_services=evaluation_services,
        trust_scoring_service=trust_scoring_service,
        max_iterations=2
    )
    
    # Run evaluation
    result = await pipeline.run_evaluation(
        prompt="What is the capital of France?",
        context={"task": "geography"}
    )
    
    print(f"Response: {result['final_response']}")
    print(f"Trust Score: {result['trust_score']['overall_score']:.3f}")
    print(f"Confidence: {result['trust_score']['confidence']:.3f}")
    
    # Print individual evaluation scores
    for eval_result in result['evaluation_results']:
        print(f"{eval_result['metric_name']}: {eval_result['score']:.3f} "
              f"(confidence: {eval_result['confidence']:.3f})")

if __name__ == "__main__":
    asyncio.run(main())
```

### Expected Output
```
Response: The capital of France is Paris.
Trust Score: 0.87
Confidence: 0.92
factual_correctness: 0.95 (confidence: 0.90)
hallucination_detection: 0.85 (confidence: 0.88)
response_quality: 0.82 (confidence: 0.95)
```

## Benchmarking Multiple LLMs

The framework allows you to benchmark multiple LLMs using the same evaluation criteria.

### Example: Comparing OpenAI and Gemini

```python
import asyncio
from src.benchmarking.benchmark_engine import ComparativeBenchmarkEngine
from src.benchmarking.benchmark_engine import create_llm_benchmark_config
from src.llm.openai_adapter import OpenAIAdapter
from src.llm.gemini_adapter import GeminiAdapter
from src.evaluation.factual_correctness import FactualCorrectnessService
from src.evaluation.response_quality import ResponseQualityService
from src.core.trust_scoring import TrustScoringService

async def main():
    # Initialize LLM adapters
    llm_adapters = {
        "openai-gpt4": OpenAIAdapter(api_key="your-openai-key"),
        "gemini-pro": GeminiAdapter(api_key="your-gemini-key")
    }
    
    # Initialize evaluation services (using the first LLM adapter for simplicity)
    evaluation_services = [
        FactualCorrectnessService(llm_adapters["openai-gpt4"]),
        ResponseQualityService(llm_adapters["openai-gpt4"])
    ]
    
    # Initialize trust scoring service
    trust_scoring_service = TrustScoringService()
    
    # Create benchmark configuration
    config = create_llm_benchmark_config(
        benchmark_id="llm_comparison_001",
        name="LLM Comparison Benchmark",
        description="Comparing GPT-4 and Gemini Pro on factual correctness and response quality",
        prompts=[
            "What are the health benefits of regular exercise?",
            "Explain the concept of quantum entanglement in simple terms.",
            "How does photosynthesis work in plants?"
        ],
        llm_adapters=llm_adapters,
        evaluation_services=evaluation_services,
        trust_scoring_service=trust_scoring_service,
        max_iterations=2
    )
    
    # Run benchmark
    engine = ComparativeBenchmarkEngine(config)
    results = await engine.run_benchmark()
    
    # Display results
    print("\n=== Benchmark Results ===")
    for llm_name, llm_results in results.items():
        print(f"\n{llm_name}:")
        avg_trust_score = sum(r.trust_score.overall_score for r in llm_results) / len(llm_results)
        avg_time = sum(r.processing_time for r in llm_results) / len(llm_results)
        print(f"  Average Trust Score: {avg_trust_score:.3f}")
        print(f"  Average Processing Time: {avg_time:.2f}s")
        
        # Show per-prompt results
        for i, result in enumerate(llm_results):
            print(f"  Prompt {i+1}: Trust Score = {result.trust_score.overall_score:.3f}, "
                  f"Time = {result.processing_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

## Using the Dashboard

The framework includes a Streamlit dashboard for visualizing evaluation results and monitoring experiments.

### Starting the Dashboard

```bash
streamlit run src/dashboard/app.py
```

### Dashboard Features

1. **Overview**: Summary of recent experiments and key metrics
2. **Experiments**: Detailed view of individual experiment runs
3. **Model Comparison**: Side-by-side comparison of multiple LLMs
4. **Metrics**: Time-series charts of latency, trust scores, and other metrics
5. **Settings**: Configuration options for the framework

## Configuration

The framework uses Pydantic settings for configuration management. Settings can be overridden using environment variables.

### Key Settings

- `OPENAI_API_KEY`: API key for OpenAI services
- `GEMINI_API_KEY`: API key for Gemini services
- `ANTHROPIC_API_KEY`: API key for Anthropic (Claude) services
- `HF_API_KEY`: API key for HuggingFace services
- `MLFLOW_TRACKING_URI`: URI for MLflow tracking server (defaults to local SQLite)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `API_HOST`: Host for the API server (default: 0.0.0.0)
- `API_PORT`: Port for the API server (default: 8000)

### Example .env File
```
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HF_API_KEY=your_huggingface_api_key_here
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
```

## Extending the Framework

### Adding a New Evaluation Service

1. Create a new class that inherits from `EvaluationService`
2. Implement the `evaluate` method
3. Register the service in the evaluation pipeline

```python
from src.core.interfaces import EvaluationService, EvaluationResult
from src.llm.base_adapter import BaseLLMAdapter

class CustomEvaluationService(EvaluationService):
    def __init__(self, llm_adapter: BaseLLMAdapter):
        self.llm_adapter = llm_adapter
    
    async def evaluate(self, prompt: str, context: dict, response: str) -> EvaluationResult:
        # Your custom evaluation logic here
        score = 0.8  # Placeholder
        return EvaluationResult(
            score=score,
            confidence=0.9,
            details={"reasoning": "Custom evaluation completed"},
            metric_name="custom_metric"
        )
```

### Adding a New LLM Adapter

1. Create a new class that inherits from `BaseLLMAdapter`
2. Implement the `generate` and `get_model_info` methods
3. Register the adapter in your benchmark configuration

```python
from src.llm.base_adapter import BaseLLMAdapter

class CustomLLMAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, model_name: str = "custom-model"):
        super().__init__(api_key, model_name)
    
    async def generate(self, prompt: str, context: dict = None) -> str:
        # Your custom LLM generation logic here
        return "This is a response from the custom LLM."
    
    def get_model_info(self) -> dict:
        return {
            "provider": "custom",
            "model_name": self.model_name,
            "version": "1.0.0"
        }
```

## API Reference

For detailed API documentation, please refer to the [API Documentation](api/modules.html) generated by Sphinx.

## Support and Troubleshooting

If you encounter issues:
1. Check the logs in the `logs/` directory
2. Ensure all required API keys are set in your `.env` file
3. Verify that you have internet connectivity for API calls
4. Consult the FAQ section in the GitHub repository
5. Open an issue on the GitHub repository if needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.