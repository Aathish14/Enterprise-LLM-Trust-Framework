# Enterprise LLM Trust Framework

A comprehensive framework for evaluating, monitoring, and benchmarking Large Language Models (LLMs) with focus on trustworthiness, safety, and reliability.

## Overview

The Enterprise LLM Trust Framework provides tools for:
- Evaluating LLM responses across multiple dimensions (factual correctness, hallucination detection, semantic similarity, response quality, toxicity & safety)
- Computing composite trust scores from evaluation results
- Benchmarking multiple LLMs using standardized evaluation criteria
- Monitoring and tracking experiments with MLflow integration
- Visualizing results through an interactive Streamlit dashboard

## Features

- **Multi-dimensional Evaluation**: Assess LLMs across five key trust dimensions
- **Trust Scoring**: Weighted aggregation of evaluation metrics into overall trust scores
- **LLM Agnostic**: Support for OpenAI, Gemini, Claude, HuggingFace, and custom models
- **Experiment Tracking**: MLflow integration for experiment management and result tracking
- **Comparative Benchmarking**: Side-by-side comparison of multiple LLMs
- **Real-time Monitoring**: Latency tracking, failure detection, and metrics logging
- **Interactive Dashboard**: Streamlit-based visualization interface
- **Extensible Architecture**: Modular design for adding new evaluation metrics and LLM adapters

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/enterprise-llm-trust-framework.git
cd enterprise-llm-trust-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env to add your API keys
```

## Usage

### Basic Evaluation

```python
from src.core.pipeline import EvaluationPipeline
from src.llm.openai_adapter import OpenAIAdapter
from src.evaluation.factual_correctness import FactualCorrectnessService
from src.evaluation.hallucination_detection import HallucinationDetectionService
from src.core.trust_scoring import TrustScoringService

# Initialize LLM adapter
llm_adapter = OpenAIAdapter(api_key="your-openai-api-key")

# Initialize evaluation services
evaluation_services = [
    FactualCorrectnessService(llm_adapter),
    HallucinationDetectionService(llm_adapter),
    # Add other services as needed
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

print(f"Trust Score: {result['trust_score']['overall_score']}")
```

### Benchmarking Multiple LLMs

```python
from src.benchmarking.benchmark_engine import ComparativeBenchmarkEngine
from src.benchmarking.benchmark_engine import create_llm_benchmark_config
from src.llm.openai_adapter import OpenAIAdapter
from src.llm.gemini_adapter import GeminiAdapter
from src.evaluation.factual_correctness import FactualCorrectnessService
from src.evaluation.response_quality import ResponseQualityService
from src.core.trust_scoring import TrustScoringService

# Initialize LLM adapters
llm_adapters = {
    "openai-gpt4": OpenAIAdapter(api_key="your-openai-key"),
    "gemini-pro": GeminiAdapter(api_key="your-gemini-key")
}

# Initialize evaluation services
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

# Access results
for llm_name, llm_results in results.items():
    print(f"\n{llm_name} Results:")
    for result in llm_results:
        print(f"  Prompt: {result.prompt}")
        print(f"  Trust Score: {result.trust_score.overall_score:.3f}")
        print(f"  Processing Time: {result.processing_time:.2f}s")
```

### Running the Dashboard

```bash
streamlit run src/dashboard/app.py
```

## Architecture

The framework follows a modular architecture with clearly separated components:

```
src/
├── core/                 # Core interfaces and pipeline orchestration
├── evaluation/           # Individual evaluation services
├── llm/                  # LLM adapter implementations
├── observation/          # Monitoring, logging, and experiment tracking
├── benchmarking/         # Comparative benchmarking engine
├── dashboard/            # Streamlit visualization interface
└── config/               # Configuration and settings
```

Key components:
- **EvaluationPipeline**: Orchestrates the evaluation process (Generation → Evaluation → Critique → Correction → Re-evaluation)
- **TrustScoringService**: Computes weighted trust scores from evaluation results
- **LLM Adapters**: Standardized interface for different LLM providers
- **Benchmarking Engine**: Runs comparative benchmarks across multiple LLMs
- **Observation Module**: MLflow integration, experiment tracking, metrics logging
- **Dashboard**: Interactive visualization interface

## Configuration

Configuration is managed through Pydantic settings in `config/settings.py`:

```python
from config.settings import settings

# Access settings
api_host = settings.api_host
api_port = settings.api_port
debug = settings.debug
```

Environment variables can be used to override settings:
- `OPENAI_API_KEY`: OpenAI API key
- `GEMINI_API_KEY`: Gemini API key
- `ANTHROPIC_API_KEY`: Anthropic (Claude) API key
- `HF_API_KEY`: HuggingFace API key
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test modules
python -m pytest tests/unit/evaluation/
python -m pytest tests/integration/
```

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t enterprise-llm-trust-framework .

# Run container
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env enterprise-llm-trust-framework
```

### Kubernetes Deployment

See `deployment/kubernetes/` for Helm charts and Kubernetes manifests.

## API Documentation

Auto-generated API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The LangChain team for their excellent LLM orchestration library
- The MLflow team for experiment tracking capabilities
- The Streamlit team for the powerful dashboard framework
- The open-source LLM community for continued innovation