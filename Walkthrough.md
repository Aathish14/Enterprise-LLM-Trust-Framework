# Enterprise LLM Trust Framework - Walkthrough

This walkthrough provides a step-by-step guide to using the Enterprise LLM Trust Framework for evaluating, monitoring, and benchmarking Large Language Models.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Running Your First Evaluation](#running-your-first-evaluation)
3. [Benchmarking Multiple LLMs](#benchmarking-multiple-llms)
4. [Using the Dashboard](#using-the-dashboard)
5. [Extending the Framework](#extending-the-framework)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- API keys for at least one LLM provider (OpenAI, Gemini, Claude, or HuggingFace)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Aathish14/Enterprise-LLM-Trust-Framework.git
cd Enterprise-LLM-Trust-Framework
```

### Step 2: Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HF_API_KEY=your_huggingface_api_key_here
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
LOG_LEVEL=INFO
```

## Running Your First Evaluation

Let's walk through a basic evaluation of a single prompt using OpenAI's GPT-4.

### Step 1: Create a Python Script
Create a file called `first_evaluation.py`:

```python
import asyncio
from src.core.pipeline import EvaluationPipeline
from src.llm.openai_adapter import OpenAIAdapter
from src.evaluation.factual_correctness import FactualCorrectnessService
from src.evaluation.hallucination_detection import HallucinationDetectionService
from src.evaluation.response_quality import ResponseQualityService
from src.core.trust_scoring import TrustScoringService

async def main():
    print("Initializing LLM adapter...")
    # Initialize LLM adapter
    llm_adapter = OpenAIAdapter(api_key="your-openai-api-key")
    
    print("Setting up evaluation services...")
    # Initialize evaluation services
    evaluation_services = [
        FactualCorrectnessService(llm_adapter),
        HallucinationDetectionService(llm_adapter),
        ResponseQualityService(llm_adapter)
    ]
    
    print("Creating trust scoring service...")
    # Initialize trust scoring service
    trust_scoring_service = TrustScoringService()
    
    print("Building evaluation pipeline...")
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(
        llm_adapter=llm_adapter,
        evaluation_services=evaluation_services,
        trust_scoring_service=trust_scoring_service,
        max_iterations=2
    )
    
    print("Running evaluation...")
    # Run evaluation
    result = await pipeline.run_evaluation(
        prompt="What is the capital of France?",
        context={"task": "geography"}
    )
    
    # Display results
    print("\n=== Evaluation Results ===")
    print(f"Prompt: What is the capital of France?")
    print(f"Response: {result['final_response']}")
    print(f"Trust Score: {result['trust_score']['overall_score']:.3f}")
    print(f"Confidence: {result['trust_score']['confidence']:.3f}")
    
    print("\n--- Individual Metrics ---")
    for eval_result in result['evaluation_results']:
        print(f"{eval_result['metric_name']}: {eval_result['score']:.3f} "
              f"(confidence: {eval_result['confidence']:.3f})")
        print(f"  Reasoning: {eval_result['details'].get('reasoning', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Run the Script
```bash
python first_evaluation.py
```

### Expected Output
```
Initializing LLM adapter...
Setting up evaluation services...
Creating trust scoring service...
Building evaluation pipeline...
Running evaluation...

=== Evaluation Results ===
Prompt: What is the capital of France?
Response: The capital of France is Paris.
Trust Score: 0.87
Confidence: 0.92

--- Individual Metrics ---
factual_correctness: 0.95 (confidence: 0.90)
  Reasoning: The response correctly identifies Paris as the capital of France.
hallucination_detection: 0.85 (confidence: 0.88)
  Reasoning: No hallucinations detected in the response.
response_quality: 0.82 (confidence: 0.95)
  Reasoning: The response is clear, concise, and directly answers the question.
```

## Benchmarking Multiple LLMs

Now let's compare two different LLMs using the same evaluation criteria.

### Step 1: Create a Benchmark Script
Create a file called `benchmark_comparison.py`:

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
    print("Setting up LLM adapters...")
    # Initialize LLM adapters
    llm_adapters = {
        "openai-gpt4": OpenAIAdapter(api_key="your-openai-key"),
        "gemini-pro": GeminiAdapter(api_key="your-gemini-key")
    }
    
    print("Setting up evaluation services...")
    # Initialize evaluation services (using the first LLM adapter for simplicity)
    evaluation_services = [
        FactualCorrectnessService(llm_adapters["openai-gpt4"]),
        ResponseQualityService(llm_adapters["openai-gpt4"])
    ]
    
    print("Creating trust scoring service...")
    # Initialize trust scoring service
    trust_scoring_service = TrustScoringService()
    
    print("Creating benchmark configuration...")
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
    
    print("Running benchmark...")
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
            print(f"  Prompt {i+1}: '{result.prompt[:50]}...'")
            print(f"    Trust Score = {result.trust_score.overall_score:.3f}")
            print(f"    Processing Time = {result.processing_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Run the Benchmark
```bash
python benchmark_comparison.py
```

### Expected Output
```
Setting up LLM adapters...
Setting up evaluation services...
Creating trust scoring service...
Creating benchmark configuration...
Running benchmark...

=== Benchmark Results ===

openai-gpt4:
  Average Trust Score: 0.84
  Average Processing Time: 1.23s
  Prompt 1: 'What are the health benefits of regular exercise?...'
    Trust Score = 0.86
    Processing Time = 1.15s
  Prompt 2: 'Explain the concept of quantum entanglement in simple terms....'
    Trust Score = 0.82
    Processing Time = 1.31s
  Prompt 3: 'How does photosynthesis work in plants?...'
    Trust Score = 0.84
    Processing Time = 1.23s

gemini-pro:
  Average Trust Score: 0.79
  Average Processing Time: 1.45s
  Prompt 1: 'What are the health benefits of regular exercise?...'
    Trust Score = 0.81
    Processing Time = 1.38s
  Prompt 2: 'Explain the concept of quantum entanglement in simple terms....'
    Trust Score = 0.76
    Processing Time = 1.52s
  Prompt 3: 'How does photosynthesis work in plants?...'
    Trust Score = 0.80
    Processing Time = 1.45s
```

## Using the Dashboard

The framework includes a Streamlit dashboard for visualizing evaluation results and monitoring experiments.

### Step 1: Start the Dashboard
```bash
streamlit run src/dashboard/app.py
```

### Step 2: Access the Dashboard
Open your web browser and navigate to: http://localhost:8501

### Step 3: Explore the Dashboard Features

1. **Overview**: See a summary of recent experiments and key metrics
2. **Experiments**: View detailed information about individual experiment runs
3. **Model Comparison**: Compare multiple LLMs side-by-side
4. **Metrics**: View time-series charts of latency, trust scores, and other metrics
5. **Settings**: Adjust configuration options for the framework

### Step 4: Run an Evaluation Through the Dashboard
1. Go to the "Experiments" tab
2. Click "New Experiment"
3. Select an LLM provider
4. Enter a prompt
5. Click "Run Experiment"
6. View the results in real-time

## Extending the Framework

### Adding a New Evaluation Service

Let's create a simple custom evaluation service that checks response length.

#### Step 1: Create the Service
Create a file called `src/evaluation/length_check.py`:

```python
from src.core.interfaces import EvaluationService, EvaluationResult
from src.llm.base_adapter import BaseLLMAdapter

class LengthCheckService(EvaluationService):
    def __init__(self, llm_adapter: BaseLLMAdapter, optimal_length: int = 100):
        self.llm_adapter = llm_adapter
        self.optimal_length = optimal_length
    
    async def evaluate(self, prompt: str, context: dict, response: str) -> EvaluationResult:
        # Simple length-based evaluation
        response_length = len(response)
        # Score based on how close to optimal length (0-1 scale)
        if response_length == 0:
            score = 0.0
        else:
            # Closer to optimal length gets higher score
            distance = abs(response_length - self.optimal_length)
            max_distance = max(response_length, self.optimal_length)
            score = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        
        return EvaluationResult(
            score=max(0.0, min(1.0, score)),  # Ensure score is between 0 and 1
            confidence=0.9,
            details={
                "reasoning": f"Response length: {response_length}, Optimal length: {self.optimal_length}",
                "response_length": response_length,
                "optimal_length": self.optimal_length
            },
            metric_name="length_check"
        )
```

#### Step 2: Use the Service in an Evaluation
Modify your `first_evaluation.py` to include the new service:

```python
# Add this import
from src.evaluation.length_check import LengthCheckService

# In the main function, after initializing other services:
print("Setting up evaluation services...")
# Initialize evaluation services
evaluation_services = [
    FactualCorrectnessService(llm_adapter),
    HallucinationDetectionService(llm_adapter),
    ResponseQualityService(llm_adapter),
    LengthCheckService(llm_adapter, optimal_length=50)  # New service
]
```

### Adding a New LLM Adapter

Let's create a simple mock LLM adapter for testing.

#### Step 1: Create the Adapter
Create a file called `src/llm/mock_adapter.py`:

```python
from src.llm.base_adapter import BaseLLMAdapter
import asyncio

class MockLLMAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str = "mock-key", model_name: str = "mock-model"):
        super().__init__(api_key, model_name)
        self.call_count = 0
    
    async def generate(self, prompt: str, context: dict = None) -> str:
        self.call_count += 1
        # Simulate some processing time
        await asyncio.sleep(0.1)
        return f"This is a mock response to: '{prompt[:30]}...' (call #{self.call_count})"
    
    def get_model_info(self) -> dict:
        return {
            "provider": "mock",
            "model_name": self.model_name,
            "version": "1.0.0"
        }
```

#### Step 2: Use the Adapter
Modify your benchmark script to use the mock adapter:

```python
# Add this import
from src.llm.mock_adapter import MockLLMAdapter

# In the main function, replace the LLM adapters initialization:
print("Setting up LLM adapters...")
# Initialize LLM adapters
llm_adapters = {
    "mock-llm": MockLLMAdapter()
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. API Key Errors
**Error**: `AuthenticationError: No API key provided`
**Solution**: 
- Verify your `.env` file contains the correct API keys
- Ensure the `.env` file is in the project root
- Check that you've activated your virtual environment before running scripts

#### 2. Import Errors
**Error**: `ModuleNotFoundError: No module named 'src'`
**Solution**:
- Make sure you're running commands from the project root directory
- Verify that the `src` directory exists and contains the Python packages
- Check that your virtual environment is activated

#### 3. Dashboard Not Loading
**Error**: Dashboard shows a blank page or connection error
**Solution**:
- Ensure Streamlit is installed: `pip install streamlit`
- Check that port 8501 is not being used by another application
- Try accessing http://localhost:8501 in your browser
- Check the terminal where you started Streamlit for error messages

#### 4. MLflow Tracking Issues
**Error**: MLflow connection errors
**Solution**:
- Verify MLflow is installed: `pip install mlflow`
- Check that the `mlflow.db` file exists in your project directory
- Ensure you have write permissions to the directory
- Try deleting `mlflow.db` and letting the framework recreate it

### Getting Help
If you encounter issues not covered here:
1. Check the logs in the `logs/` directory
2. Review the FAQ in the GitHub repository
3. Search existing issues on GitHub
4. Open a new issue with detailed information about your problem

## Next Steps

Congratulations! You've now:
1. Set up the Enterprise LLM Trust Framework
2. Run your first LLM evaluation
3. Benchmarked multiple LLMs
4. Used the dashboard for visualization
5. Extended the framework with custom components

From here, you can:
- Experiment with different LLMs and prompts
- Create custom evaluation metrics tailored to your use case
- Integrate the framework into your MLOps pipeline
- Contribute to the open-source project on GitHub

Happy evaluating!