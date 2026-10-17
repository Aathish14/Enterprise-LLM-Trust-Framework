<#
.SYNOPSIS
    Fix model routing in src/dashboard/app.py to prevent 404 errors
    Routes GPT models to OpenAIAdapter, NVIDIA models to NvidiaNIMAdapter
#>

$ErrorActionPreference = "Stop"
Set-Location "D:\Enterprise-LLM-Trust-Framework"

Write-Host "=== Fixing model routing in dashboard ===" -ForegroundColor Cyan

# Read the current app.py
$content = [System.IO.File]::ReadAllText("src\dashboard\app.py", [System.Text.Encoding]::UTF8)

# 1. Add OpenAIAdapter import
$oldImports = @"
from src.llm.nvidia_nim_adapter import NvidiaNIMAdapter
from src.core.pipeline import EvaluationPipeline"
$newImports = @"
from src.llm.nvidia_nim_adapter import NvidiaNIMAdapter
from src.llm.openai_adapter import OpenAIAdapter
from src.core.pipeline import EvaluationPipeline"

$content = $content -replace [regex]::Escape($oldImports), $newImports
Write-Host "✓ Added OpenAIAdapter import" -ForegroundColor Green

# 2. Update Settings page Default Model dropdown (line ~396)
$oldSettingsDropdown = 'st.selectbox("Default Model", ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b", "gpt-4", "claude-3"], help="Default model for evaluations")'
$newSettingsDropdown = 'st.selectbox("Default Model", ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b", "meta/llama-3.3-70b-instruct", "meta/llama-3.1-8b-instruct", "gpt-4o"], help="Default model for evaluations")'

$content = $content -replace [regex]::Escape($oldSettingsDropdown), $newSettingsDropdown
Write-Host "✓ Updated Settings page model dropdown" -ForegroundColor Green

# 3. Update Playground page model dropdown (line ~434-436)
$oldPlaygroundDropdown = @'
model_choice = st.selectbox(
                "Model",
                ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b", "gpt-4", "claude-3-opus"],
                help="Select the model to evaluate"
            )'
$newPlaygroundDropdown = @'
model_choice = st.selectbox(
                "Model",
                ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b", "meta/llama-3.3-70b-instruct", "meta/llama-3.1-8b-instruct", "gpt-4o"],
                help="Select the model to evaluate"
            )'

$content = $content -replace [regex]::Escape($oldPlaygroundDropdown), $newPlaygroundDropdown
Write-Host "✓ Updated Playground page model dropdown" -ForegroundColor Green

# 4. Update run_pipeline_task to route based on model prefix
$oldPipelineTask = @'
async def run_pipeline_task(prompt_text, context_text, model, temperature, max_tokens):
    nim = NvidiaNIMAdapter(model=model)
    evaluator = FactualCorrectnessEvaluator(llm_adapter=nim)
    scorer = WeightedTrustScoringService()
    pipeline = EvaluationPipeline(llm_adapter=nim, evaluation_services=[evaluator], trust_scoring_service=scorer)
    
    exp_id = experiment_tracker.create_experiment("llm_trust_framework", "Production Benchmark")
    run_id = experiment_tracker.start_run(exp_id, "ui_live_eval", {"model": model})
    
    res = await pipeline.run_evaluation(prompt=prompt_text, context={"expected_facts": context_text})
    metrics_logger.log_evaluation_pipeline_metrics(res, run_id)
    experiment_tracker.end_run(run_id)
    return res'

$newPipelineTask = @'
def _create_llm_adapter(model: str, temperature: float, max_tokens: int):
    """Factory function to create the appropriate LLM adapter based on model name."""
    if model.startswith("gpt-"):
        # Route GPT models to OpenAIAdapter
        return OpenAIAdapter(model_name=model)
    else:
        # Route NVIDIA NIM models (nemotron, llama, etc.) to NvidiaNIMAdapter
        return NvidiaNIMAdapter(model=model, temperature=temperature, max_tokens=max_tokens)

async def run_pipeline_task(prompt_text, context_text, model, temperature, max_tokens):
    # Create appropriate adapter based on model
    llm_adapter = _create_llm_adapter(model, temperature, max_tokens)
    evaluator = FactualCorrectnessEvaluator(llm_adapter=llm_adapter)
    scorer = WeightedTrustScoringService()
    pipeline = EvaluationPipeline(llm_adapter=llm_adapter, evaluation_services=[evaluator], trust_scoring_service=scorer)
    
    exp_id = experiment_tracker.create_experiment("llm_trust_framework", "Production Benchmark")
    run_id = experiment_tracker.start_run(exp_id, "ui_live_eval", {"model": model})
    
    res = await pipeline.run_evaluation(prompt=prompt_text, context={"expected_facts": context_text})
    metrics_logger.log_evaluation_pipeline_metrics(res, run_id)
    experiment_tracker.end_run(run_id)
    return res'

$content = $content -replace [regex]::Escape($oldPipelineTask), $newPipelineTask
Write-Host "✓ Updated run_pipeline_task with model routing logic" -ForegroundColor Green

# 5. Update the playground header text
$oldHeader = 'st.markdown("Run real-time evaluations against **NVIDIA NIM models** (Nemotron Ultra, Nemotron Lightning) directly from the UI.")'
$newHeader = 'st.markdown("Run real-time evaluations against **NVIDIA NIM models** (Nemotron, Llama) or **OpenAI GPT-4o** directly from the UI.")'

$content = $content -replace [regex]::Escape($oldHeader), $newHeader
Write-Host "✓ Updated Playground header text" -ForegroundColor Green

# Write the modified file
[System.IO.File]::WriteAllText("src\dashboard\app.py", $content, [System.Text.Encoding]::UTF8)

Write-Host "`n=== Verifying syntax ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m py_compile src\dashboard\app.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Syntax OK - All changes applied successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Syntax error" -ForegroundColor Red
    exit 1
}