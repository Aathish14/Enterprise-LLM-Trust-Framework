<#
.SYNOPSIS
    Fix MLflow run management to prevent orphaned ACTIVE runs
    1. mlflow_integration.py: start_run() auto-closes orphaned runs
    2. app.py: run_pipeline_task() uses try/finally for guaranteed cleanup
#>

$ErrorActionPreference = "Stop"
Set-Location "D:\Enterprise-LLM-Trust-Framework"

Write-Host "=== Fixing MLflow run management ===" -ForegroundColor Cyan

# =============================================================================
# FIX 1: mlflow_integration.py - start_run() checks for and closes orphaned runs
# =============================================================================
$mlflowPath = "src\observation\mlflow_integration.py"
$mlflowContent = [System.IO.File]::ReadAllText($mlflowPath, [System.Text.Encoding]::UTF8)

# Update start_run method to check for and close active runs
$oldStartRun = @'
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None, nested: bool = False):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to apply to the run
            nested: Whether to start a nested run
            
        Returns:
            Context manager for the MLflow run
        """
        try:
            # Start MLflow run
            run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
            
            logger.info(f"Started MLflow run: {run.info.run_id}")
            # Return a context manager that provides access to self for logging
            return self._MLflowRunContext(run, self)
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise
'@

$newStartRun = @'
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None, nested: bool = False):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to apply to the run
            nested: Whether to start a nested run
            
        Returns:
            Context manager for the MLflow run
        """
        try:
            # Check for and close any orphaned active runs before starting a new one
            # (unless nested=True, which allows nested runs within an active parent)
            if not nested:
                active_run = mlflow.active_run()
                if active_run:
                    logger.warning(f"Found orphaned active run {active_run.info.run_id}, closing it before starting new run")
                    try:
                        mlflow.end_run(status="KILLED")
                        logger.info("Successfully closed orphaned run")
                    except Exception as e:
                        logger.error(f"Failed to close orphaned run: {e}")
            
            # Start MLflow run
            run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
            
            logger.info(f"Started MLflow run: {run.info.run_id}")
            # Return a context manager that provides access to self for logging
            return self._MLflowRunContext(run, self)
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise
'@

$mlflowContent = $mlflowContent -replace [regex]::Escape($oldStartRun), $newStartRun
[System.IO.File]::WriteAllText($mlflowPath, $mlflowContent, [System.Text.Encoding]::UTF8)
Write-Host "✓ Updated mlflow_integration.py start_run()" -ForegroundColor Green

# =============================================================================
# FIX 2: app.py - run_pipeline_task() uses try/finally for guaranteed cleanup
# =============================================================================
$appPath = "src\dashboard\app.py"
$appContent = [System.IO.File]::ReadAllText($appPath, [System.Text.Encoding]::UTF8)

$oldRunPipeline = @'
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
    return res
'@

$newRunPipeline = @'
async def run_pipeline_task(prompt_text, context_text, model, temperature, max_tokens):
    # Create appropriate adapter based on model
    llm_adapter = _create_llm_adapter(model, temperature, max_tokens)
    evaluator = FactualCorrectnessEvaluator(llm_adapter=llm_adapter)
    scorer = WeightedTrustScoringService()
    pipeline = EvaluationPipeline(llm_adapter=llm_adapter, evaluation_services=[evaluator], trust_scoring_service=scorer)
    
    exp_id = experiment_tracker.create_experiment("llm_trust_framework", "Production Benchmark")
    run_id = experiment_tracker.start_run(exp_id, "ui_live_eval", {"model": model})
    
    try:
        res = await pipeline.run_evaluation(prompt=prompt_text, context={"expected_facts": context_text})
        metrics_logger.log_evaluation_pipeline_metrics(res, run_id)
        return res
    except Exception as e:
        # Log the error to MLflow before ending run
        logger.error(f"Evaluation failed: {e}")
        try:
            mlflow.log_param("error", str(e))
            mlflow.set_tag("status", "FAILED")
        except Exception:
            pass  # Best effort logging
        raise
    finally:
        # GUARANTEED cleanup - always end the run and MLflow run
        experiment_tracker.end_run(run_id)
        # Also ensure MLflow run is closed (safety net)
        try:
            if mlflow.active_run():
                mlflow.end_run(status="FINISHED")
        except Exception:
            pass  # Best effort cleanup
'@

$appContent = $appContent -replace [regex]::Escape($oldRunPipeline), $newRunPipeline
[System.IO.File]::WriteAllText($appPath, $appContent, [System.Text.Encoding]::UTF8)
Write-Host "✓ Updated app.py run_pipeline_task() with try/finally" -ForegroundColor Green

# =============================================================================
# VERIFY
# =============================================================================
Write-Host "`n=== Verifying syntax ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m py_compile src\observation\mlflow_integration.py
if ($LASTEXITCODE -ne 0) { Write-Host "✗ mlflow_integration.py syntax error" -ForegroundColor Red; exit 1 }
Write-Host "✓ mlflow_integration.py syntax OK" -ForegroundColor Green

& ".\.venv\Scripts\python.exe" -m py_compile src\dashboard\app.py
if ($LASTEXITCODE -ne 0) { Write-Host "✗ app.py syntax error" -ForegroundColor Red; exit 1 }
Write-Host "✓ app.py syntax OK" -ForegroundColor Green

Write-Host "`n=== Running unit tests ===" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pytest tests/unit/observation/ tests/unit/dashboard/ -v --tb=short 2>&1

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host "`n=== ALL TESTS PASSED ===" -ForegroundColor Green
} else {
    Write-Host "`n=== SOME TESTS FAILED (exit code: $exitCode) ===" -ForegroundColor Red
}
exit $exitCode