<#
.SYNOPSIS
    Start the Enterprise LLM Trust Framework Streamlit dashboard.

    Launches src/dashboard/app.py via the project venv.
#>

$ErrorActionPreference = "Stop"

Set-Location "D:\Enterprise-LLM-Trust-Framework"

Write-Host "=== Starting Enterprise LLM Trust Framework dashboard ===" -ForegroundColor Cyan

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "x .\.venv\Scripts\python.exe not found" -ForegroundColor Red
    exit 1
}

# Ensure the dashboard and tracker are present before launching.
if (-not (Test-Path "src\dashboard\app.py")) {
    Write-Host "x src\dashboard\app.py not found. Run .\fix_mlflow_dashboard.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Launching Streamlit on http://localhost:8501 ..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m streamlit run src\dashboard\app.py --server.port 8501 --server.headless true

if ($LASTEXITCODE -ne 0) {
    Write-Host "x Streamlit exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
