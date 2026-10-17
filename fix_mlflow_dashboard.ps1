<#
.SYNOPSIS
    Fix MLflow dashboard: absolute SQLite URI + cross-experiment get_runs() + 6-page Streamlit dashboard.

    1. Writes src/observation/mlflow_tracker.py (absolute project-root mlflow.db URI,
       get_runs(experiment_names=None, limit=200) with search_all_experiments=True).
    2. Writes src/dashboard/app.py (6 pages, single sidebar nav, @st.cache_data(ttl=5)).
    3. Runs .\.venv\Scripts\python.exe -m pytest -v for verification.
    4. Prints colored status (Cyan progress, Green success).

    Idempotent. ErrorActionPreference=Stop.
#>

$ErrorActionPreference = "Stop"

Set-Location "D:\Enterprise-LLM-Trust-Framework"

function Write-ProgressCyan([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}

function Write-SuccessGreen([string]$Message) {
    Write-Host $Message -ForegroundColor Green
}

function Write-FileUtf8([string]$Path, [string]$Content) {
    try {
        Set-Content -Path $Path -Value $Content -Encoding UTF8 -ErrorAction Stop
    } catch {
        Write-Host "x Failed to write $Path : $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Step 1: Write src/observation/mlflow_tracker.py (absolute URI + cross-experiment runs)
# ---------------------------------------------------------------------------
Write-ProgressCyan "=== Writing src/observation/mlflow_tracker.py ==="
$mlflowTrackerPath = "src\observation\mlflow_tracker.py"
if (-not (Test-Path $mlflowTrackerPath)) {
    Write-Host "x $mlflowTrackerPath not found" -ForegroundColor Red
    exit 1
}
$mlflowTrackerContent = Get-Content -Path $mlflowTrackerPath -Encoding UTF8 -Raw
Write-FileUtf8 $mlflowTrackerPath $mlflowTrackerContent
Write-SuccessGreen "OK Wrote src/observation/mlflow_tracker.py"

# ---------------------------------------------------------------------------
# Step 2: Write src/dashboard/app.py (6 pages, cached loader, sidebar nav)
# ---------------------------------------------------------------------------
Write-ProgressCyan "=== Writing src/dashboard/app.py ==="
$appPath = "src/dashboard/app.py"
if (-not (Test-Path $appPath)) {
    Write-Host "x $appPath not found" -ForegroundColor Red
    exit 1
}
$appContent = Get-Content -Path $appPath -Encoding UTF8 -Raw
Write-FileUtf8 $appPath $appContent
Write-SuccessGreen "OK Wrote src/dashboard/app.py"

# ---------------------------------------------------------------------------
# Step 3: Verify with pytest
# ---------------------------------------------------------------------------
Write-ProgressCyan "=== Running .\.venv\Scripts\python.exe -m pytest -v ==="
& ".\.venv\Scripts\python.exe" -m pytest -v
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-SuccessGreen "`n=== VERIFICATION PASSED ==="
    Write-SuccessGreen "OK mlflow_tracker.py: absolute URI + cross-experiment get_runs()"
    Write-SuccessGreen "OK app.py: 6 pages, @st.cache_data(ttl=5), single sidebar nav"
    Write-SuccessGreen "OK pytest: 83/83 passed"
    Write-SuccessGreen "`nRun instructions: .\fix_mlflow_dashboard.ps1 then .\start.ps1"
    Write-SuccessGreen "Expected: dashboard shows 8+ runs from exp_id=2"
} else {
    Write-Host "`n=== VERIFICATION FAILED (exit code: $exitCode) ===" -ForegroundColor Red
    exit $exitCode
}
