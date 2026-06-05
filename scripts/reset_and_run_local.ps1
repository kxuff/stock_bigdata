param(
    [string]$NinerouterKey = $env:NINEROUTER_KEY,
    [string]$FinbertUrl = $env:FINBERT_API_URL,
    [string]$Symbols = $env:US_STOCK_EOD_SYMBOLS,
    [string]$Symbol = "",
    [string]$RunDate = "2026-05-29",
    [switch]$ResetDocker,
    [switch]$RemoveLocalData
)

$ErrorActionPreference = "Stop"
$DefaultSymbols = "AAPL,MSFT,NVDA"

function Step($Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Invoke-Logged($Command) {
    Write-Host "+ $Command" -ForegroundColor DarkGray
    Invoke-Expression $Command
}

function Remove-GeneratedPath($Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $ResolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    $ResolvedTarget = (Resolve-Path -LiteralPath $Path).Path
    if (-not ($ResolvedTarget.StartsWith($ResolvedRoot, [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Refusing to remove path outside repository root: $ResolvedTarget"
    }

    Write-Host "Removing $ResolvedTarget" -ForegroundColor Yellow
    Remove-Item -Recurse -Force -LiteralPath $ResolvedTarget
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

Require-Command docker

Step "Repository root: $Root"

if ([string]::IsNullOrWhiteSpace($Symbols)) {
    $Symbols = if ([string]::IsNullOrWhiteSpace($Symbol)) { $DefaultSymbols } else { $Symbol }
}

if ([string]::IsNullOrWhiteSpace($FinbertUrl)) {
    throw "FINBERT_API_URL is required. Set `$env:FINBERT_API_URL or pass -FinbertUrl."
}

if ([string]::IsNullOrWhiteSpace($NinerouterKey)) {
    throw "NINEROUTER_KEY is required. Set `$env:NINEROUTER_KEY or pass -NinerouterKey."
}

if ($ResetDocker) {
    Step "Reset Docker containers and volumes"
    Invoke-Logged "docker compose down -v --remove-orphans"
}

if ($RemoveLocalData) {
    Step "Remove local generated data"
    foreach ($Path in @("data/eod_batch", "airflow/logs", "ivy2")) {
        Remove-GeneratedPath $Path
    }
}

$env:NINEROUTER_KEY = $NinerouterKey
$env:FINBERT_API_URL = $FinbertUrl

Step "Start Docker stack"
Invoke-Logged "docker compose up -d --build"
Invoke-Logged "docker compose ps"

Step "Check FinBERT health"
try {
    $Health = Invoke-RestMethod -Uri "$FinbertUrl/health" -Headers @{ "ngrok-skip-browser-warning" = "1" } -TimeoutSec 30
    $Health | ConvertTo-Json -Depth 10
} catch {
    throw "FinBERT health check failed for $FinbertUrl. $($_.Exception.Message)"
}

Step "Run EOD initial pipeline"
$EodCommand = @(
    "docker compose exec -T",
    "-e PYTHONPATH='/opt/airflow/plugins'",
    "-e US_STOCK_EOD_DATA_DIR='/opt/airflow/data/eod_batch'",
    "-e US_STOCK_SPARK_EXECUTOR_MEMORY='1g'",
    "-e US_STOCK_SPARK_EXECUTOR_CORES='1'",
    "-e US_STOCK_SPARK_CORES_MAX='1'",
    "-e US_STOCK_EOD_SYMBOLS='$Symbols'",
    "-e US_STOCK_INITIAL_LOAD='true'",
    "-e US_STOCK_BACKFILL_CALENDAR_DAYS='500'",
    "-e FINBERT_API_URL='$FinbertUrl'",
    "-e FINBERT_API_TIMEOUT='10'",
    "airflow-webserver python /opt/airflow/plugins/eod_inference/run_eod_pipeline.py --run-date $RunDate"
) -join " "
Invoke-Logged $EodCommand

Step "Start ORCA API and worker"
Invoke-Logged "docker compose up -d --build orca-api orca-worker"

Step "Check ORCA API health"
$Healthz = Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -TimeoutSec 30
$Healthz | ConvertTo-Json -Depth 10
$Status = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/status" -TimeoutSec 30
$Status | ConvertTo-Json -Depth 10

Step "Check ORCA readiness for first demo symbol"
$FirstSymbol = ($Symbols -split ",")[0].Trim().ToUpper()
$Readiness = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/data/readiness?symbols=$FirstSymbol&decision_mode=single_symbol_advisory" -TimeoutSec 120
$Readiness | ConvertTo-Json -Depth 10

Step "Check ORCA coverage and picks"
$CoverageSymbols = [System.Uri]::EscapeDataString($Symbols)
$Coverage = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/data/coverage?symbols=$CoverageSymbols" -TimeoutSec 120
$Coverage | ConvertTo-Json -Depth 10
$Picks = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/advisory/picks?limit=25&min_pred_a=0.06&max_risk_prob=0.3" -TimeoutSec 120
$Picks | ConvertTo-Json -Depth 10

Step "Done"
