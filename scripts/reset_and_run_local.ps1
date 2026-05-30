param(
    [string]$NinerouterKey = $env:NINEROUTER_KEY,
    [string]$Symbol = "AAPL",
    [string]$RunDate = "2026-05-29",
    [switch]$ResetDocker,
    [switch]$RemoveLocalData
)

$ErrorActionPreference = "Stop"
$FinbertUrl = "https://parrot-sublease-preamble.ngrok-free.dev"

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

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

Require-Command docker

Step "Repository root: $Root"

if ($ResetDocker) {
    Step "Reset Docker containers and volumes"
    Invoke-Logged "docker compose down -v --remove-orphans"
}

if ($RemoveLocalData) {
    Step "Remove local generated data"
    foreach ($Path in @("data", "airflow/logs", "ivy2")) {
        if (Test-Path -LiteralPath $Path) {
            Write-Host "Removing $Path" -ForegroundColor Yellow
            Remove-Item -Recurse -Force -LiteralPath $Path
        }
    }
}

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
    "-e US_STOCK_EOD_DATA_DIR='/tmp/eod_batch'",
    "-e US_STOCK_SPARK_EXECUTOR_MEMORY='1g'",
    "-e US_STOCK_SPARK_EXECUTOR_CORES='1'",
    "-e US_STOCK_SPARK_CORES_MAX='1'",
    "-e US_STOCK_EOD_SYMBOLS='$Symbol'",
    "-e US_STOCK_INITIAL_LOAD='true'",
    "-e US_STOCK_BACKFILL_CALENDAR_DAYS='500'",
    "-e FINBERT_API_URL='$FinbertUrl'",
    "-e FINBERT_API_TIMEOUT='10'",
    "airflow-webserver python /opt/airflow/plugins/eod_inference/run_eod_pipeline.py --run-date $RunDate"
) -join " "
Invoke-Logged $EodCommand

if ([string]::IsNullOrWhiteSpace($NinerouterKey)) {
    throw "NINEROUTER_KEY is required. Set `$env:NINEROUTER_KEY or pass -NinerouterKey."
}

Step "Start ORCA API"
$env:NINEROUTER_KEY = $NinerouterKey
Invoke-Logged "docker compose up -d --build orca-api"

Step "Done"
