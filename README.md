# Stock Big Data Local Run Guide

This repository runs a local stock data lakehouse and advisory demo: EOD ingestion, ML features/predictions, FinBERT sentiment, valuation context, Iceberg/Nessie/MinIO storage, and live ORCA advisory API.

## Requirements

- Docker Desktop / Docker Compose
- Python on host for tests
- `uv` for ORCA local development
- Running FinBERT HTTP API
- Running 9router gateway at `http://localhost:20128/v1`

Sentiment requires FinBERT. If `FINBERT_API_URL` is missing or unreachable, the EOD pipeline fails fast. There is no lexical sentiment fallback.

## Start the stack

```bash
docker compose up -d
docker compose ps
```

Core services:

```text
airflow-webserver
airflow-scheduler
spark-master
spark-worker
minio
nessie
postgres-airflow
orca-api
```

Useful URLs:

```text
Airflow: http://localhost:8085
Spark master UI: http://localhost:8080
Spark worker UI: http://localhost:8084
MinIO: http://localhost:9001
Nessie API: http://localhost:19120/api/v2
ORCA API: http://localhost:8000
```

## Streamlit placeholder UI

Run the mock-first multipage UI:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Pages include Dashboard, AI Chat, and AI Stock Picks. No backend calls run by default.

## Configure secrets

Do not commit API keys. Set the 9router key in the current shell before starting ORCA:

```powershell
$env:NINEROUTER_KEY="sk-..."
```

Linux/macOS:

```bash
export NINEROUTER_KEY="sk-..."
```

`docker-compose.yml` passes it as:

```yaml
LLM_API_KEY: ${NINEROUTER_KEY:-dummy}
```

## Check FinBERT

Replace URL with your running FinBERT endpoint:

```bash
curl -H 'ngrok-skip-browser-warning: 1' https://your-finbert-url/health
```

Expected shape:

```json
{"status":"ok","model":"ProsusAI/finbert","device":"cuda"}
```

## First EOD run

Use initial backfill on the first run because feature engineering needs lookback history.

You can run the full reset/bootstrap flow with the helper script:

```powershell
.\scripts\reset_and_run_local.ps1 `
  -ResetDocker `
  -RemoveLocalData `
  -NinerouterKey 'sk-...'
```

The script resets optional local state, starts Docker services, checks FinBERT, runs EOD initial load, and starts ORCA API. Any failed step throws and stops the script.

PowerShell:

```powershell
docker compose exec -T `
  -e PYTHONPATH='/opt/airflow/plugins' `
  -e US_STOCK_EOD_DATA_DIR='/tmp/eod_batch' `
  -e US_STOCK_SPARK_EXECUTOR_MEMORY='1g' `
  -e US_STOCK_SPARK_EXECUTOR_CORES='1' `
  -e US_STOCK_SPARK_CORES_MAX='1' `
  -e US_STOCK_EOD_SYMBOLS='AAPL' `
  -e US_STOCK_INITIAL_LOAD='true' `
  -e US_STOCK_BACKFILL_CALENDAR_DAYS='500' `
  -e FINBERT_API_URL='https://your-finbert-url' `
  -e FINBERT_API_TIMEOUT='10' `
  airflow-webserver python /opt/airflow/plugins/eod_inference/run_eod_pipeline.py --run-date 2026-05-29
```

Bash:

```bash
docker compose exec -T \
  -e PYTHONPATH='/opt/airflow/plugins' \
  -e US_STOCK_EOD_DATA_DIR='/tmp/eod_batch' \
  -e US_STOCK_SPARK_EXECUTOR_MEMORY='1g' \
  -e US_STOCK_SPARK_EXECUTOR_CORES='1' \
  -e US_STOCK_SPARK_CORES_MAX='1' \
  -e US_STOCK_EOD_SYMBOLS='AAPL' \
  -e US_STOCK_INITIAL_LOAD='true' \
  -e US_STOCK_BACKFILL_CALENDAR_DAYS='500' \
  -e FINBERT_API_URL='https://your-finbert-url' \
  -e FINBERT_API_TIMEOUT='10' \
  airflow-webserver python /opt/airflow/plugins/eod_inference/run_eod_pipeline.py --run-date 2026-05-29
```

Pipeline stages:

```text
extract_eod_prices
→ clean_validate_prices
→ engineer_features
→ run_ml_inference
→ save_predictions
```

Expected manifest fields:

```json
{
  "prediction_rows": 1,
  "orca_context_rows": 1,
  "sentiment_rows": 1,
  "valuation_rows": 1,
  "orca_context_includes": [
    "market_features",
    "ml_predictions",
    "risk_snapshot",
    "sentiment_snapshot",
    "valuation_snapshot"
  ]
}
```

## Iceberg tables written

```text
nessie.ml_ready.stock_predictions
nessie.ml_ready.stock_price_features
nessie.curated.us_stock_eod_prices
nessie.ml_ready.stock_sentiment_context
nessie.ml_ready.stock_valuation_context
```

ORCA reads these tables directly. It does not use the removed deterministic `orca_upstream.json` CLI path.

## Later EOD runs

After history exists, remove these first-run flags:

```bash
-e US_STOCK_INITIAL_LOAD='true'
-e US_STOCK_BACKFILL_CALENDAR_DAYS='500'
```

Keep `FINBERT_API_URL` set.

## Run ORCA API

ORCA is included in `docker-compose.yml` as `orca-api`.

```powershell
$env:NINEROUTER_KEY="sk-..."
docker compose up -d --build orca-api
```

The compose service uses local Spark inside the ORCA container for lower-latency Iceberg reads:

```yaml
ORCA_SPARK_MASTER: local[2]
```

Call the API:

```powershell
$body = @'
{
  "request_id": "req_live_20260529_001",
  "timestamp": "2026-05-29T23:30:00Z",
  "as_of_timestamp": "2026-05-29T23:30:00Z",
  "user_query": "Should I buy AAPL today?",
  "decision_mode": "single_symbol_advisory",
  "symbols": ["AAPL"],
  "user_context": {
    "risk_tolerance": "MODERATE",
    "investment_horizon": "SHORT_TERM",
    "target_sectors": ["Technology"],
    "excluded_symbols": [],
    "max_single_asset_weight": 40,
    "allow_cash_position": true,
    "custom_constraints": {"avoid_high_volatility": true}
  },
  "metadata": {
    "client": "web",
    "locale": "en-US",
    "as_of_date": "2026-05-29"
  }
}
'@

Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8000/api/v1/advisory/decision' `
  -Method Post `
  -ContentType 'application/json' `
  -Body $body `
  -TimeoutSec 600
```

Expected response fields:

```text
symbol
recommendation
confidence
requires_human_review
decision_rationale
supporting_signals
conflicting_signals
risk_warnings
```

ORCA live advisory can be slow because CrewAI runs multiple LLM-backed tasks.

## Run tests

ORCA provider tests:

```bash
cd orca-agent-advisory
uv run --python 3.12 pytest tests/test_bigdata_ml_provider.py
```

EOD sentiment tests:

```powershell
$env:PYTHONPATH='airflow/plugins'
python -m pytest tests/test_agent_context.py
```

Compile EOD plugins:

```bash
python -m compileall airflow/plugins/eod_inference
```
