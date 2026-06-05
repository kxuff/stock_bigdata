# End-to-End Local Demo Runbook

Run the local ORCA stock demo from EOD ingestion to advisory UI.

Default demo symbols:

```text
AAPL,MSFT,NVDA
```

Default run date:

```text
2026-05-29
```

## 1. Requirements

- Docker Desktop with Compose.
- Host Python for root tests.
- `uv` for ORCA tests.
- Live FinBERT HTTP API. `FINBERT_API_URL` is required; there is no offline lexical fallback.
- Live ORCA LLM gateway key through `NINEROUTER_KEY`.

## 2. One-Command Demo, Windows PowerShell

From the repository root:

```powershell
$env:FINBERT_API_URL='https://your-finbert-url'
$env:NINEROUTER_KEY='sk-...'

.\scripts\reset_and_run_local.ps1 `
  -ResetDocker `
  -RemoveLocalData `
  -FinbertUrl $env:FINBERT_API_URL `
  -NinerouterKey $env:NINEROUTER_KEY
```

What the script does:

```text
1. Keeps model artifacts under data/models.
2. Deletes only generated demo data: data/eod_batch, airflow/logs, ivy2.
3. Starts Docker services.
4. Checks FinBERT /health and fails early if unavailable.
5. Checks the ORCA LLM key and fails early if missing.
6. Runs EOD initial load for AAPL,MSFT,NVDA on 2026-05-29.
7. Starts orca-api and orca-worker.
8. Verifies /healthz, /api/v1/status, /api/v1/data/readiness, /api/v1/data/coverage, and /api/v1/advisory/picks.
```

Override demo inputs when needed:

```powershell
.\scripts\reset_and_run_local.ps1 `
  -FinbertUrl $env:FINBERT_API_URL `
  -NinerouterKey $env:NINEROUTER_KEY `
  -Symbols 'AAPL,MSFT,NVDA' `
  -RunDate '2026-05-29'
```

## 3. Bash Equivalent

```bash
export FINBERT_API_URL='https://your-finbert-url'
export NINEROUTER_KEY='sk-...'

docker compose up -d --build

curl -H 'ngrok-skip-browser-warning: 1' "$FINBERT_API_URL/health"

docker compose exec -T \
  -e PYTHONPATH='/opt/airflow/plugins' \
  -e US_STOCK_EOD_DATA_DIR='/opt/airflow/data/eod_batch' \
  -e US_STOCK_SPARK_EXECUTOR_MEMORY='1g' \
  -e US_STOCK_SPARK_EXECUTOR_CORES='1' \
  -e US_STOCK_SPARK_CORES_MAX='1' \
  -e US_STOCK_EOD_SYMBOLS='AAPL,MSFT,NVDA' \
  -e US_STOCK_INITIAL_LOAD='true' \
  -e US_STOCK_BACKFILL_CALENDAR_DAYS='500' \
  -e FINBERT_API_URL="$FINBERT_API_URL" \
  -e FINBERT_API_TIMEOUT='10' \
  airflow-webserver python /opt/airflow/plugins/eod_inference/run_eod_pipeline.py --run-date 2026-05-29

docker compose up -d --build orca-api orca-worker
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/api/v1/status
curl 'http://127.0.0.1:8000/api/v1/data/readiness?symbols=AAPL&decision_mode=single_symbol_advisory'
curl 'http://127.0.0.1:8000/api/v1/data/coverage?symbols=AAPL,MSFT,NVDA'
curl 'http://127.0.0.1:8000/api/v1/advisory/picks?limit=25&min_pred_a=0.06&max_risk_prob=0.3'
```

## 4. EOD Pipeline Output

Pipeline stages:

```text
extract_eod_prices
-> clean_validate_prices
-> engineer_features
-> run_ml_inference
-> save_predictions
```

Expected manifest shape:

```json
{
  "prediction_rows": 3,
  "orca_context_rows": 3,
  "orca_context_includes": [
    "market_features",
    "ml_predictions",
    "risk_snapshot",
    "sentiment_snapshot",
    "valuation_snapshot"
  ],
  "orca_context_excludes": [
    "portfolio_snapshot"
  ],
  "sentiment_rows": 3,
  "valuation_rows": 3,
  "orca_upstream_context": "/opt/airflow/data/eod_batch/staging/20260529/orca_upstream.json",
  "prediction_table": "nessie.ml_ready.stock_predictions_v2"
}
```

Host-visible staging path:

```text
data/eod_batch/staging/20260529
```

## 5. Iceberg Tables

```text
nessie.ml_ready.stock_predictions_v2
nessie.ml_ready.stock_price_features
nessie.curated.us_stock_eod_prices
nessie.ml_ready.stock_sentiment_context
nessie.ml_ready.stock_valuation_context
```

ORCA reads Iceberg/Nessie/MinIO directly. The removed deterministic `orca_upstream.json` CLI path is not used by runtime advisory APIs.

## 6. ORCA Runtime Env

`docker-compose.yml` sets the local demo defaults:

```env
ORCA_TOOL_RESULT_PROVIDER=bigdata
ORCA_ICEBERG_CATALOG=nessie
ORCA_SPARK_MASTER=local[2]
ORCA_ML_PREDICTION_TABLE=ml_ready.stock_predictions_v2
ORCA_ML_FEATURE_TABLE=ml_ready.stock_price_features
ORCA_CURATED_PRICE_TABLE=curated.us_stock_eod_prices
ORCA_SENTIMENT_TABLE=ml_ready.stock_sentiment_context
ORCA_VALUATION_TABLE=ml_ready.stock_valuation_context
```

## 7. Public Demo APIs

```text
GET /api/v1/data/readiness?symbols=AAPL&decision_mode=single_symbol_advisory
GET /api/v1/data/coverage?symbols=AAPL,MSFT,NVDA
GET /api/v1/advisory/picks?limit=25&min_pred_a=0.06&max_risk_prob=0.3
GET /api/v1/advisory/picks/AAPL
POST /api/v1/advisory/decision
POST /api/v1/agent/query-jobs
```

## 8. Streamlit

```powershell
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

AI Chat checks ORCA API health and data coverage before submitting. AI Stock Picks prefers `/api/v1/advisory/picks` and falls back to local parquet only for dev/offline inspection.

## 9. Demo Checklist

- Docker stack is up.
- FinBERT `/health` returns OK.
- EOD initial load completed for `AAPL,MSFT,NVDA`.
- `/healthz` and `/api/v1/status` return OK.
- `/api/v1/data/readiness` returns ready for at least `AAPL`.
- `/api/v1/data/coverage` reports market, ML, and risk ready for demo symbols.
- `/api/v1/advisory/picks` returns ranked rows or a clear no-prediction warning.
- Streamlit AI Chat can submit an ORCA job for a ready symbol.
- Streamlit AI Stock Picks can create an ORCA agent query job from a pick.

## 10. Known Limitations

- Streaming `ml.stock_predictions` remains read-only in this phase; there is no realtime ML prediction writer yet.
- Production auth, decision history UI, and realtime prediction writing are deferred.
- Demo requires live FinBERT and live LLM gateway credentials.

## 11. Troubleshooting

Spark executor stuck or no resources:

```powershell
docker compose restart spark-master spark-worker
```

Quick Spark smoke test:

```powershell
docker compose exec -T `
  -e PYTHONPATH='/opt/airflow/plugins' `
  -e US_STOCK_SPARK_EXECUTOR_MEMORY='1g' `
  -e US_STOCK_SPARK_EXECUTOR_CORES='1' `
  -e US_STOCK_SPARK_CORES_MAX='1' `
  airflow-webserver python -c "from eod_inference.iceberg import build_spark, stop_spark; s=build_spark(); print('count', s.range(5).count()); stop_spark(s)"
```
