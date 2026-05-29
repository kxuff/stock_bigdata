# End-to-End Local Runbook

Run local stock platform from data collection/backfill to ORCA advisory output.

This runbook assumes dev/local Docker Compose. It uses 3 symbols for faster demo:

```text
AAPL,MSFT,NVDA
```

Remove `US_STOCK_EOD_SYMBOLS` to run all default symbols.

## 0. Repositories

Expected folders:

```text
/home/ming/Desktop/bigdata/stock_bigdata
/home/ming/Desktop/bigdata/orca-agent-advisory
```

## 1. Start Docker stack

From `stock_bigdata`:

```bash
cd /home/ming/Desktop/bigdata/stock_bigdata
docker compose up -d
docker compose ps
```

Required services:

```text
airflow-webserver
airflow-scheduler
spark-master
spark-worker
minio
nessie
postgres-airflow
broker
```

## 2. First run: build history + predictions + ORCA context

First run must use initial backfill because feature engineering needs enough lookback history.

```bash
cd /home/ming/Desktop/bigdata/stock_bigdata

docker compose exec -T \
  -e PYTHONPATH='/opt/airflow/plugins' \
  -e US_STOCK_SPARK_EXECUTOR_MEMORY='1g' \
  -e US_STOCK_SPARK_EXECUTOR_CORES='1' \
  -e US_STOCK_SPARK_CORES_MAX='1' \
  -e US_STOCK_EOD_SYMBOLS='AAPL,MSFT,NVDA' \
  -e US_STOCK_INITIAL_LOAD='true' \
  -e US_STOCK_BACKFILL_CALENDAR_DAYS='500' \
  -e FINBERT_API_URL='https://parrot-sublease-preamble.ngrok-free.dev' \
  -e FINBERT_API_TIMEOUT='3' \
  -e FINBERT_FAILURE_LIMIT='3' \
  airflow-webserver python /opt/airflow/plugins/eod_inference/run_eod_pipeline.py \
  --run-date 2026-05-29
```

What this command does:

```text
extract_eod_prices
→ clean_validate_prices
→ engineer_features
→ run_ml_inference
→ save_predictions
```

`run_ml_inference()` auto-builds sentiment and valuation context when missing, then writes ORCA upstream context.

Expected output shape:

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
  "prediction_table": "nessie.ml_ready.stock_predictions"
}
```

Host path for advisory input:

```text
/home/ming/Desktop/bigdata/stock_bigdata/data/eod_batch/staging/20260529/orca_upstream.json
```

## 3. Later runs: incremental mode

After history exists, remove initial-load flags:

```bash
cd /home/ming/Desktop/bigdata/stock_bigdata

docker compose exec -T \
  -e PYTHONPATH='/opt/airflow/plugins' \
  -e US_STOCK_SPARK_EXECUTOR_MEMORY='1g' \
  -e US_STOCK_SPARK_EXECUTOR_CORES='1' \
  -e US_STOCK_SPARK_CORES_MAX='1' \
  -e US_STOCK_EOD_SYMBOLS='AAPL,MSFT,NVDA' \
  -e FINBERT_API_URL='https://parrot-sublease-preamble.ngrok-free.dev' \
  -e FINBERT_API_TIMEOUT='3' \
  -e FINBERT_FAILURE_LIMIT='3' \
  airflow-webserver python /opt/airflow/plugins/eod_inference/run_eod_pipeline.py \
  --run-date 2026-05-29
```

If incremental run fails with not enough lookback history, rerun first-run command with:

```bash
-e US_STOCK_INITIAL_LOAD='true'
-e US_STOCK_BACKFILL_CALENDAR_DAYS='500'
```

## 4. Run ORCA advisory

From `orca-agent-advisory`:

```bash
cd /home/ming/Desktop/bigdata/orca-agent-advisory

uv run python tools/run_upstream_advisory.py \
  --request samples/normal_request.json \
  --upstream /home/ming/Desktop/bigdata/stock_bigdata/data/eod_batch/staging/20260529/orca_upstream.json \
  --source-ref 'nessie.ml_ready.stock_predictions:2026-05-29' \
  --output-dir outputs/advisory_decisions
```

Expected output fields:

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

Output file is written under:

```text
/home/ming/Desktop/bigdata/orca-agent-advisory/outputs/advisory_decisions
```

## 5. Optional: run all default symbols

Remove this line from EOD command:

```bash
-e US_STOCK_EOD_SYMBOLS='AAPL,MSFT,NVDA'
```

This is slower, but closer to full demo scope.

## 6. Optional: no FinBERT server

If FinBERT/ngrok is unavailable, remove:

```bash
-e FINBERT_API_URL='https://parrot-sublease-preamble.ngrok-free.dev'
```

Sentiment falls back to lexical scoring. Output can still include `sentiment_snapshot`, but source refs will not cite FinBERT.

## 7. Troubleshooting

### Spark executor stuck / no resources

Restart Spark services:

```bash
cd /home/ming/Desktop/bigdata/stock_bigdata
docker compose restart spark-master spark-worker
```

Then smoke test:

```bash
docker compose exec -T \
  -e PYTHONPATH='/opt/airflow/plugins' \
  -e US_STOCK_SPARK_EXECUTOR_MEMORY='1g' \
  -e US_STOCK_SPARK_EXECUTOR_CORES='1' \
  -e US_STOCK_SPARK_CORES_MAX='1' \
  airflow-webserver python - <<'PY'
from eod_inference.iceberg import build_spark, stop_spark

spark = build_spark()
try:
    print('count', spark.range(5).count())
finally:
    stop_spark(spark)
PY
```

Expected:

```text
count 5
```

### Missing sentiment/valuation in ORCA context

Check output contains:

```text
sentiment_snapshot
valuation_snapshot
```

If missing, verify current branch/code has:

```text
airflow/plugins/eod_inference/agent_context.py
airflow/plugins/eod_inference/orca_context.py
airflow/plugins/eod_inference/run_eod_pipeline.py
```

### Not enough lookback history

Error shape:

```text
Not enough lookback history for feature inference. Required 260
```

Fix: run initial load again:

```bash
-e US_STOCK_INITIAL_LOAD='true'
-e US_STOCK_BACKFILL_CALENDAR_DAYS='500'
```

### Airflow webserver unhealthy

For manual `docker compose exec`, webserver health can be unhealthy while Python commands still work. If commands fail, restart:

```bash
docker compose restart airflow-webserver airflow-scheduler
```

## 8. Quick validation commands

```bash
cd /home/ming/Desktop/bigdata/stock_bigdata
python -m compileall airflow/plugins/eod_inference spark_jobs
```

```bash
cd /home/ming/Desktop/bigdata/orca-agent-advisory
uv run pytest
```
