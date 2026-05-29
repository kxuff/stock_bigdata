# ORCA Agent Integration Handoff

## Status

Branch: `agent-integration`

PR branch pushed:

- `stock_bigdata`: `origin/agent-integration`
- `orca-agent-advisory`: `origin/agent-integration`

Current readiness: PR-ready / controlled rollout. Not full autonomous production yet because FinBERT is still served through a dev ngrok endpoint and yfinance is still the demo fundamentals/news provider.

## What changed

### `stock_bigdata`

The EOD batch now emits an ORCA-ready upstream context artifact:

```text
data/eod_batch/staging/<YYYYMMDD>/orca_upstream.json
```

Included contexts:

- `market_features`
- `ml_predictions`
- `risk_snapshot`
- `sentiment_snapshot` when news data exists
- `valuation_snapshot` when fundamentals data exists

Main files:

- `airflow/plugins/eod_inference/agent_context.py`
  - builds sentiment and valuation context
  - calls optional FinBERT API when `FINBERT_API_URL` is set
  - falls back to lexical sentiment if FinBERT unavailable
  - adds sentiment freshness metadata
  - adds valuation quality/method/freshness metadata
- `airflow/plugins/eod_inference/orca_context.py`
  - merges predictions, features, sentiment, and valuation into `orca_upstream.json`
  - emits explicit `sentiment_source_refs` and `valuation_source_refs`
- `airflow/plugins/eod_inference/inference.py`
  - validates Model A and Model C outputs are calibrated probabilities in `[0, 1]`
  - writes ORCA context after inference
- `notebooks/kaggle_finbert_api.ipynb`
  - dev/staging FinBERT FastAPI notebook for Kaggle/ngrok

### `orca-agent-advisory`

Main files:

- `tools/run_upstream_advisory.py`
  - converts `orca_upstream.json` rows into `ToolResultBundle`
  - uses `pred_a` as raw `probability_up`
  - treats `final_score` as risk-adjusted trend score only
  - maps source-specific freshness per tool
  - maps sentiment and valuation metadata
  - handles missing `risk_prob` with explicit neutral fallback factor
- `app/schemas/tool_results.py`
  - added optional sentiment freshness metadata
  - added optional valuation quality/method/freshness metadata
- `app/agents/sentiment_agent.py`
  - degrades stale sentiment context
- `app/agents/valuation_agent.py`
  - degrades low/unknown/stale valuation context
- `tests/test_run_upstream_advisory.py`
  - covers source refs, probability mapping, risk fallback, freshness metadata
- `tests/test_schemas.py`, `tests/test_core_agents.py`
  - cover metadata schema and valuation quality degradation

## Latest validated run

Input run date: `2026-05-26`

Latest upstream artifact:

```text
/home/ming/Desktop/bigdata/stock_bigdata/data/eod_batch/staging/20260526/orca_upstream.json
```

Latest ORCA run:

```text
run_id: run_4768126cbb224ab880449d62b3cad92a
recommendation: HOLD
confidence: 0.34
requires_human_review: true
review_reasons:
  - LOW_CONFIDENCE
  - HIGH_RISK
  - CONFLICTING_SIGNALS
```

AAPL key values:

```text
pred_a: 0.491075933
risk_prob: 0.8894067407
final_score: 0.054309688
sentiment_label: MIXED
sentiment_score: 0.289057649
valuation_label: FAIRLY_VALUED
valuation_method: analyst_target
valuation_quality: MEDIUM
sector_sample_count: 15
```

Interpretation:

- `pred_a` is near coin flip, not a strong buy signal.
- `risk_prob` is critical, so ORCA correctly requires human review.
- Sentiment is mixed even with FinBERT.
- Valuation is fair and medium quality, so it does not push a strong BUY/SELL.

## How to rerun

### 1. Start/verify Docker stack

From `stock_bigdata`:

```bash
docker compose ps
```

Required services include:

- `airflow-webserver`
- `airflow-scheduler`
- `spark-master`
- `spark-worker`
- `minio`
- `nessie`
- `postgres-airflow`

### 2. Optional FinBERT via ngrok

For dev/staging, use Kaggle notebook:

```text
notebooks/kaggle_finbert_api.ipynb
```

Set env when running agent context:

```bash
FINBERT_API_URL="https://parrot-sublease-preamble.ngrok-free.dev"
FINBERT_API_TIMEOUT=3
FINBERT_FAILURE_LIMIT=3
```

Behavior:

- if FinBERT succeeds, `source_refs` includes `finbert:ProsusAI/finbert`
- if FinBERT fails, lexical fallback is used and FinBERT is not cited

### 3. Run EOD pipeline and generate ORCA upstream context

For first local run, enable initial load/backfill so the feature job has enough lookback history:

```bash
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

Expected output includes:

```json
{
  "orca_context_includes": [
    "market_features",
    "ml_predictions",
    "risk_snapshot",
    "sentiment_snapshot",
    "valuation_snapshot"
  ],
  "orca_context_excludes": ["portfolio_snapshot"],
  "orca_upstream_context": "/opt/airflow/data/eod_batch/staging/20260529/orca_upstream.json"
}
```

After initial history exists, run incrementally without the initial-load flags:

```bash
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

To run all default symbols, remove:

```bash
-e US_STOCK_EOD_SYMBOLS='AAPL,MSFT,NVDA'
```

`run_ml_inference()` auto-builds sentiment and valuation context if the feature manifest does not already include them. `US_STOCK_SPARK_DRIVER_HOST` normally does not need to be set because the Spark driver host is auto-detected from the running Airflow container.

### 4. Run ORCA advisory

From `orca-agent-advisory`:

```bash
uv run python tools/run_upstream_advisory.py \
  --request samples/normal_request.json \
  --upstream /home/ming/Desktop/bigdata/stock_bigdata/data/eod_batch/staging/20260529/orca_upstream.json \
  --source-ref 'nessie.ml_ready.stock_predictions:2026-05-29' \
  --output-dir outputs/advisory_decisions
```

### Legacy: regenerate only ORCA upstream context

If feature and agent context manifests already exist, rerun only inference:

```bash
docker compose exec -T \
  -e PYTHONPATH='/opt/airflow/plugins' \
  airflow-webserver python - <<'PY'
import json
from eod_inference.pipeline import run_ml_inference

with open('/opt/airflow/data/eod_batch/staging/20260529/feature_manifest.json') as f:
    manifest = json.load(f)

result = run_ml_inference(manifest)
print(json.dumps({k: result.get(k) for k in [
    'prediction_rows',
    'orca_context_rows',
    'orca_context_includes',
    'orca_context_excludes',
    'orca_upstream_context',
]}, indent=2))
PY
```

## Validation commands

### `stock_bigdata`

```bash
python -m compileall airflow/plugins/eod_inference spark_jobs
```

### `orca-agent-advisory`

```bash
uv run pytest
```

Last result:

```text
78 passed, 1 warning
```

## Known caveats

### FinBERT

Current FinBERT endpoint is dev/staging only:

```text
https://parrot-sublease-preamble.ngrok-free.dev
```

No GPU is available locally. Do not treat Kaggle/ngrok as production serving.

Production path later:

- containerized FastAPI service
- stable internal URL
- auth/private network
- `/health`, `/sentiment`, `/sentiment/batch`
- metrics and alerts
- article-score cache by URL/title hash

### yfinance

yfinance may return transient HTTP 400 or missing fields. Latest rerun produced:

```text
sentiment_rows: 50
valuation_rows: 48
```

Pipeline still succeeded and AAPL context was present. For production-grade fundamentals/news, replace yfinance with a stronger provider.

### Risk model

Risk remains high because Model C outputs:

```text
risk_prob: 0.8894067407
```

ORCA maps this to:

```text
risk_label: CRITICAL
```

This is not an ORCA mapping bug. Next work should inspect Model C calibration and feature importance.

### Dirty working trees

Both repos still contain unrelated dirty/runtime files. Commits in this branch are scoped, but clean before opening PR if the PR UI shows unrelated files.

## Recommended next work

1. Clean unrelated dirty files before PR.
2. Open PRs from `agent-integration` in both repos.
3. Replace yfinance provider for production fundamentals/news.
4. Calibrate/backtest Model C risk output.
5. Later, deploy FinBERT as managed service when GPU or stable inference infra exists.
