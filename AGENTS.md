# AGENTS.md

## Repo shape
- Local lakehouse/advisory demo: Airflow EOD pipeline writes Iceberg/Nessie/MinIO tables, ORCA FastAPI reads them, Streamlit is mock-first UI.
- `airflow/dags/us_stock_eod_inference.py` defines DAG `us_stock_eod_inference`, scheduled `30 18 * * 1-5`.
- `airflow/plugins/eod_inference/run_eod_pipeline.py` is EOD CLI entrypoint; root pytest relies on `airflow/plugins` in `PYTHONPATH`.
- `orca-agent-advisory/app/main.py` is ORCA API; `orca-agent-advisory/app/worker.py` is RQ worker.
- `streamlit_app/app.py` is standalone placeholder UI; no backend calls by default.
- `docs/end_to_end_local_runbook.md` has stale Linux paths/table names; prefer current code, README, and `docker-compose.yml`.

## Commands
- Start stack: `docker compose up -d`; check: `docker compose ps`.
- Full destructive local bootstrap: `./scripts/reset_and_run_local.ps1 -ResetDocker -RemoveLocalData -NinerouterKey 'sk-...'`.
- ORCA focused test: `cd orca-agent-advisory && uv run --python 3.12 pytest tests/test_bigdata_ml_provider.py`.
- Root EOD/context test from PowerShell: `$env:PYTHONPATH='airflow/plugins'; python -m pytest tests/test_agent_context.py`.
- Compile EOD plugins: `python -m compileall airflow/plugins/eod_inference`.
- Streamlit UI: `pip install -r streamlit_app/requirements.txt`; then `streamlit run streamlit_app/app.py`.

## Runtime prerequisites
- Live ORCA needs `NINEROUTER_KEY` and 9router at `http://localhost:20128/v1`; compose passes `LLM_BASE_URL=http://host.docker.internal:20128/v1`.
- Sentiment has no lexical fallback: missing/unreachable `FINBERT_API_URL` makes EOD fail fast.
- `scripts/reset_and_run_local.ps1` hardcodes FinBERT URL `https://parrot-sublease-preamble.ngrok-free.dev`.
- `orca-agent-advisory` requires Python `>=3.11,<3.14`; Dockerfile uses uv, Python 3.12, Spark 3.5.0, Java 17.

## EOD/lakehouse quirks
- First EOD run needs `US_STOCK_INITIAL_LOAD=true` and `US_STOCK_BACKFILL_CALENDAR_DAYS=500`; later runs remove both flags.
- Current prediction table is `ml_ready.stock_predictions_v2`; ignore older docs/README lines saying `ml_ready.stock_predictions`.
- Current ORCA table env in compose: `ORCA_ML_PREDICTION_TABLE=ml_ready.stock_predictions_v2`, features `ml_ready.stock_price_features`, prices `curated.us_stock_eod_prices`, sentiment `ml_ready.stock_sentiment_context`, valuation `ml_ready.stock_valuation_context`.
- EOD table defaults live in `airflow/plugins/eod_inference/config.py`; it also normalizes legacy symbols like `BRK.B -> BRK-B` and drops delisted aliases set to `None`.
- MinIO init image strips CRLF from `minio-init.sh`; avoid reintroducing Windows line-ending assumptions there.

## Do not commit
- Secrets: `.env`, `orca-agent-advisory/.env`, API keys.
- Generated/local state: `data/eod_batch`, `airflow/logs`, `ivy2`, `.slim/`, generated codemaps unless explicitly requested.
- `scripts/reset_and_run_local.ps1 -RemoveLocalData` deletes `data`, `airflow/logs`, and `ivy2`; treat as destructive.
