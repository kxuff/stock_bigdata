# US Stock EOD Batch Inference

This DAG runs after the US market close and keeps one local Parquet-backed serving store under `/opt/airflow/data/eod_batch`.

Pipeline stages:

1. `extract_eod_prices`: downloads yfinance daily OHLCV data for target symbols plus `SPY`.
2. `clean_validate_prices`: normalizes schema and validates prices, symbols, and minimum history.
3. `engineer_features`: imports `spark_jobs/ml_features.py` and calls `compute_price_features`.
4. `run_ml_inference`: loads model artifacts and verifies the training feature column order.
5. `save_predictions`: upserts prediction rows and writes pipeline state.

Initial backfill is automatic per symbol. If the local store has fewer than `US_STOCK_MIN_LOOKBACK_DAYS` rows, the extractor downloads `US_STOCK_BACKFILL_CALENDAR_DAYS` calendar days. Once enough history exists, it only requests dates after the last stored trading day.

If the DAG runs on a US market holiday and yfinance returns no new daily bar, the feature task raises an Airflow skip so stale rows are not re-scored.

Important Airflow Variables:

| Variable | Default |
| --- | --- |
| `US_STOCK_EOD_SYMBOLS` | `AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,NFLX,AMD,INTC` |
| `US_STOCK_EOD_DATA_DIR` | `/opt/airflow/data/eod_batch` |
| `US_STOCK_MIN_LOOKBACK_DAYS` | `260` |
| `US_STOCK_BACKFILL_CALENDAR_DAYS` | `430` |
| `US_STOCK_MODEL_A_PATH` | `/opt/airflow/data/models/model_a.joblib` |
| `US_STOCK_MODEL_C_PATH` | `/opt/airflow/data/models/model_c.joblib` |
| `US_STOCK_REQUIRE_RISK_MODEL` | `false` |
| `ML_FEATURE_VERSION` | `price_v1_notebook_ac` |

Model artifacts can be a fitted estimator or a dictionary:

```python
{
    "model": fitted_model,
    "feature_columns": PRICE_FEATURE_COLUMNS,
    "model_version": "model_a_2026_05_25"
}
```

For training-serving skew prevention, the notebook should import `PRICE_FEATURE_COLUMNS` and `compute_price_features` from `spark_jobs/ml_features.py` instead of maintaining a separate feature function in notebook cells. The DAG does not define feature formulas.
