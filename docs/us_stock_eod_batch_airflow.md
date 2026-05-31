# US Stock EOD Batch Inference

This DAG runs after the US market close and writes the serving pipeline to Iceberg tables through the configured Spark catalog. The table data files are written as Parquet under explicit MinIO `s3a://...` locations; Iceberg still keeps JSON metadata files under each table's `metadata/` directory.

Pipeline stages:

1. `extract_eod_prices`: downloads yfinance daily OHLCV data for target symbols plus `SPY`, then upserts it into the raw Iceberg table.
2. `clean_validate_prices`: normalizes schema with PySpark, validates prices/symbols/history, then upserts into the curated Iceberg table.
3. `engineer_features`: reads curated prices, imports `spark_jobs/ml_features.py`, computes the notebook feature contract, then upserts into the `ml_ready` Iceberg table.
4. `run_ml_inference`: loads model artifacts and verifies the training feature column order.
5. `save_predictions`: upserts prediction rows into the `ml_ready` prediction table and writes pipeline state.

Backfill is controlled manually with the Airflow Variable `US_STOCK_INITIAL_LOAD`.

- Set `US_STOCK_INITIAL_LOAD=true` to download `US_STOCK_BACKFILL_CALENDAR_DAYS` calendar days for every symbol and upsert those rows into raw Iceberg.
- Set `US_STOCK_INITIAL_LOAD=false` for the normal incremental path. The extractor reads the max `Datetime` already present in the raw table per symbol and downloads only later bars. If a symbol has no raw history and initial load is false, it only requests the DAG run date, so history validation can fail until you run a backfill.

## Streamlit AI Stock Picks contract

`run_ml_inference()` writes `predictions.parquet` under the EOD staging directory. The Streamlit AI Stock Picks page reads the latest batch locally, or from `ML_INFERENCE_PICKS_URL` when an external endpoint is configured, and displays this normalized contract:

| Column | Meaning |
| --- | --- |
| `Date` | Ngày phát sinh tín hiệu mua. |
| `Ticker` | Mã cổ phiếu. |
| `Entry_Price` | Giá mua đề xuất tại thời điểm tín hiệu, sourced from `Close` in the feature batch. |
| `Pred_A` | Mức tăng giá dự báo bởi mô hình, dạng thập phân. Ví dụ `0.0885` tương ứng dự báo tăng `8.85%`. |
| `Risk_Prob_%` | Chỉ số rủi ro dự báo của mô hình, hiển thị theo phần trăm. |
| `FinalScore` | Điểm xếp hạng cuối cùng, computed as `Pred_A * (1 - RiskProb)` when risk is available. |

Local Streamlit config options:

| Env var | Purpose |
| --- | --- |
| `ML_INFERENCE_PICKS_URL` | Optional HTTP endpoint returning a JSON list, or `{ "data": [...] }`, with the contract above. |
| `ML_INFERENCE_PREDICTIONS_PATH` | Optional direct path to `predictions.parquet`. |
| `ML_INFERENCE_MANIFEST_PATH` | Optional direct path to `inference_manifest.json`. |
| `US_STOCK_EOD_DATA_DIR` | EOD batch data directory used to discover the latest `staging/*/inference_manifest.json`. |

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
| `US_STOCK_INITIAL_LOAD` | `false` |
| `ICEBERG_CATALOG` | `nessie` |
| `US_STOCK_RAW_PRICE_TABLE` | `raw.us_stock_eod_prices` |
| `US_STOCK_CURATED_PRICE_TABLE` | `curated.us_stock_eod_prices` |
| `US_STOCK_ML_READY_FEATURE_TABLE` | `ml_ready.stock_price_features` |
| `US_STOCK_ML_READY_PREDICTION_TABLE` | `ml_ready.stock_predictions` |
| `US_STOCK_RAW_PRICE_LOCATION` | `s3a://bronze/raw/us_stock_eod_prices` |
| `US_STOCK_CURATED_PRICE_LOCATION` | `s3a://silver/curated/us_stock_eod_prices` |
| `US_STOCK_ML_READY_FEATURE_LOCATION` | `s3a://prediction/ml_ready/stock_price_features` |
| `US_STOCK_ML_READY_PREDICTION_LOCATION` | `s3a://prediction/ml_ready/stock_predictions` |
| `ML_FEATURE_VERSION` | `price_v1_notebook_ac` |

Implementation files:

- `airflow/plugins/eod_inference/extract.py`: yfinance extraction and raw Iceberg upsert.
- `airflow/plugins/eod_inference/clean.py`: Spark cleaning, validation, curated Iceberg upsert.
- `airflow/plugins/eod_inference/features.py`: curated read, feature computation, ml_ready feature upsert.
- `airflow/plugins/eod_inference/inference.py`: model artifact loading and pandas inference batch.
- `airflow/plugins/eod_inference/save.py`: prediction Iceberg upsert and state update.
- `airflow/plugins/eod_inference/iceberg.py`: Spark session, Iceberg DDL, merge helpers.
- `airflow/plugins/eod_inference/pipeline.py`: compatibility re-export for the DAG imports.

Model artifacts can be a fitted estimator or a dictionary:

```python
{
    "model": fitted_model,
    "feature_columns": PRICE_FEATURE_COLUMNS,
    "model_version": "model_a_2026_05_25"
}
```

Recommended export cell for `bd-ml-filtering.ipynb`:

```python
from pathlib import Path
import joblib

from spark_jobs.ml_features import PRICE_FEATURE_COLUMNS

artifact_dir = Path("/opt/airflow/data/models")
artifact_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": fitted_model_a,
        "feature_columns": list(PRICE_FEATURE_COLUMNS),
        "model_version": "model_a_2026_05_26",
    },
    artifact_dir / "model_a.joblib",
)

# Optional risk model used for risk_prob.
joblib.dump(
    {
        "model": fitted_model_c,
        "feature_columns": list(PRICE_FEATURE_COLUMNS),
        "model_version": "model_c_2026_05_26",
    },
    artifact_dir / "model_c.joblib",
)
```

For local notebooks outside the Airflow container, export to the mounted repo path `data/models/model_a.joblib` and keep `US_STOCK_MODEL_A_PATH=/opt/airflow/data/models/model_a.joblib`.

For training-serving skew prevention, the notebook should import `PRICE_FEATURE_COLUMNS` and `compute_price_features` from `spark_jobs/ml_features.py` instead of maintaining a separate feature function in notebook cells. The DAG does not define feature formulas.
