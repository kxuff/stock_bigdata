from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.models import Variable
from airflow.operators.python import PythonOperator


for plugin_path in [Path("/opt/airflow/plugins"), Path(__file__).resolve().parents[1] / "plugins"]:
    if plugin_path.exists() and str(plugin_path) not in sys.path:
        sys.path.insert(0, str(plugin_path))

def _sync_airflow_variables_to_env() -> None:
    """Keep runtime config in Airflow Variables without making DAG parsing brittle."""
    variable_names = [
        "US_STOCK_EOD_SYMBOLS",
        "US_STOCK_EOD_DATA_DIR",
        "US_STOCK_MIN_LOOKBACK_DAYS",
        "US_STOCK_BACKFILL_CALENDAR_DAYS",
        "US_STOCK_MODEL_A_PATH",
        "US_STOCK_MODEL_C_PATH",
        "US_STOCK_REQUIRE_RISK_MODEL",
        "US_STOCK_INITIAL_LOAD",
        "ICEBERG_CATALOG",
        "US_STOCK_RAW_PRICE_TABLE",
        "US_STOCK_CURATED_PRICE_TABLE",
        "US_STOCK_ML_READY_FEATURE_TABLE",
        "US_STOCK_ML_READY_PREDICTION_TABLE",
        "US_STOCK_RAW_PRICE_LOCATION",
        "US_STOCK_CURATED_PRICE_LOCATION",
        "US_STOCK_ML_READY_FEATURE_LOCATION",
        "US_STOCK_ML_READY_PREDICTION_LOCATION",
        "US_STOCK_SPARK_EXECUTOR_MEMORY",
        "US_STOCK_SPARK_EXECUTOR_CORES",
        "US_STOCK_SPARK_CORES_MAX",
        "US_STOCK_SPARK_DRIVER_HOST",
        "US_STOCK_SPARK_DRIVER_PORT",
        "US_STOCK_SPARK_BLOCKMANAGER_PORT",
        "US_STOCK_YFINANCE_MAX_WORKERS",
        "US_STOCK_PANDAS_STAGING_BASE",
        "S3_ENDPOINT_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "ML_FEATURE_VERSION",
    ]
    for name in variable_names:
        value = Variable.get(name, default_var=None)
        if value is not None:
            os.environ[name] = value


def _extract(**context):
    _sync_airflow_variables_to_env()
    from eod_inference.pipeline import extract_eod_prices
    return extract_eod_prices(context["ds"])


def _clean(**context):
    _sync_airflow_variables_to_env()
    from eod_inference.pipeline import clean_validate_prices
    manifest = context["ti"].xcom_pull(task_ids="extract_eod_prices")
    return clean_validate_prices(manifest)


def _features(**context):
    _sync_airflow_variables_to_env()
    from eod_inference.pipeline import engineer_features, NoNewEodData
    manifest = context["ti"].xcom_pull(task_ids="clean_validate_prices")
    try:
        return engineer_features(manifest)
    except NoNewEodData as exc:
        raise AirflowSkipException(str(exc)) from exc


def _inference(**context):
    _sync_airflow_variables_to_env()
    from eod_inference.pipeline import run_ml_inference
    manifest = context["ti"].xcom_pull(task_ids="engineer_features")
    return run_ml_inference(manifest)


def _save(**context):
    _sync_airflow_variables_to_env()
    from eod_inference.pipeline import save_predictions
    manifest = context["ti"].xcom_pull(task_ids="run_ml_inference")
    return save_predictions(manifest)


with DAG(
    dag_id="us_stock_eod_inference",
    description="Daily yfinance EOD batch inference with Iceberg raw, curated, ml_ready tables.",
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    schedule="30 18 * * 1-5",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "data-platform",
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["stocks", "batch", "ml-inference", "yfinance"],
) as dag:
    extract = PythonOperator(
        task_id="extract_eod_prices",
        python_callable=_extract,
    )

    clean = PythonOperator(
        task_id="clean_validate_prices",
        python_callable=_clean,
    )

    features = PythonOperator(
        task_id="engineer_features",
        python_callable=_features,
    )

    inference = PythonOperator(
        task_id="run_ml_inference",
        python_callable=_inference,
    )

    save = PythonOperator(
        task_id="save_predictions",
        python_callable=_save,
    )

    extract >> clean >> features >> inference >> save

