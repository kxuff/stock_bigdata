from __future__ import annotations

import logging
import os
import socket
import uuid
from datetime import date, datetime
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    array,
    col,
    countDistinct,
    current_timestamp,
    length,
    lit,
    max as spark_max,
    to_date,
    to_timestamp,
    trim,
    upper,
    when,
)

from eod_inference.config import MARKET_CONTEXT_SYMBOL, PipelineConfig
from eod_inference.exceptions import PipelineValidationError
from eod_inference.feature_contract import PRICE_FEATURE_COLUMNS


logger = logging.getLogger(__name__)

PANDAS_STAGING_BASE = os.getenv("US_STOCK_PANDAS_STAGING_BASE", "s3a://bronze/tmp/eod_inference/pandas")


def build_spark() -> SparkSession:
    driver_host = os.getenv("US_STOCK_SPARK_DRIVER_HOST", socket.gethostname())
    return (
        SparkSession.builder
        .appName("Airflow_Nessie_Iceberg_Pipeline")
        
        # 1. Cluster & Deploy Mode Config
        .config("spark.master", "spark://spark-master:7077")
        .config("spark.submit.deployMode", "client")
        .config("spark.driver.host", driver_host)
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.driver.port", os.getenv("US_STOCK_SPARK_DRIVER_PORT", "7079"))
        .config("spark.blockManager.port", os.getenv("US_STOCK_SPARK_BLOCKMANAGER_PORT", "7080"))
        .config("spark.executor.memory", os.getenv("US_STOCK_SPARK_EXECUTOR_MEMORY", "3584m"))
        .config("spark.executor.cores", os.getenv("US_STOCK_SPARK_EXECUTOR_CORES", "2"))
        .config("spark.cores.max", os.getenv("US_STOCK_SPARK_CORES_MAX", "2"))
        
        # 3. Download & Load Jars (Kafka, Iceberg, Nessie, AWS S3)
        .config(
            "spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2,"
            "org.projectnessie.nessie-integrations:nessie-spark-extensions-3.5_2.12:0.77.1,"
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
            "org.postgresql:postgresql:42.6.0"
        )
        
        # 4. Extensions & Catalog Default Settings
        .config(
            "spark.sql.extensions", 
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,"
            "org.projectnessie.spark.extensions.NessieSparkSessionExtensions"
        )
    
        # 5. Nessie Catalog Configuration
        .config("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.nessie.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog")
        .config("spark.sql.catalog.nessie.uri", "http://nessie:19120/api/v2")
        .config("spark.sql.catalog.nessie.ref", "main")
        .config("spark.sql.catalog.nessie.authentication.type", "NONE")
        .config("spark.sql.catalog.nessie.warehouse", "s3a://bronze/warehouse")
        .config("spark.sql.defaultCatalog", os.getenv("ICEBERG_CATALOG", "nessie"))
        
        # 6. S3A / MinIO Storage Configuration
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "admin")
        .config("spark.hadoop.fs.s3a.secret.key", "password")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        
        .config("spark.sql.shuffle.partitions", "10") # Giảm số phân vùng do dữ liệu mỗi ngày nhỏ
        .config("spark.network.timeout", "800s")      # Tăng thời gian chờ mạng trong Docker
        .config("spark.executor.heartbeatInterval", "60s")
        .getOrCreate()
    )


def stop_spark(spark: SparkSession | None) -> None:
    if spark is None:
        return
    try:
        spark.stop()
    except Exception as exc:
        print(f"Unable to stop Spark cleanly: {exc}")


def namespace(table_name: str) -> str:
    return ".".join(table_name.split(".")[:-1])


def table_ref(table_name: str) -> str:
    return ".".join(f"`{part}`" for part in table_name.split("."))


def spark_table(spark: SparkSession, table_name: str):
    return spark.table(table_ref(table_name))


def ensure_iceberg_tables(spark: SparkSession, config: PipelineConfig) -> None:
    for table_name in [
        config.raw_price_table,
        config.curated_price_table,
        config.ml_ready_feature_table,
        config.ml_ready_prediction_table,
        config.ml_ready_sentiment_table,
        config.ml_ready_valuation_table,
    ]:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {table_ref(namespace(table_name))}")

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_ref(config.raw_price_table)} (
            Datetime timestamp,
            Symbol string,
            Open double,
            High double,
            Low double,
            Close double,
            Adj_Close double,
            Volume long,
            Dividends double,
            Stock_Splits double,
            source string,
            etl_load timestamp
        )
        USING iceberg
        PARTITIONED BY (days(Datetime))
        LOCATION '{config.raw_price_location}'
        TBLPROPERTIES ('write.format.default'='parquet')
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_ref(config.curated_price_table)} (
            Datetime timestamp,
            Symbol string,
            Open double,
            High double,
            Low double,
            Close double,
            Adj_Close double,
            Volume long,
            Dividends double,
            Stock_Splits double,
            daily_range double,
            close_position double,
            source string,
            etl_load timestamp,
            process_date timestamp
        )
        USING iceberg
        PARTITIONED BY (days(Datetime))
        LOCATION '{config.curated_price_location}'
        TBLPROPERTIES ('write.format.default'='parquet')
        """
    )
    feature_columns_sql = ",\n            ".join(f"{name} double" for name in PRICE_FEATURE_COLUMNS)
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_ref(config.ml_ready_feature_table)} (
            Datetime timestamp,
            Symbol string,
            {feature_columns_sql},
            feature_vector array<double>,
            feature_version string,
            source_batch_id long,
            process_date timestamp
        )
        USING iceberg
        PARTITIONED BY (days(Datetime))
        LOCATION '{config.ml_ready_feature_location}'
        TBLPROPERTIES ('write.format.default'='parquet')
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_ref(config.ml_ready_prediction_table)} (
            Datetime timestamp,
            Symbol string,
            model_version string,
            entry_price double,
            pred_a double,
            risk_prob double,
            final_score double,
            feature_version string,
            source_feature_process_date timestamp,
            process_date timestamp
        )
        USING iceberg
        PARTITIONED BY (days(Datetime))
        LOCATION '{config.ml_ready_prediction_location}'
        TBLPROPERTIES ('write.format.default'='parquet')
        """
    )
    _ensure_table_columns(
        spark,
        config.ml_ready_prediction_table,
        {"entry_price": "double"},
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_ref(config.ml_ready_sentiment_table)} (
            as_of_date date,
            Symbol string,
            sentiment_score double,
            sentiment_label string,
            article_count int,
            latest_article_published_at timestamp,
            oldest_article_published_at timestamp,
            sentiment_scored_at timestamp,
            stale_article_count int,
            top_drivers array<string>,
            source_refs array<string>,
            process_date timestamp
        )
        USING iceberg
        PARTITIONED BY (as_of_date)
        LOCATION '{config.ml_ready_sentiment_location}'
        TBLPROPERTIES ('write.format.default'='parquet')
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_ref(config.ml_ready_valuation_table)} (
            as_of_date date,
            Symbol string,
            valuation_label string,
            pe_ratio double,
            sector_pe_ratio double,
            fair_value_estimate double,
            upside_downside_pct double,
            valuation_method string,
            valuation_quality string,
            valuation_fetched_at timestamp,
            fundamentals_as_of date,
            sector_sample_count int,
            analyst_count double,
            source_refs array<string>,
            process_date timestamp
        )
        USING iceberg
        PARTITIONED BY (as_of_date)
        LOCATION '{config.ml_ready_valuation_location}'
        TBLPROPERTIES ('write.format.default'='parquet')
        """
    )


def _ensure_table_columns(spark: SparkSession, table_name: str, columns: dict[str, str]) -> None:
    existing_columns = set(spark_table(spark, table_name).columns)
    for column_name, column_type in columns.items():
        if column_name not in existing_columns:
            spark.sql(f"ALTER TABLE {table_ref(table_name)} ADD COLUMN {column_name} {column_type}")


def max_dates_by_symbol(spark: SparkSession, table_name: str) -> dict[str, date]:
    rows = spark_table(spark, table_name).groupBy("Symbol").agg(spark_max("Datetime").alias("max_datetime")).collect()
    return {
        row.Symbol: pd.Timestamp(row.max_datetime).date()
        for row in rows
        if row.Symbol and row.max_datetime is not None
    }

POSTGRES_OPTIONS = {
        "url": "jdbc:postgresql://postgres:5432/stock_db", 
        "driver": "org.postgresql.Driver",
        "user": "postgres",
        "password": "postgres"
    }

def merge_pandas_to_iceberg(
    spark: SparkSession,
    updates: pd.DataFrame,
    table_name: str,
    *,
    keys: list[str],
) -> None:
    if updates.empty:
        return
    staging_path = write_pandas_parquet(updates, table_name)
    config = PipelineConfig.from_env()
    try:
        df = spark.read.parquet(staging_path)

        _write_postgres_side_sink(df, table_name, config)
                
        if "feature_vector" in spark_table(spark, table_name).columns and "feature_vector" not in df.columns:
            df = df.withColumn("feature_vector", array(*[col(name).cast("double") for name in PRICE_FEATURE_COLUMNS]))
        merge_spark_to_iceberg(df, table_name, keys=keys)
    finally:
        delete_spark_path(spark, staging_path)


def _write_postgres_side_sink(df, table_name: str, config: PipelineConfig) -> None:
    dbtable = None
    if table_name == config.ml_ready_prediction_table:
        dbtable = "stock_inference"
    elif table_name == config.curated_price_table:
        dbtable = "stock_eod"

    if dbtable is None:
        return

    try:
        df.write.format("jdbc") \
            .options(**POSTGRES_OPTIONS) \
            .option("dbtable", dbtable) \
            .mode("append") \
            .save()
    except Exception as exc:  # noqa: BLE001 - Postgres side sink must not block Iceberg.
        logger.warning("postgres side-write skipped for %s -> %s: %s", table_name, dbtable, exc)


def write_pandas_parquet(updates: pd.DataFrame, table_name: str) -> str:
    staging_path = _staging_path(table_name)
    updates.to_parquet(
        _s3a_to_s3_uri(staging_path),
        index=False,
        engine="pyarrow",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
        storage_options=_pandas_s3_storage_options(),
    )
    return staging_path


def delete_spark_path(spark: SparkSession, path: str | None) -> None:
    if not path:
        return
    try:
        hadoop_conf = spark._jsc.hadoopConfiguration()
        jvm_path = spark._jvm.org.apache.hadoop.fs.Path(path)
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(jvm_path.toUri(), hadoop_conf)
        fs.delete(jvm_path, True)
    except Exception as exc:
        print(f"Unable to delete Spark staging path {path}: {exc}")


def _staging_path(table_name: str) -> str:
    safe_table_name = table_name.replace(".", "_")
    return f"{PANDAS_STAGING_BASE.rstrip('/')}/{safe_table_name}/{datetime.utcnow():%Y%m%d%H%M%S}_{uuid.uuid4().hex}.parquet"


def _s3a_to_s3_uri(uri: str) -> str:
    return uri.replace("s3a://", "s3://", 1)


def _pandas_s3_storage_options() -> dict[str, Any]:
    return {
        "key": os.getenv("AWS_ACCESS_KEY_ID", os.getenv("MINIO_ACCESS_KEY", "admin")),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY", os.getenv("MINIO_SECRET_KEY", "password")),
        "client_kwargs": {
            "endpoint_url": os.getenv("S3_ENDPOINT_URL", os.getenv("MINIO_ENDPOINT", "http://minio:9000")),
        },
    }


def merge_spark_to_iceberg(df, table_name: str, *, keys: list[str]) -> None:
    spark = df.sparkSession
    target_columns = spark_table(spark, table_name).columns
    for column in target_columns:
        if column not in df.columns:
            df = df.withColumn(column, lit(None))
            
    updates = df.select(*target_columns).dropDuplicates(keys)
    sort_columns = [column for column in ["Datetime", "as_of_date", "Symbol"] if column in updates.columns]
    if sort_columns:
        updates = updates.sortWithinPartitions(*sort_columns)
    view_name = f"updates_{table_name.replace('.', '_')}_{int(datetime.utcnow().timestamp())}"
    updates.createOrReplaceTempView(view_name)
    join_condition = " AND ".join([f"target.{key} <=> updates.{key}" for key in keys])
    spark.sql(
        f"""
        MERGE INTO {table_ref(table_name)} AS target
        USING {view_name} AS updates
        ON {join_condition}
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )


def clean_prices(raw, target_date: date):
    clean = (
        raw.select(
            to_timestamp(col("Datetime")).alias("Datetime"),
            upper(trim(col("Symbol"))).alias("Symbol"),
            col("Open").cast("double").alias("Open"),
            col("High").cast("double").alias("High"),
            col("Low").cast("double").alias("Low"),
            col("Close").cast("double").alias("Close"),
            col("Adj_Close").cast("double").alias("Adj_Close"),
            col("Volume").cast("long").alias("Volume"),
            col("Dividends").cast("double").alias("Dividends"),
            col("Stock_Splits").cast("double").alias("Stock_Splits"),
            col("source"),
            col("etl_load"),
        )
        .where(col("Datetime").isNotNull())
        .where(to_date(col("Datetime")) <= lit(target_date.isoformat()))
        .where(col("Symbol").isNotNull() & (length(col("Symbol")) > 0))
        .where(col("Open").isNotNull() & (col("Open") > 0))
        .where(col("High").isNotNull() & (col("High") > 0))
        .where(col("Low").isNotNull() & (col("Low") > 0))
        .where(col("Close").isNotNull() & (col("Close") > 0))
        .where(col("High") >= col("Low"))
        .where(col("Volume").isNotNull() & (col("Volume") >= 0))
    )
    return clean.select(
        "Datetime",
        "Symbol",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Volume",
        "Dividends",
        "Stock_Splits",
        (col("High") - col("Low")).alias("daily_range"),
        when((col("High") - col("Low")) > 0, (col("Close") - col("Low")) / (col("High") - col("Low")))
        .otherwise(None)
        .alias("close_position"),
        "source",
        "etl_load",
        current_timestamp().alias("process_date"),
    )


def validate_price_history(
    spark: SparkSession,
    table_name: str,
    symbols: list[str],
    min_lookback_days: int,
) -> None:
    needed_symbols = sorted(set(symbols + [MARKET_CONTEXT_SYMBOL]))
    history = (
        spark_table(spark, table_name)
        .where(col("Symbol").isin(needed_symbols))
        .groupBy("Symbol")
        .agg(countDistinct(to_date(col("Datetime"))).alias("history_days"))
    )
    counts = {row.Symbol: int(row.history_days) for row in history.collect()}
    missing_symbols = sorted(set(needed_symbols) - set(counts))
    if missing_symbols:
        logger.warning("Skipping symbols with no cleaned history: %s", missing_symbols)

    present_symbols = [symbol for symbol in needed_symbols if symbol in counts]
    if not present_symbols:
        raise PipelineValidationError("No symbol history is available after cleaning.")

    short_history = {
        symbol: counts[symbol]
        for symbol in present_symbols
        if counts.get(symbol, 0) < min_lookback_days
    }
    if short_history:
        raise PipelineValidationError(
            f"Not enough lookback history for feature inference. Required {min_lookback_days}; got {short_history}"
        )


def max_date_iso(spark: SparkSession, table_name: str) -> str | None:
    row = spark_table(spark, table_name).agg(spark_max("Datetime").alias("max_datetime")).first()
    if row is None or row.max_datetime is None:
        return None
    return pd.Timestamp(row.max_datetime).date().isoformat()


def symbol_state(spark: SparkSession, table_name: str, symbols: list[str]) -> dict[str, dict[str, Any]]:
    rows = (
        spark_table(spark, table_name)
        .where(col("Symbol").isin(symbols))
        .groupBy("Symbol")
        .agg(countDistinct(to_date(col("Datetime"))).alias("rows"), spark_max("Datetime").alias("last_date"))
        .collect()
    )
    state = {
        row.Symbol: {
            "rows": int(row.rows),
            "last_date": pd.Timestamp(row.last_date).date().isoformat() if row.last_date is not None else None,
        }
        for row in rows
    }
    return {symbol: state.get(symbol, {"rows": 0, "last_date": None}) for symbol in symbols}
