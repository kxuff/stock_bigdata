from __future__ import annotations

import os
import signal

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, current_timestamp, lit

from ml_features import PRICE_FEATURE_COLUMNS, compute_price_features


CATALOG = os.getenv("ICEBERG_CATALOG", "nessie")
CHECKPOINT_BASE = os.getenv("ML_CHECKPOINT_BASE", "s3a://prediction/checkpoints/silver_to_ml_features")
FEATURE_VERSION = os.getenv("ML_FEATURE_VERSION", "price_v1_notebook_ac")
SILVER_MARKET_TABLE = f"{CATALOG}.silver.stock_market"
ML_FEATURE_TABLE = f"{CATALOG}.ml.stock_price_features"


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("SilverToMLPriceFeatures")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.executor.memory", "3500m")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "2")
        .getOrCreate()
    )


def ensure_tables(spark: SparkSession) -> None:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.ml")
    feature_columns_sql = ",\n            ".join(f"{name} double" for name in PRICE_FEATURE_COLUMNS)
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {ML_FEATURE_TABLE} (
            Datetime timestamp,
            Symbol string,
            {feature_columns_sql},
            feature_vector array<double>,
            feature_version string,
            source_batch_id long,
            process_date timestamp
        )
        USING iceberg
        PARTITIONED BY (days(Datetime), Symbol)
        LOCATION 's3a://prediction/stock_price_features'
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.ml.stock_predictions (
            Datetime timestamp,
            Symbol string,
            model_version string,
            pred_a double,
            risk_prob double,
            final_score double,
            feature_version string,
            source_feature_process_date timestamp,
            process_date timestamp
        )
        USING iceberg
        PARTITIONED BY (days(Datetime), Symbol)
        LOCATION 's3a://prediction/stock_predictions'
        """
    )


def silver_market_stream(spark: SparkSession):
    return spark.readStream.format("iceberg").load(SILVER_MARKET_TABLE)


def write_features_for_batch(batch_df, batch_id: int) -> None:
    if batch_df.rdd.isEmpty():
        return

    spark = batch_df.sparkSession
    batch_keys = batch_df.select("Datetime", "Symbol").dropna().dropDuplicates()
    changed_symbols = [row.Symbol for row in batch_keys.select("Symbol").distinct().collect()]
    candidate_symbols = sorted(symbol for symbol in changed_symbols if symbol != "SPY")
    if not candidate_symbols:
        return

    history_symbols = sorted(set(candidate_symbols + ["SPY"]))
    history_df = (
        spark.table(SILVER_MARKET_TABLE)
        .where(col("Symbol").isin(history_symbols))
        .select("Datetime", "Symbol", "Open", "High", "Low", "Close", "Volume")
    )
    history_pdf = history_df.toPandas()
    if history_pdf.empty:
        return

    spy_pdf = history_pdf[history_pdf["Symbol"] == "SPY"].sort_values("Datetime")
    price_pdf = history_pdf[history_pdf["Symbol"].isin(candidate_symbols)].sort_values(["Symbol", "Datetime"])
    if spy_pdf.empty or price_pdf.empty:
        return

    features_pdf = compute_price_features(
        price_pdf,
        spy_pdf.set_index("Datetime")["Close"],
        drop_incomplete=True,
    )
    if features_pdf.empty:
        return

    keys_pdf = batch_keys.toPandas()
    keys_pdf = keys_pdf[keys_pdf["Symbol"] != "SPY"]
    keys_pdf["Datetime"] = pd.to_datetime(keys_pdf["Datetime"], utc=True).dt.tz_localize(None).dt.floor("D")
    keys_pdf = keys_pdf.drop_duplicates()
    features_pdf["Datetime"] = pd.to_datetime(features_pdf["Datetime"])
    output_pdf = features_pdf.merge(keys_pdf, on=["Datetime", "Symbol"], how="inner")
    if output_pdf.empty:
        return

    output_df = spark.createDataFrame(output_pdf)
    output_df = (
        output_df.select(
            "Datetime",
            "Symbol",
            *[col(name).cast("double").alias(name) for name in PRICE_FEATURE_COLUMNS],
        )
        .withColumn("feature_vector", array(*[col(name).cast("double") for name in PRICE_FEATURE_COLUMNS]))
        .withColumn("feature_version", lit(FEATURE_VERSION))
        .withColumn("source_batch_id", lit(int(batch_id)))
        .withColumn("process_date", current_timestamp())
    )

    view_name = f"updates_stock_price_features_{batch_id}"
    output_df.createOrReplaceTempView(view_name)
    spark.sql(
        f"""
        MERGE INTO {ML_FEATURE_TABLE} AS target
        USING {view_name} AS updates
        ON target.Datetime = updates.Datetime
           AND target.Symbol = updates.Symbol
           AND target.feature_version = updates.feature_version
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )


def start_stream(spark: SparkSession):
    return (
        silver_market_stream(spark)
        .writeStream.foreachBatch(write_features_for_batch)
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/stock_price_features")
        .outputMode("append")
        .start()
    )


if __name__ == "__main__":
    spark = None
    queries = []
    stop_requested = False

    def request_shutdown(_signum=None, _frame=None):
        global stop_requested
        print("Shutdown requested...")
        stop_requested = True

    def shutdown():
        print("Stopping streaming queries...")
        for query in queries:
            try:
                query.stop()
            except Exception as e:
                print(f"Unable to stop query cleanly: {e}")
        if spark is not None:
            try:
                spark.stop()
            except Exception as e:
                print(f"Unable to stop Spark cleanly: {e}")

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    try:
        spark = build_spark()
        ensure_tables(spark)
        queries.append(start_stream(spark))
        while not stop_requested:
            if spark.streams.awaitAnyTermination(5):
                break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        shutdown()
