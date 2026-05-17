import signal

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    concat_ws,
    current_timestamp,
    length,
    lower,
    regexp_replace,
    trim,
    when,
)


CATALOG = "nessie"
CHECKPOINT_BASE = "s3a://silver/checkpoints/bronze_to_silver"


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("BronzeToSilverIceberg")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.executor.memory", "1g")
        .config("spark.executor.cores", "1")
        .config("spark.cores.max", "1")
        .getOrCreate()
    )


def ensure_tables(spark: SparkSession) -> None:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.silver")
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.silver.stock_market (
            Datetime timestamp,
            Symbol string,
            Open double,
            High double,
            Low double,
            Close double,
            Volume long,
            Dividends double,
            Stock_Splits double,
            daily_range double,
            close_position double,
            etl_load timestamp,
            process_date timestamp
        )
        USING iceberg
        PARTITIONED BY (days(Datetime), Symbol)
        LOCATION 's3a://silver/stock_market'
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.silver.stock_news (
            Datetime timestamp,
            Symbol string,
            category string,
            headline string,
            source string,
            summary string,
            url string,
            event_timestamp long,
            news_text string,
            has_image boolean,
            etl_load timestamp,
            process_date timestamp
        )
        USING iceberg
        PARTITIONED BY (days(Datetime), Symbol)
        LOCATION 's3a://silver/stock_news'
        """
    )


def market_silver_stream(spark: SparkSession):
    bronze = spark.readStream.format("iceberg").load(f"{CATALOG}.bronze.stock_market")
    clean = (
        bronze.select(
            "Datetime",
            trim(col("Symbol")).alias("Symbol"),
            col("Open").cast("double").alias("Open"),
            col("High").cast("double").alias("High"),
            col("Low").cast("double").alias("Low"),
            col("Close").cast("double").alias("Close"),
            col("Volume").cast("long").alias("Volume"),
            col("Dividends").cast("double").alias("Dividends"),
            col("Stock_Splits").cast("double").alias("Stock_Splits"),
            "etl_load",
        )
        .where(col("Datetime").isNotNull())
        .where(col("Symbol").isNotNull() & (length(col("Symbol")) > 0))
        .where(col("Close").isNotNull() & (col("Close") > 0))
        .where(col("High").isNull() | col("Low").isNull() | (col("High") >= col("Low")))
    )
    return clean.select(
        "Datetime",
        "Symbol",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock_Splits",
        (col("High") - col("Low")).alias("daily_range"),
        when((col("High") - col("Low")) > 0, (col("Close") - col("Low")) / (col("High") - col("Low")))
        .otherwise(None)
        .alias("close_position"),
        "etl_load",
        current_timestamp().alias("process_date"),
    )


def news_silver_stream(spark: SparkSession):
    bronze = spark.readStream.format("iceberg").load(f"{CATALOG}.bronze.stock_news")
    clean = (
        bronze.select(
            "Datetime",
            trim(col("Symbol")).alias("Symbol"),
            lower(trim(col("category"))).alias("category"),
            trim(col("headline")).alias("headline"),
            trim(col("source")).alias("source"),
            regexp_replace(trim(col("summary")), r"\s+", " ").alias("summary"),
            trim(col("url")).alias("url"),
            "event_timestamp",
            "image",
            "etl_load",
        )
        .where(col("Datetime").isNotNull())
        .where(col("Symbol").isNotNull() & (length(col("Symbol")) > 0))
        .where(col("headline").isNotNull() & (length(col("headline")) > 0))
        .where(col("url").isNotNull() & (length(col("url")) > 0))
    )
    return clean.select(
        "Datetime",
        "Symbol",
        "category",
        "headline",
        "source",
        "summary",
        "url",
        "event_timestamp",
        regexp_replace(trim(concat_ws(" ", col("headline"), col("summary"))), r"\s+", " ").alias("news_text"),
        (col("image").isNotNull() & (length(col("image")) > 0)).alias("has_image"),
        "etl_load",
        current_timestamp().alias("process_date"),
    )


def write_iceberg_stream(df, table_name: str, checkpoint: str):
    def append_batch(batch_df, _batch_id):
        batch_df.writeTo(table_name).append()

    return (
        df.writeStream.foreachBatch(append_batch)
        .option("checkpointLocation", checkpoint)
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

        queries.extend([
            write_iceberg_stream(
                market_silver_stream(spark),
                f"{CATALOG}.silver.stock_market",
                f"{CHECKPOINT_BASE}/stock_market",
            ),
            write_iceberg_stream(
                news_silver_stream(spark),
                f"{CATALOG}.silver.stock_news",
                f"{CHECKPOINT_BASE}/stock_news",
            ),
        ])
        while not stop_requested:
            if spark.streams.awaitAnyTermination(5):
                break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        shutdown()
