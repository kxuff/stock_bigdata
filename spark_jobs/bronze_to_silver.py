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
    spark = build_spark()
    ensure_tables(spark)

    queries = [
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
    ]
    spark.streams.awaitAnyTermination()
