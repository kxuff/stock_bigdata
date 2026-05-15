from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, explode, from_json, from_unixtime, to_timestamp, when
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)


CATALOG = "nessie"
KAFKA_BOOTSTRAP_SERVERS = "broker:29092"
CHECKPOINT_BASE = "s3a://bronze/checkpoints/kafka_to_bronze"

MARKET_SCHEMA = StructType(
    [
        StructField("Datetime", StringType()),
        StructField("Open", DoubleType()),
        StructField("High", DoubleType()),
        StructField("Low", DoubleType()),
        StructField("Close", DoubleType()),
        StructField("Volume", LongType()),
        StructField("Dividends", DoubleType()),
        StructField("Stock Splits", DoubleType()),
        StructField("Symbol", StringType()),
    ]
)

NEWS_SCHEMA = StructType(
    [
        StructField("category", StringType()),
        StructField("datetime", LongType()),
        StructField("headline", StringType()),
        StructField("id", LongType()),
        StructField("image", StringType()),
        StructField("related", StringType()),
        StructField("source", StringType()),
        StructField("summary", StringType()),
        StructField("url", StringType()),
        StructField("Symbol", StringType()),
    ]
)


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("KafkaToBronzeIceberg")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def ensure_tables(spark: SparkSession) -> None:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.bronze")
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.bronze.stock_market (
            Datetime timestamp,
            Open double,
            High double,
            Low double,
            Close double,
            Volume long,
            Dividends double,
            Stock_Splits double,
            Symbol string,
            kafka_topic string,
            kafka_partition int,
            kafka_offset long,
            etl_load timestamp
        )
        USING iceberg
        PARTITIONED BY (days(Datetime))
        LOCATION 's3a://bronze/stock_market'
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.bronze.stock_news (
            Datetime timestamp,
            category string,
            event_timestamp long,
            headline string,
            id long,
            image string,
            related string,
            source string,
            summary string,
            url string,
            Symbol string,
            kafka_topic string,
            kafka_partition int,
            kafka_offset long,
            etl_load timestamp
        )
        USING iceberg
        PARTITIONED BY (days(Datetime))
        LOCATION 's3a://bronze/stock_news'
        """
    )


def read_kafka_topic(spark: SparkSession, topic: str):
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
        .select(
            col("value").cast("string").alias("payload"),
            col("topic").alias("kafka_topic"),
            col("partition").alias("kafka_partition"),
            col("offset").alias("kafka_offset"),
        )
    )


def market_stream(spark: SparkSession):
    return (
        read_kafka_topic(spark, "stock_market")
        .select("*", from_json(col("payload"), MARKET_SCHEMA).alias("data"))
        .select(
            to_timestamp(col("data.Datetime")).alias("Datetime"),
            col("data.Open").alias("Open"),
            col("data.High").alias("High"),
            col("data.Low").alias("Low"),
            col("data.Close").alias("Close"),
            col("data.Volume").alias("Volume"),
            col("data.Dividends").alias("Dividends"),
            col("data").getField("Stock Splits").alias("Stock_Splits"),
            col("data.Symbol").alias("Symbol"),
            "kafka_topic",
            "kafka_partition",
            "kafka_offset",
            current_timestamp().alias("etl_load"),
        )
    )


def news_stream(spark: SparkSession):
    source = read_kafka_topic(spark, "stock_news")
    array_records = (
        source.select(
            explode(from_json(col("payload"), ArrayType(NEWS_SCHEMA))).alias("data"),
            "kafka_topic",
            "kafka_partition",
            "kafka_offset",
        )
        .where(col("data").isNotNull())
    )
    object_records = (
        source.select(
            from_json(col("payload"), NEWS_SCHEMA).alias("data"),
            "kafka_topic",
            "kafka_partition",
            "kafka_offset",
        )
        .where(col("data").isNotNull())
    )
    records = array_records.unionByName(object_records)
    return records.select(
        to_timestamp(from_unixtime(col("data.datetime"))).alias("Datetime"),
        col("data.category").alias("category"),
        col("data.datetime").alias("event_timestamp"),
        col("data.headline").alias("headline"),
        col("data.id").alias("id"),
        col("data.image").alias("image"),
        col("data.related").alias("related"),
        col("data.source").alias("source"),
        col("data.summary").alias("summary"),
        col("data.url").alias("url"),
        when(col("data.Symbol").isNotNull(), col("data.Symbol"))
        .otherwise(col("data.related"))
        .alias("Symbol"),
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        current_timestamp().alias("etl_load"),
    )


def write_iceberg_stream(df, table_name: str, checkpoint: str):
    def append_batch(batch_df, _batch_id):
        batch_df.where(col("Datetime").isNotNull()).writeTo(table_name).append()

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
            market_stream(spark),
            f"{CATALOG}.bronze.stock_market",
            f"{CHECKPOINT_BASE}/stock_market",
        ),
        write_iceberg_stream(
            news_stream(spark),
            f"{CATALOG}.bronze.stock_news",
            f"{CHECKPOINT_BASE}/stock_news",
        ),
    ]
    spark.streams.awaitAnyTermination()
