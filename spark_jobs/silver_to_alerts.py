from __future__ import annotations

import json
import math
import os
import signal
from datetime import timedelta
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


CATALOG = os.getenv("ICEBERG_CATALOG", "nessie")
CHECKPOINT_BASE = os.getenv("ALERT_CHECKPOINT_BASE", "s3a://alert/checkpoints/silver_to_alerts")
SILVER_MARKET_TABLE = os.getenv("SILVER_MARKET_TABLE", f"{CATALOG}.silver.stock_market_v2")
ALERT_TABLE = os.getenv("ALERT_TABLE", f"{CATALOG}.alert.stock_market_alerts")
ALERT_LOCATION = os.getenv("ALERT_LOCATION", "s3a://alert/stock_market_alerts")

LOOKBACK_MINUTES = int(os.getenv("ALERT_LOOKBACK_MINUTES", "240"))
RSI_OVERBOUGHT = float(os.getenv("ALERT_RSI_OVERBOUGHT", "70"))
RSI_OVERSOLD = float(os.getenv("ALERT_RSI_OVERSOLD", "30"))
RVOL_SPIKE = float(os.getenv("ALERT_RVOL_SPIKE", "3"))
RVOL_EXTREME = float(os.getenv("ALERT_RVOL_EXTREME", "5"))
VOLATILITY_SPIKE_MULTIPLIER = float(os.getenv("ALERT_VOLATILITY_SPIKE_MULTIPLIER", "2"))
MOMENTUM_MIN_PCT = float(os.getenv("ALERT_MOMENTUM_MIN_PCT", "0.005"))

ALERT_SCHEMA = StructType(
    [
        StructField("symbol", StringType(), False),
        StructField("event_time", TimestampType(), False),
        StructField("alert_type", StringType(), False),
        StructField("alert_level", StringType(), False),
        StructField("feature_name", StringType(), False),
        StructField("feature_value", DoubleType(), True),
        StructField("threshold_value", DoubleType(), True),
        StructField("message", StringType(), False),
        StructField("metadata", StringType(), False),
        StructField("source_batch_id", LongType(), False),
        StructField("process_date", TimestampType(), True),
    ]
)


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("SilverToStockMarketAlerts")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.executor.memory", "2g")
        .config("spark.executor.cores", "1")
        .config("spark.cores.max", "1")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.kafka.maxRatePerPartition", "100")
        .getOrCreate()
    )


def ensure_tables(spark: SparkSession) -> None:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.alert")
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {ALERT_TABLE} (
            symbol string,
            event_time timestamp,
            alert_type string,
            alert_level string,
            feature_name string,
            feature_value double,
            threshold_value double,
            message string,
            metadata string,
            source_batch_id long,
            process_date timestamp
        )
        USING iceberg
        PARTITIONED BY (days(event_time), symbol)
        LOCATION '{ALERT_LOCATION}'
        TBLPROPERTIES ('write.format.default'='parquet')
        """
    )


def silver_market_stream(spark: SparkSession):
    return spark.readStream.format("iceberg").load(SILVER_MARKET_TABLE)


def normalize_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def compute_features(history_pdf: pd.DataFrame) -> pd.DataFrame:
    if history_pdf.empty:
        return history_pdf

    pdf = history_pdf.copy()
    pdf["Datetime"] = normalize_datetime(pdf["Datetime"])
    pdf = pdf.sort_values(["Symbol", "Datetime", "process_date"], na_position="first")
    pdf = pdf.drop_duplicates(["Symbol", "Datetime"], keep="last")

    feature_frames = []
    for _, group in pdf.groupby("Symbol", sort=False):
        group = group.sort_values("Datetime").copy()
        close = group["Close"].astype(float)
        high = group["High"].astype(float)
        low = group["Low"].astype(float)
        volume = group["Volume"].astype(float)

        prev_close = close.shift(1)
        price_change = close.diff()
        gain = price_change.clip(lower=0)
        loss = (-price_change).clip(lower=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss

        group["rsi14"] = 100 - (100 / (1 + rs))
        group.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100
        group.loc[(avg_loss == 0) & (avg_gain == 0), "rsi14"] = 50

        group["ema9"] = close.ewm(span=9, adjust=False, min_periods=9).mean()
        group["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
        group["ema50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
        group["prev_ema9"] = group["ema9"].shift(1)
        group["prev_ema20"] = group["ema20"].shift(1)
        group["prev_ema50"] = group["ema50"].shift(1)

        group["avg_volume20"] = volume.rolling(20, min_periods=20).mean()
        group["rvol20"] = volume / group["avg_volume20"]

        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        group["atr14"] = true_range.rolling(14, min_periods=14).mean()
        group["avg_atr50"] = group["atr14"].rolling(50, min_periods=20).mean()

        returns = close.pct_change()
        group["volatility20"] = returns.rolling(20, min_periods=20).std()
        group["avg_volatility50"] = group["volatility20"].rolling(50, min_periods=20).mean()

        group["sma20"] = close.rolling(20, min_periods=20).mean()
        group["std20"] = close.rolling(20, min_periods=20).std()
        group["bb_upper20"] = group["sma20"] + 2 * group["std20"]
        group["bb_lower20"] = group["sma20"] - 2 * group["std20"]

        group["momentum5"] = close - close.shift(5)
        group["prev_momentum5"] = group["momentum5"].shift(1)
        feature_frames.append(group)

    return pd.concat(feature_frames, ignore_index=True)


def row_metadata(row: pd.Series, batch_id: int) -> str:
    keys = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "rsi14",
        "ema9",
        "ema20",
        "ema50",
        "avg_volume20",
        "rvol20",
        "atr14",
        "avg_atr50",
        "volatility20",
        "avg_volatility50",
        "sma20",
        "bb_upper20",
        "bb_lower20",
        "momentum5",
        "prev_momentum5",
    ]
    payload = {key: safe_float(row.get(key)) for key in keys}
    payload["source_batch_id"] = int(batch_id)
    payload["timeframe"] = "1m"
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def build_alert(
    row: pd.Series,
    batch_id: int,
    alert_type: str,
    alert_level: str,
    feature_name: str,
    feature_value: Any,
    threshold_value: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "symbol": str(row["Symbol"]),
        "event_time": row["Datetime"].to_pydatetime(),
        "alert_type": alert_type,
        "alert_level": alert_level,
        "feature_name": feature_name,
        "feature_value": safe_float(feature_value),
        "threshold_value": safe_float(threshold_value),
        "message": message,
        "metadata": row_metadata(row, batch_id),
        "source_batch_id": int(batch_id),
        "process_date": None,
    }


def detect_alerts(features_pdf: pd.DataFrame, batch_keys_pdf: pd.DataFrame, batch_id: int) -> list[dict[str, Any]]:
    if features_pdf.empty or batch_keys_pdf.empty:
        return []

    keys_pdf = batch_keys_pdf.copy()
    keys_pdf["Datetime"] = normalize_datetime(keys_pdf["Datetime"])
    keys_pdf = keys_pdf.drop_duplicates(["Datetime", "Symbol"])
    current_pdf = features_pdf.merge(keys_pdf, on=["Datetime", "Symbol"], how="inner")
    if current_pdf.empty:
        return []

    alerts: list[dict[str, Any]] = []
    for _, row in current_pdf.iterrows():
        symbol = row["Symbol"]
        close = safe_float(row.get("Close"))
        rsi14 = safe_float(row.get("rsi14"))
        rvol20 = safe_float(row.get("rvol20"))
        atr14 = safe_float(row.get("atr14"))
        avg_atr50 = safe_float(row.get("avg_atr50"))
        volatility20 = safe_float(row.get("volatility20"))
        avg_volatility50 = safe_float(row.get("avg_volatility50"))
        bb_upper20 = safe_float(row.get("bb_upper20"))
        bb_lower20 = safe_float(row.get("bb_lower20"))
        ema9 = safe_float(row.get("ema9"))
        ema20 = safe_float(row.get("ema20"))
        ema50 = safe_float(row.get("ema50"))
        prev_ema9 = safe_float(row.get("prev_ema9"))
        prev_ema20 = safe_float(row.get("prev_ema20"))
        prev_ema50 = safe_float(row.get("prev_ema50"))
        momentum5 = safe_float(row.get("momentum5"))
        prev_momentum5 = safe_float(row.get("prev_momentum5"))

        if rsi14 is not None and rsi14 > RSI_OVERBOUGHT:
            alerts.append(
                build_alert(
                    row,
                    batch_id,
                    "RSI_OVERBOUGHT",
                    "MEDIUM",
                    "rsi14",
                    rsi14,
                    RSI_OVERBOUGHT,
                    f"{symbol} RSI14={rsi14:.2f} > {RSI_OVERBOUGHT:.2f}: overbought",
                )
            )
        if rsi14 is not None and rsi14 < RSI_OVERSOLD:
            alerts.append(
                build_alert(
                    row,
                    batch_id,
                    "RSI_OVERSOLD",
                    "MEDIUM",
                    "rsi14",
                    rsi14,
                    RSI_OVERSOLD,
                    f"{symbol} RSI14={rsi14:.2f} < {RSI_OVERSOLD:.2f}: oversold",
                )
            )

        if rvol20 is not None and rvol20 > RVOL_SPIKE:
            level = "HIGH" if rvol20 > RVOL_EXTREME else "MEDIUM"
            alerts.append(
                build_alert(
                    row,
                    batch_id,
                    "VOLUME_SPIKE",
                    level,
                    "rvol20",
                    rvol20,
                    RVOL_SPIKE,
                    f"{symbol} volume is {rvol20:.2f}x the 20-minute average",
                )
            )

        if close is not None and bb_upper20 is not None and close > bb_upper20:
            alerts.append(
                build_alert(
                    row,
                    batch_id,
                    "PRICE_ABOVE_BOLLINGER_UPPER",
                    "MEDIUM",
                    "close_vs_bb_upper20",
                    close,
                    bb_upper20,
                    f"{symbol} close={close:.4f} crossed above Bollinger upper band={bb_upper20:.4f}",
                )
            )
        if close is not None and bb_lower20 is not None and close < bb_lower20:
            alerts.append(
                build_alert(
                    row,
                    batch_id,
                    "PRICE_BELOW_BOLLINGER_LOWER",
                    "MEDIUM",
                    "close_vs_bb_lower20",
                    close,
                    bb_lower20,
                    f"{symbol} close={close:.4f} crossed below Bollinger lower band={bb_lower20:.4f}",
                )
            )

        if all(value is not None for value in [ema9, ema20, prev_ema9, prev_ema20]):
            if ema9 > ema20 and prev_ema9 <= prev_ema20:
                alerts.append(
                    build_alert(
                        row,
                        batch_id,
                        "BULLISH_EMA_CROSS",
                        "MEDIUM",
                        "ema9_vs_ema20",
                        ema9,
                        ema20,
                        f"{symbol} EMA9 crossed above EMA20",
                    )
                )
            if ema9 < ema20 and prev_ema9 >= prev_ema20:
                alerts.append(
                    build_alert(
                        row,
                        batch_id,
                        "BEARISH_EMA_CROSS",
                        "MEDIUM",
                        "ema9_vs_ema20",
                        ema9,
                        ema20,
                        f"{symbol} EMA9 crossed below EMA20",
                    )
                )

        if all(value is not None for value in [ema20, ema50, prev_ema20, prev_ema50]):
            if ema20 > ema50 and prev_ema20 <= prev_ema50:
                alerts.append(
                    build_alert(
                        row,
                        batch_id,
                        "BULLISH_TREND_EMA_CROSS",
                        "HIGH",
                        "ema20_vs_ema50",
                        ema20,
                        ema50,
                        f"{symbol} EMA20 crossed above EMA50",
                    )
                )
            if ema20 < ema50 and prev_ema20 >= prev_ema50:
                alerts.append(
                    build_alert(
                        row,
                        batch_id,
                        "BEARISH_TREND_EMA_CROSS",
                        "HIGH",
                        "ema20_vs_ema50",
                        ema20,
                        ema50,
                        f"{symbol} EMA20 crossed below EMA50",
                    )
                )

        if atr14 is not None and avg_atr50 is not None and avg_atr50 > 0:
            threshold = avg_atr50 * VOLATILITY_SPIKE_MULTIPLIER
            if atr14 > threshold:
                alerts.append(
                    build_alert(
                        row,
                        batch_id,
                        "ATR_SPIKE",
                        "HIGH",
                        "atr14",
                        atr14,
                        threshold,
                        f"{symbol} ATR14={atr14:.4f} exceeded {VOLATILITY_SPIKE_MULTIPLIER:.1f}x baseline",
                    )
                )
        if volatility20 is not None and avg_volatility50 is not None and avg_volatility50 > 0:
            threshold = avg_volatility50 * VOLATILITY_SPIKE_MULTIPLIER
            if volatility20 > threshold:
                alerts.append(
                    build_alert(
                        row,
                        batch_id,
                        "VOLATILITY_SPIKE",
                        "HIGH",
                        "volatility20",
                        volatility20,
                        threshold,
                        f"{symbol} volatility20={volatility20:.6f} exceeded {VOLATILITY_SPIKE_MULTIPLIER:.1f}x baseline",
                    )
                )

        if all(value is not None for value in [close, momentum5, prev_momentum5]) and close > 0:
            min_abs_momentum = max(close * MOMENTUM_MIN_PCT, atr14 or 0)
            if prev_momentum5 < 0 < momentum5 and abs(momentum5) >= min_abs_momentum:
                alerts.append(
                    build_alert(
                        row,
                        batch_id,
                        "BULLISH_MOMENTUM_REVERSAL",
                        "MEDIUM",
                        "momentum5",
                        momentum5,
                        prev_momentum5,
                        f"{symbol} 5-minute momentum reversed upward strongly",
                    )
                )
            if prev_momentum5 > 0 > momentum5 and abs(momentum5) >= min_abs_momentum:
                alerts.append(
                    build_alert(
                        row,
                        batch_id,
                        "BEARISH_MOMENTUM_REVERSAL",
                        "MEDIUM",
                        "momentum5",
                        momentum5,
                        prev_momentum5,
                        f"{symbol} 5-minute momentum reversed downward strongly",
                    )
                )

    return alerts


def write_alerts_for_batch(batch_df, batch_id: int) -> None:
    if batch_df.rdd.isEmpty():
        return

    spark = batch_df.sparkSession
    batch_keys = (
        batch_df.select("Datetime", "Symbol")
        .where(col("Datetime").isNotNull() & col("Symbol").isNotNull())
        .dropDuplicates()
    )
    batch_keys_pdf = batch_keys.toPandas()
    if batch_keys_pdf.empty:
        return

    batch_keys_pdf["Datetime"] = normalize_datetime(batch_keys_pdf["Datetime"])
    changed_symbols = sorted(batch_keys_pdf["Symbol"].dropna().unique().tolist())
    if not changed_symbols:
        return

    min_event_time = batch_keys_pdf["Datetime"].min()
    max_event_time = batch_keys_pdf["Datetime"].max()
    history_start = min_event_time - timedelta(minutes=LOOKBACK_MINUTES)

    history_df = (
        spark.table(SILVER_MARKET_TABLE)
        .where(col("Symbol").isin(changed_symbols))
        .where((col("Datetime") >= lit(history_start)) & (col("Datetime") <= lit(max_event_time)))
        .select("Datetime", "Symbol", "Open", "High", "Low", "Close", "Volume", "process_date")
    )
    history_pdf = history_df.toPandas()
    if history_pdf.empty:
        return

    features_pdf = compute_features(history_pdf)
    alerts = detect_alerts(features_pdf, batch_keys_pdf, batch_id)
    if not alerts:
        return

    alerts_df = spark.createDataFrame(alerts, schema=ALERT_SCHEMA).withColumn("process_date", current_timestamp())
    alerts_df.writeTo(ALERT_TABLE).append()
    
    POSTGRES_OPTIONS = {
        "url": "jdbc:postgresql://postgres:5432/stock_db", 
        "driver": "org.postgresql.Driver",
        "dbtable": "stock_alerts", 
        "user": "postgres",
        "password": "postgres"
    }
    
    alerts_df.write.format("jdbc").options(**POSTGRES_OPTIONS).mode("append").save()


def start_stream(spark: SparkSession):
    return (
        silver_market_stream(spark)
        .writeStream.foreachBatch(write_alerts_for_batch)
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/stock_market_alerts")
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
