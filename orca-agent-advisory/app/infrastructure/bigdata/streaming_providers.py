from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.infrastructure.bigdata.bigdata_ml_provider import BigdataMlTableConfig, _build_spark_session


STREAMING_TABLES = {
    "silver_prices": "silver.stock_market",
    "silver_indicators": "silver.stock_market_indicator",
    "silver_news": "silver.stock_news_v2",
    "ml_features": "ml.stock_price_features",
    "ml_predictions": "ml.stock_predictions",
    "alerts": "alert.stock_market_alerts",
}


@dataclass
class SparkStreamingProvider:
    table_config: BigdataMlTableConfig | None = None

    def __post_init__(self) -> None:
        if self.table_config is None:
            self.table_config = BigdataMlTableConfig(spark_app_name="orca-streaming-observability")

    def get_pipeline_health(self, lookback_minutes: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for stage, table in STREAMING_TABLES.items():
            rows.append(self._table_health(stage, table))
        return rows

    def get_symbol_freshness(self, symbols: list[str], lookback_minutes: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for table in [STREAMING_TABLES["silver_prices"], STREAMING_TABLES["silver_indicators"], STREAMING_TABLES["ml_features"], STREAMING_TABLES["ml_predictions"]]:
            result.extend(self._symbol_latest_rows(table, symbols))
        return result

    def get_ingestion_lag(self, lookback_minutes: int) -> list[dict[str, Any]]:
        return [self._lag_row(table) for table in STREAMING_TABLES.values()]

    def get_latest_alerts(self, symbols: list[str], severities: list[str], limit: int, lookback_minutes: int) -> list[dict[str, Any]]:
        try:
            from pyspark.sql import functions as F  # type: ignore[import-not-found]

            df = self._spark().table(self._ref(STREAMING_TABLES["alerts"]))
            symbol_col = _first_col(df.columns, ["Symbol", "symbol"])
            severity_col = _first_col(df.columns, ["severity", "Severity", "alert_severity"])
            time_col = _time_col(df.columns)
            if symbols and symbol_col:
                df = df.where(F.col(symbol_col).isin([s.upper() for s in symbols]))
            if severities and severity_col:
                df = df.where(F.col(severity_col).isin(severities))
            if time_col:
                df = df.orderBy(F.col(time_col).desc())
            return [r.asDict(recursive=True) for r in df.limit(int(limit)).collect()]
        except Exception as exc:  # noqa: BLE001
            return [{"table": STREAMING_TABLES["alerts"], "status": "error", "error": str(exc)}]

    def get_active_symbol_alerts(self, symbol: str, lookback_minutes: int) -> list[dict[str, Any]]:
        return self.get_latest_alerts([symbol], [], 20, lookback_minutes)

    def find_quality_incidents(self, symbols: list[str], lookback_minutes: int, limit: int) -> list[dict[str, Any]]:
        incidents: list[dict[str, Any]] = []
        for table in [STREAMING_TABLES["silver_prices"], STREAMING_TABLES["silver_indicators"], STREAMING_TABLES["ml_features"]]:
            incidents.extend(self._null_symbol_incidents(table, symbols, limit))
        return incidents[:limit]

    def compare_streaming_to_batch_features(self, symbols: list[str], as_of_date: str | None) -> list[dict[str, Any]]:
        return [
            {
                "symbol": symbol,
                "feature": "streaming_vs_batch",
                "streaming_value": None,
                "batch_value": None,
                "delta": None,
                "status": "diagnostic_only",
                "error": "Batch feature table mapping not configured for read-only drift comparison.",
            }
            for symbol in symbols
        ]

    def inspect_topics(self) -> list[dict[str, Any]]:
        return [
            {"topic": topic, "status": "diagnostic_only", "sample": {}, "limitation": "Kafka direct topic inspection not enabled; use Iceberg streaming tables for read-only diagnostics."}
            for topic in ["stock-market", "stock-news", "stock-market-alerts"]
        ]

    def _spark(self) -> Any:
        from pyspark.sql import SparkSession  # type: ignore[import-not-found]

        return _build_spark_session(SparkSession.builder, self.table_config.spark_app_name or "orca-streaming-observability", self.table_config)

    def _ref(self, table: str) -> str:
        return self.table_config.table_ref(table)

    def _table_health(self, stage: str, table: str) -> dict[str, Any]:
        try:
            df = self._spark().table(self._ref(table))
            time_col = _time_col(df.columns)
            latest = df.agg({time_col: "max"}).collect()[0][0] if time_col else None
            return {"stage": stage, "table": table, "status": "ok", "latest_timestamp": _string(latest), "row_count": df.limit(1000).count()}
        except Exception as exc:  # noqa: BLE001
            return {"stage": stage, "table": table, "status": "error", "error": str(exc)}

    def _lag_row(self, table: str) -> dict[str, Any]:
        row = self._table_health(table, table)
        row["lag_minutes"] = _lag_minutes(row.get("latest_timestamp"))
        return row

    def _symbol_latest_rows(self, table: str, symbols: list[str]) -> list[dict[str, Any]]:
        try:
            from pyspark.sql import Window, functions as F  # type: ignore[import-not-found]

            df = self._spark().table(self._ref(table))
            symbol_col = _first_col(df.columns, ["Symbol", "symbol"])
            time_col = _time_col(df.columns)
            if symbol_col and symbols:
                df = df.where(F.col(symbol_col).isin([s.upper() for s in symbols]))
            if symbol_col and time_col:
                window = Window.partitionBy(symbol_col).orderBy(F.col(time_col).desc())
                data = df.withColumn("_rn", F.row_number().over(window)).where(F.col("_rn") == 1).collect()
            else:
                data = df.limit(1).collect()
            return [{"symbol": r.asDict().get(symbol_col), "table": table, "latest_timestamp": _string(r.asDict().get(time_col)), "lag_minutes": _lag_minutes(_string(r.asDict().get(time_col))), "status": "ok"} for r in data]
        except Exception as exc:  # noqa: BLE001
            return [{"table": table, "status": "error", "error": str(exc)}]

    def _null_symbol_incidents(self, table: str, symbols: list[str], limit: int) -> list[dict[str, Any]]:
        try:
            from pyspark.sql import functions as F  # type: ignore[import-not-found]

            df = self._spark().table(self._ref(table))
            symbol_col = _first_col(df.columns, ["Symbol", "symbol"])
            time_col = _time_col(df.columns)
            if symbol_col and symbols:
                df = df.where(F.col(symbol_col).isin([s.upper() for s in symbols]))
            if symbol_col:
                df = df.where(F.col(symbol_col).isNull() | (F.col(symbol_col) == ""))
            rows = df.limit(int(limit)).collect()
            return [{"symbol": r.asDict().get(symbol_col), "table": table, "incident_type": "missing_symbol", "message": "Missing or blank symbol detected.", "timestamp": _string(r.asDict().get(time_col))} for r in rows]
        except Exception as exc:  # noqa: BLE001
            return [{"symbol": None, "table": table, "incident_type": "diagnostic_error", "message": str(exc), "timestamp": None}]


def _first_col(columns: list[str], candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in columns), None)


def _time_col(columns: list[str]) -> str | None:
    return _first_col(columns, ["Datetime", "timestamp", "event_time", "process_time", "process_date", "created_at"])


def _string(value: Any) -> str | None:
    return None if value is None else str(value)


def _lag_minutes(value: str | None) -> float | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - dt).total_seconds() / 60, 2)
    except ValueError:
        return None
