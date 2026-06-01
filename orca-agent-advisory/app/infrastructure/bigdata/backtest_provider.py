from dataclasses import dataclass
from typing import Any

from app.application.ports.backtest_provider import BacktestProviderResult, BacktestRequest
from app.infrastructure.bigdata.bigdata_ml_provider import BigdataMlTableConfig, _build_spark_session


@dataclass
class IcebergSparkBacktestProvider:
    table_config: BigdataMlTableConfig | None = None

    def __post_init__(self) -> None:
        if self.table_config is None:
            self.table_config = BigdataMlTableConfig(spark_app_name="orca-backtest-provider")

    def is_available(self) -> bool:
        try:
            from pyspark.sql import SparkSession  # type: ignore[import-not-found]

            spark = _build_spark_session(SparkSession.builder, self.table_config.spark_app_name or "orca-backtest-provider", self.table_config)
            spark.table(self.table_config.table_ref(self.table_config.curated_price_table)).limit(1).collect()
            return True
        except Exception:  # noqa: BLE001
            return False

    def run_backtest(self, request: BacktestRequest) -> BacktestProviderResult:
        from pyspark.sql import functions as F, SparkSession  # type: ignore[import-not-found]

        spark = _build_spark_session(SparkSession.builder, self.table_config.spark_app_name or "orca-backtest-provider", self.table_config)
        table = self.table_config.table_ref(self.table_config.curated_price_table)
        df = spark.table(table)
        date_col = "Date" if "Date" in df.columns else "Datetime"
        symbol_col = "Symbol" if "Symbol" in df.columns else "symbol"
        close_col = "Close" if "Close" in df.columns else "close"
        filtered = df.where(F.col(symbol_col).isin([s.upper() for s in request.symbols]))
        if request.start_date:
            filtered = filtered.where(F.to_date(F.col(date_col)) >= F.lit(request.start_date))
        if request.end_date:
            filtered = filtered.where(F.to_date(F.col(date_col)) <= F.lit(request.end_date))
        rows = filtered.select(F.col(symbol_col).alias("symbol"), F.to_date(F.col(date_col)).alias("date"), F.col(close_col).cast("double").alias("close")).dropna().orderBy("date", "symbol").collect()
        points: list[dict[str, Any]] = [{"date": str(r["date"]), "symbol": r["symbol"], "close": float(r["close"])} for r in rows]
        if not points:
            return BacktestProviderResult(metrics={"rows": 0}, trades_summary={"trades": 0}, warnings=["no Iceberg price rows found for backtest request"])
        first = points[0]["close"]
        last = points[-1]["close"]
        total_return = round((last / first - 1.0) if first else 0.0, 6)
        step = max(1, len(points) // 25)
        equity_curve = [{"date": p["date"], "equity": round(1.0 + ((p["close"] / first - 1.0) if first else 0.0), 6)} for p in points[::step]][:25]
        return BacktestProviderResult(metrics={"total_return": total_return, "rows": len(points), "symbols": sorted(set(request.symbols))}, trades_summary={"trades": 0, "strategy": request.strategy}, equity_curve_sampled=equity_curve)
