from dataclasses import dataclass
from typing import Any

from app.infrastructure.bigdata.bigdata_ml_provider import BigdataMlTableConfig, _build_spark_session


@dataclass
class BigdataMarketScreenProvider:
    table_config: BigdataMlTableConfig | None = None

    def __post_init__(self) -> None:
        if self.table_config is None:
            self.table_config = BigdataMlTableConfig(spark_app_name="orca-market-screen-provider")

    def screen_latest(self, limit: int = 10) -> list[dict[str, Any]]:
        from pyspark.sql import Window, functions as F, SparkSession  # type: ignore[import-not-found]

        try:
            spark = _build_spark_session(SparkSession.builder, self.table_config.spark_app_name or "orca-market-screen-provider", self.table_config)
            p = spark.table(self.table_config.table_ref(self.table_config.prediction_table))
            latest_date = p.agg(F.max("Datetime").alias("max_datetime")).collect()[0]["max_datetime"]
            if latest_date is None:
                return []
            window = Window.orderBy(F.col("final_score").desc_nulls_last())
            rows = p.where(F.col("Datetime") == F.lit(latest_date)).withColumn("rank", F.row_number().over(window)).where(F.col("rank") <= int(limit)).limit(int(limit)).collect()
            return [row.asDict(recursive=True) for row in rows]
        except Exception:  # noqa: BLE001 - route must fail soft when lakehouse table is absent.
            return []

    def load_symbols(self, symbols: list[str]) -> list[dict[str, Any]]:
        from pyspark.sql import Window, functions as F, SparkSession  # type: ignore[import-not-found]

        normalized = [symbol.upper() for symbol in symbols]
        if not normalized:
            return []
        try:
            spark = _build_spark_session(SparkSession.builder, self.table_config.spark_app_name or "orca-market-screen-provider", self.table_config)
            p = spark.table(self.table_config.table_ref(self.table_config.prediction_table)).where(F.col("Symbol").isin(normalized))
            window = Window.partitionBy("Symbol").orderBy(F.col("Datetime").desc())
            rows = p.withColumn("_rn", F.row_number().over(window)).where(F.col("_rn") == 1).drop("_rn").limit(len(normalized)).collect()
            return [row.asDict(recursive=True) for row in rows]
        except Exception:  # noqa: BLE001 - route must fail soft when lakehouse table is absent.
            return []

    def diagnose(self) -> dict[str, Any]:
        from pyspark.sql import functions as F, SparkSession  # type: ignore[import-not-found]

        spark = _build_spark_session(SparkSession.builder, self.table_config.spark_app_name or "orca-market-screen-provider", self.table_config)
        table = self.table_config.table_ref(self.table_config.prediction_table)
        try:
            df = spark.table(table)
            columns = df.columns
            sample_count = df.limit(1000).count()
            diagnostics: dict[str, Any] = {"prediction_table": table, "sample_row_count": sample_count, "columns": columns, "status": "ok"}
            if "Datetime" in columns:
                latest = df.agg(F.max("Datetime").alias("max_datetime")).collect()[0]["max_datetime"]
                diagnostics["latest_datetime"] = str(latest) if latest is not None else None
                diagnostics["freshness"] = str(latest) if latest is not None else None
                if latest is None:
                    diagnostics["status"] = "warning"
                    diagnostics["warnings"] = ["prediction table has no Datetime values"]
            else:
                diagnostics["status"] = "warning"
                diagnostics["warnings"] = ["prediction table missing Datetime column"]
            return diagnostics
        except Exception as exc:  # noqa: BLE001
            return {"prediction_table": table, "status": "error", "error": str(exc)}
