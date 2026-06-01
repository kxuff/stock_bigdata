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

        spark = _build_spark_session(SparkSession.builder, self.table_config.spark_app_name or "orca-market-screen-provider", self.table_config)
        p = spark.table(self.table_config.table_ref(self.table_config.prediction_table))
        latest_date = p.agg(F.max("Datetime").alias("max_datetime")).collect()[0]["max_datetime"]
        if latest_date is None:
            return []
        window = Window.orderBy(F.col("final_score").desc_nulls_last())
        rows = p.where(F.col("Datetime") == F.lit(latest_date)).withColumn("rank", F.row_number().over(window)).where(F.col("rank") <= int(limit)).collect()
        return [row.asDict(recursive=True) for row in rows]

    def load_symbols(self, symbols: list[str]) -> list[dict[str, Any]]:
        from pyspark.sql import Window, functions as F, SparkSession  # type: ignore[import-not-found]

        normalized = [symbol.upper() for symbol in symbols]
        if not normalized:
            return []
        spark = _build_spark_session(SparkSession.builder, self.table_config.spark_app_name or "orca-market-screen-provider", self.table_config)
        p = spark.table(self.table_config.table_ref(self.table_config.prediction_table)).where(F.col("Symbol").isin(normalized))
        window = Window.partitionBy("Symbol").orderBy(F.col("Datetime").desc())
        rows = p.withColumn("_rn", F.row_number().over(window)).where(F.col("_rn") == 1).drop("_rn").collect()
        return [row.asDict(recursive=True) for row in rows]

    def diagnose(self) -> dict[str, Any]:
        from pyspark.sql import SparkSession  # type: ignore[import-not-found]

        spark = _build_spark_session(SparkSession.builder, self.table_config.spark_app_name or "orca-market-screen-provider", self.table_config)
        table = self.table_config.table_ref(self.table_config.prediction_table)
        try:
            count = spark.table(table).limit(1000).count()
            return {"prediction_table": table, "sample_row_count": count, "status": "ok"}
        except Exception as exc:  # noqa: BLE001
            return {"prediction_table": table, "status": "error", "error": str(exc)}
