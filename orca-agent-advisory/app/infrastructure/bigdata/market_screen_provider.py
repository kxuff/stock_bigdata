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
            if p.limit(1).count() == 0:
                return []
            latest_per_symbol = Window.partitionBy("Symbol").orderBy(F.col("Datetime").desc_nulls_last())
            rank_by_score = Window.orderBy(F.col("final_score").desc_nulls_last())
            top = (
                p.withColumn("_symbol_rn", F.row_number().over(latest_per_symbol))
                .where(F.col("_symbol_rn") == 1)
                .drop("_symbol_rn")
                .withColumn("rank", F.row_number().over(rank_by_score))
                .where(F.col("rank") <= int(limit))
                .limit(int(limit))
            )
            top = self._enrich(spark, top)
            return [row.asDict(recursive=True) for row in top.collect()]
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
            top = (
                p.withColumn("_rn", F.row_number().over(window))
                .where(F.col("_rn") == 1)
                .drop("_rn")
                .limit(len(normalized))
            )
            top = self._enrich(spark, top)
            return [row.asDict(recursive=True) for row in top.collect()]
        except Exception:  # noqa: BLE001 - route must fail soft when lakehouse table is absent.
            return []

    def _enrich(self, spark, df):
        """Enrich predictions with price, technicals, and news sentiment (best-effort)."""
        from pyspark.sql import Window, functions as F  # type: ignore[import-not-found]

        # ── 1. latest_price from curated EOD prices ──────────────────────────
        try:
            price_tbl = self.table_config.table_ref("curated.us_stock_eod_prices")
            prices = spark.table(price_tbl)
            w_p = Window.partitionBy("Symbol").orderBy(F.col("Datetime").desc())
            latest_price = (
                prices.select("Symbol", "Close", "Datetime")
                .withColumn("_rn", F.row_number().over(w_p))
                .where(F.col("_rn") == 1)
                .drop("_rn", "Datetime")
                .withColumnRenamed("Close", "latest_price")
            )
            df = df.join(latest_price, on="Symbol", how="left")
        except Exception:  # noqa: BLE001
            pass

        # ── 2. Technical indicators from ml_ready.stock_price_features (if exists) ─
        try:
            feat_tbl = self.table_config.table_ref("ml_ready.stock_price_features")
            features = spark.table(feat_tbl)
            feat_cols = [c for c in ("r1", "r3", "r5", "RSI14", "RVOL20", "ATR14") if c in features.columns]
            if feat_cols:
                w_f = Window.partitionBy("Symbol").orderBy(F.col("Datetime").desc())
                latest_feat = (
                    features.select("Symbol", "Datetime", *feat_cols)
                    .withColumn("_rn", F.row_number().over(w_f))
                    .where(F.col("_rn") == 1)
                    .drop("_rn", "Datetime")
                )
                df = df.join(latest_feat, on="Symbol", how="left")
        except Exception:  # noqa: BLE001 - market_features table may not exist yet
            pass

        # ── 3. News/sentiment context from ml_ready.stock_sentiment_context ──
        try:
            sentiment_tbl = self.table_config.table_ref(self.table_config.sentiment_table)
            sentiment = spark.table(sentiment_tbl)
            sentiment_cols = [
                c
                for c in (
                    "sentiment_label",
                    "sentiment_score",
                    "article_count",
                    "latest_article_published_at",
                    "stale_article_count",
                    "top_drivers",
                    "source_refs",
                )
                if c in sentiment.columns
            ]
            if sentiment_cols:
                order_col = "as_of_date" if "as_of_date" in sentiment.columns else "process_date"
                w_s = Window.partitionBy("Symbol").orderBy(F.col(order_col).desc_nulls_last())
                latest_sentiment = (
                    sentiment.select("Symbol", order_col, *sentiment_cols)
                    .withColumn("_rn", F.row_number().over(w_s))
                    .where(F.col("_rn") == 1)
                    .drop("_rn", order_col)
                )
                df = df.join(latest_sentiment, on="Symbol", how="left")
        except Exception:  # noqa: BLE001 - sentiment context may not exist yet
            pass

        return df

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
