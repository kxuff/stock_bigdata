import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping

from app.application.mappers.bigdata_ml_mapper import (
    is_valid_context_row,
    market_feature_from_row,
    prediction_from_row,
    row_sort_key,
)
from app.schemas.enums import ToolStatus
from app.schemas.enums import DecisionMode
from app.schemas.enums import RiskLabel, SentimentLabel, ValuationLabel
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import (
    Freshness,
    HoldingSnapshot,
    MarketFeature,
    MarketFeatureToolResult,
    MlPrediction,
    MlPredictionToolResult,
    PortfolioSnapshot,
    PortfolioToolResult,
    RiskSnapshot,
    RiskToolResult,
    SentimentSnapshot,
    SentimentToolResult,
    ToolResultBundle,
    ValuationSnapshot,
    ValuationToolResult,
)

logger = logging.getLogger(__name__)


RowLoader = Callable[[AdvisoryDecisionRequest], Iterable[Mapping[str, Any]]]


@dataclass(frozen=True)
class BigdataMlTableConfig:
    prediction_table: str = field(
        default_factory=lambda: os.getenv("ORCA_ML_PREDICTION_TABLE", "ml_ready.stock_predictions_v2")
    )
    feature_table: str = field(
        default_factory=lambda: os.getenv("ORCA_ML_FEATURE_TABLE", "ml_ready.stock_price_features")
    )
    curated_price_table: str = field(
        default_factory=lambda: os.getenv("ORCA_CURATED_PRICE_TABLE", "curated.us_stock_eod_prices")
    )
    sentiment_table: str = field(
        default_factory=lambda: os.getenv("ORCA_SENTIMENT_TABLE", "ml_ready.stock_sentiment_context")
    )
    valuation_table: str = field(
        default_factory=lambda: os.getenv("ORCA_VALUATION_TABLE", "ml_ready.stock_valuation_context")
    )
    catalog: str = field(default_factory=lambda: os.getenv("ORCA_ICEBERG_CATALOG", "nessie"))
    max_age_seconds: int = field(
        default_factory=lambda: int(os.getenv("ORCA_CONTEXT_MAX_AGE_SECONDS", "86400"))
    )
    spark_app_name: str | None = None
    spark_master: str = field(default_factory=lambda: os.getenv("ORCA_SPARK_MASTER", "spark://localhost:7077"))
    nessie_uri: str = field(default_factory=lambda: os.getenv("ORCA_NESSIE_URI", "http://localhost:19120/api/v2"))
    s3_endpoint_url: str = field(default_factory=lambda: os.getenv("ORCA_S3_ENDPOINT_URL", "http://localhost:9000"))
    minio_access_key: str = field(default_factory=lambda: os.getenv("ORCA_MINIO_ACCESS_KEY", "admin"))
    minio_secret_key: str = field(default_factory=lambda: os.getenv("ORCA_MINIO_SECRET_KEY", "password"))

    def table_ref(self, table_name: str) -> str:
        if not self.catalog or table_name.startswith(f"{self.catalog}."):
            return table_name
        return f"{self.catalog}.{table_name}"


class BigdataMlToolResultProvider:
    """ToolResultProvider-compatible adapter for joined bigdata/ML rows."""

    def __init__(
        self,
        row_loader: RowLoader | None = None,
        table_config: BigdataMlTableConfig | None = None,
    ) -> None:
        self.row_loader = row_loader
        self.table_config = table_config or BigdataMlTableConfig()

    def get_tool_results(self, request: AdvisoryDecisionRequest) -> ToolResultBundle:
        rows = list((self.row_loader or self._load_live_rows)(request))
        rows_by_symbol = _valid_latest_rows_by_symbol(rows, set(request.symbols))
        found_symbols = [symbol for symbol in request.symbols if symbol in rows_by_symbol]
        missing_symbols = [symbol for symbol in request.symbols if symbol not in rows_by_symbol]

        if not found_symbols:
            status = ToolStatus.UNAVAILABLE
            error_message = "no valid bigdata ML context found for requested symbols"
            as_of_timestamp = request.as_of_timestamp
        elif missing_symbols:
            status = ToolStatus.PARTIAL
            error_message = "missing valid bigdata ML context for symbols: " + ", ".join(missing_symbols)
            as_of_timestamp = _freshest_timestamp(rows_by_symbol.values(), request.as_of_timestamp)
        else:
            status = ToolStatus.SUCCESS
            error_message = None
            as_of_timestamp = _freshest_timestamp(rows_by_symbol.values(), request.as_of_timestamp)

        freshness = Freshness(
            is_stale=_is_stale(as_of_timestamp, request.as_of_timestamp, self.table_config.max_age_seconds),
            last_updated_at=as_of_timestamp,
            max_age_seconds=self.table_config.max_age_seconds,
        )

        ml_data: dict[str, MlPrediction] = {}
        market_data: dict[str, MarketFeature] = {}
        risk_data: dict[str, RiskSnapshot] = {}
        sentiment_data: dict[str, SentimentSnapshot] = {}
        valuation_data: dict[str, ValuationSnapshot] = {}
        market_refs: list[str] = []
        ml_refs: list[str] = []
        risk_refs: list[str] = []
        sentiment_refs: list[str] = []
        valuation_refs: list[str] = []

        for symbol in found_symbols:
            row = rows_by_symbol[symbol]
            prediction = prediction_from_row(row)
            ml_data[symbol] = prediction
            market_data[symbol] = market_feature_from_row(row, prediction.predicted_direction)
            risk_data[symbol] = _risk_snapshot_from_row(row)
            sentiment = _sentiment_snapshot_from_row(row)
            if sentiment is not None:
                sentiment_data[symbol] = sentiment
            valuation = _valuation_snapshot_from_row(row)
            if valuation is not None:
                valuation_data[symbol] = valuation
            date_ref = str(row.get("Datetime") or row.get("prediction_process_date") or "")
            prediction_ref = f"{self.table_config.prediction_table}:{symbol}:{date_ref}"
            feature_ref = f"{self.table_config.feature_table}:{symbol}:{date_ref}"
            curated_ref = f"{self.table_config.curated_price_table}:{symbol}:{date_ref}"
            market_refs.extend([feature_ref, curated_ref])
            ml_refs.append(prediction_ref)
            risk_refs.extend([prediction_ref, feature_ref])
            if sentiment is not None:
                sentiment_refs.append(f"{self.table_config.sentiment_table}:{symbol}:{row.get('sentiment_as_of_date') or date_ref}")
            if valuation is not None:
                valuation_refs.append(f"{self.table_config.valuation_table}:{symbol}:{row.get('valuation_as_of_date') or date_ref}")

        bundle_kwargs: dict[str, Any] = {
            "market_features": MarketFeatureToolResult(
                tool="MarketFeatureTool",
                status=status,
                request_id=request.request_id,
                as_of_timestamp=as_of_timestamp,
                freshness=freshness,
                source_refs=market_refs,
                error_message=error_message,
                data=market_data,
            ),
            "ml_predictions": MlPredictionToolResult(
                tool="MlPredictionTool",
                status=status,
                request_id=request.request_id,
                as_of_timestamp=as_of_timestamp,
                freshness=freshness,
                source_refs=ml_refs,
                error_message=error_message,
                data=ml_data,
            ),
            "risk_snapshot": RiskToolResult(
                tool="RiskFeatureTool",
                status=status,
                request_id=request.request_id,
                as_of_timestamp=as_of_timestamp,
                freshness=freshness,
                source_refs=risk_refs,
                error_message=error_message,
                data=risk_data,
            ),
        }
        if sentiment_data:
            bundle_kwargs["sentiment_snapshot"] = SentimentToolResult(
                tool="SentimentSnapshotTool",
                status=status,
                request_id=request.request_id,
                as_of_timestamp=as_of_timestamp,
                freshness=freshness,
                source_refs=sentiment_refs,
                error_message=error_message,
                data=sentiment_data,
            )
        if valuation_data:
            bundle_kwargs["valuation_snapshot"] = ValuationToolResult(
                tool="ValuationSnapshotTool",
                status=status,
                request_id=request.request_id,
                as_of_timestamp=as_of_timestamp,
                freshness=freshness,
                source_refs=valuation_refs,
                error_message=error_message,
                data=valuation_data,
            )
        if request.decision_mode == DecisionMode.PORTFOLIO_RECOMMENDATION:
            bundle_kwargs["portfolio_snapshot"] = _portfolio_tool_result_from_request(
                request,
                as_of_timestamp=as_of_timestamp,
                freshness=freshness,
            )

        return ToolResultBundle(**bundle_kwargs)

    def _load_live_rows(self, request: AdvisoryDecisionRequest) -> Iterable[Mapping[str, Any]]:
        from pyspark.sql import SparkSession, functions as F  # type: ignore[import-not-found]

        app_name = self.table_config.spark_app_name or "orca-bigdata-ml-provider"
        spark = _build_spark_session(SparkSession.builder, app_name, self.table_config)
        symbols = [symbol.upper() for symbol in request.symbols]
        as_of_date = request.metadata.get("as_of_date")

        p = spark.table(self.table_config.table_ref(self.table_config.prediction_table)).alias("p")
        f = spark.table(self.table_config.table_ref(self.table_config.feature_table)).alias("f")
        c = spark.table(self.table_config.table_ref(self.table_config.curated_price_table)).alias("c")

        p = p.where(F.col("Symbol").isin(symbols))
        if as_of_date:
            p = p.where(F.to_date(F.col("Datetime")) <= F.lit(str(as_of_date)))
        max_datetime = p.agg(F.max("Datetime").alias("max_datetime")).collect()[0]["max_datetime"]
        if max_datetime is None:
            return []

        p = p.where(F.col("Datetime") == F.lit(max_datetime))
        join_keys = ["Symbol", "Datetime", "feature_version"]
        joined = (
            p.join(f, join_keys, "left")
            .join(c, ["Symbol", "Datetime"], "left")
            .select(
                F.col("Symbol"),
                F.col("Datetime"),
                F.col("p.model_version"),
                F.col("p.pred_a"),
                F.col("p.risk_prob"),
                F.col("f.vol20"),
                F.col("f.maxdd20"),
                F.col("f.beta_60D"),
                F.col("p.final_score"),
                F.col("p.feature_version"),
                F.col("p.process_date").alias("prediction_process_date"),
                F.col("p.source_feature_process_date"),
                F.col("c.Close"),
                F.col("f.r1"),
                F.col("f.RVOL20"),
                F.col("f.RSI14"),
                F.col("f.MACD_hist"),
                F.col("f.BB_pctB"),
                F.col("f.BB_width"),
                F.col("f.EMA20_50_spread"),
                F.col("f.EMA20_slope"),
                F.col("f.ROC10"),
                F.col("f.ADX14"),
            )
        )
        rows = [row.asDict(recursive=True) for row in joined.collect()]
        rows_by_symbol = {str(row.get("Symbol") or "").upper(): row for row in rows}
        self._merge_optional_context(spark, rows_by_symbol, symbols, as_of_date)
        return rows

    def _merge_optional_context(self, spark: Any, rows_by_symbol: dict[str, dict[str, Any]], symbols: list[str], as_of_date: Any) -> None:
        from pyspark.sql import Window, functions as F  # type: ignore[import-not-found]

        for table_name, prefix in ((self.table_config.sentiment_table, "sentiment"), (self.table_config.valuation_table, "valuation")):
            try:
                table = spark.table(self.table_config.table_ref(table_name)).where(F.col("Symbol").isin(symbols))
                date_col = "as_of_date" if "as_of_date" in table.columns else "Datetime"
                if as_of_date:
                    table = table.where(F.to_date(F.col(date_col)) <= F.lit(str(as_of_date)))
                window = Window.partitionBy("Symbol").orderBy(F.col(date_col).desc())
                latest = table.withColumn("_rn", F.row_number().over(window)).where(F.col("_rn") == 1).drop("_rn")
                for row in latest.collect():
                    data = row.asDict(recursive=True)
                    symbol = str(data.get("Symbol") or "").upper()
                    if symbol not in rows_by_symbol:
                        continue
                    rows_by_symbol[symbol].update(data)
                    rows_by_symbol[symbol][f"{prefix}_as_of_date"] = data.get(date_col)
            except Exception as exc:
                logger.warning("failed to merge optional %s context table %s", prefix, table_name, exc_info=True)
                continue


def _valid_latest_rows_by_symbol(
    rows: Iterable[Mapping[str, Any]], requested_symbols: set[str]
) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        symbol = str(row.get("Symbol") or "").strip().upper()
        if symbol not in requested_symbols or not is_valid_context_row(row):
            continue
        current = latest.get(symbol)
        if current is None or row_sort_key(row) > row_sort_key(current):
            latest[symbol] = row
    return latest


def _build_spark_session(builder: Any, app_name: str, config: BigdataMlTableConfig) -> Any:
    return (
        builder.appName(app_name)
        .master(config.spark_master)
        .config(
            "spark.jars.packages",
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2,"
            "org.projectnessie.nessie-integrations:nessie-spark-extensions-3.5_2.12:0.77.1,"
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,"
            "org.projectnessie.spark.extensions.NessieSparkSessionExtensions",
        )
        .config(f"spark.sql.catalog.{config.catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{config.catalog}.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog")
        .config(f"spark.sql.catalog.{config.catalog}.uri", config.nessie_uri)
        .config(f"spark.sql.catalog.{config.catalog}.ref", "main")
        .config(f"spark.sql.catalog.{config.catalog}.authentication.type", "NONE")
        .config(f"spark.sql.catalog.{config.catalog}.warehouse", "s3a://bronze/warehouse")
        .config("spark.sql.defaultCatalog", config.catalog)
        .config("spark.hadoop.fs.s3a.endpoint", config.s3_endpoint_url)
        .config("spark.hadoop.fs.s3a.access.key", config.minio_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", config.minio_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.sql.shuffle.partitions", "10")
        .getOrCreate()
    )


def _freshest_timestamp(rows: Iterable[Mapping[str, Any]], fallback: datetime) -> datetime:
    timestamps = [max(row_sort_key(row)) for row in rows]
    timestamps = [timestamp for timestamp in timestamps if timestamp != datetime.min]
    return max(timestamps) if timestamps else fallback


def _is_stale(last_updated_at: datetime, request_as_of: datetime, max_age_seconds: int) -> bool:
    last_updated_at = _without_timezone(last_updated_at)
    request_as_of = _without_timezone(request_as_of)
    age_seconds = (request_as_of - last_updated_at).total_seconds()
    return age_seconds > max_age_seconds


def _without_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone().replace(tzinfo=None)


def _risk_snapshot_from_row(row: Mapping[str, Any]) -> RiskSnapshot:
    risk_prob = _float(row.get("risk_prob"), 0.0)
    vol20 = max(0.0, _float(row.get("vol20"), 0.0))
    maxdd20 = _float(row.get("maxdd20"), 0.0)
    beta = row.get("beta_60D")
    factors = [f"upstream risk_prob={risk_prob:.4f}"]
    if vol20 >= 0.04:
        factors.append(f"high volatility vol20={vol20:.4f}")
    if maxdd20 <= -0.10:
        factors.append(f"large drawdown maxdd20={maxdd20:.4f}")
    return RiskSnapshot(
        risk_label=_risk_label(risk_prob),
        volatility_30d=vol20,
        max_drawdown_90d=maxdd20,
        beta=_float(beta) if beta is not None else None,
        risk_factors=factors,
        confidence_cap=max(0.25, min(0.9, 1 - risk_prob / 2)),
    )


def _risk_label(risk_prob: float) -> RiskLabel:
    if risk_prob >= 0.75:
        return RiskLabel.CRITICAL
    if risk_prob >= 0.6:
        return RiskLabel.HIGH
    if risk_prob >= 0.35:
        return RiskLabel.MEDIUM
    return RiskLabel.LOW


def _sentiment_snapshot_from_row(row: Mapping[str, Any]) -> SentimentSnapshot | None:
    raw_label = row.get("sentiment_label") or row.get("label")
    if raw_label is None:
        return None
    return SentimentSnapshot(
        sentiment_label=_enum_value(SentimentLabel, raw_label, SentimentLabel.UNAVAILABLE),
        sentiment_score=_float(row.get("sentiment_score") or row.get("score"), 0.0),
        article_count=max(0, int(_float(row.get("article_count") or row.get("news_count"), 0.0))),
        top_drivers=_list_value(row.get("top_drivers") or row.get("drivers")),
        latest_article_published_at=_datetime_or_none(row.get("latest_article_published_at")),
        oldest_article_published_at=_datetime_or_none(row.get("oldest_article_published_at")),
        sentiment_scored_at=_datetime_or_none(row.get("sentiment_scored_at") or row.get("scored_at")),
        stale_article_count=_int_or_none(row.get("stale_article_count")),
    )


def _valuation_snapshot_from_row(row: Mapping[str, Any]) -> ValuationSnapshot | None:
    raw_label = row.get("valuation_label") or row.get("label")
    if raw_label is None:
        return None
    return ValuationSnapshot(
        valuation_label=_enum_value(ValuationLabel, raw_label, ValuationLabel.UNKNOWN),
        pe_ratio=_positive_float_or_none(row.get("pe_ratio")),
        sector_pe_ratio=_positive_float_or_none(row.get("sector_pe_ratio")),
        fair_value_estimate=_positive_float_or_none(row.get("fair_value_estimate")),
        upside_downside_pct=_float_or_none(row.get("upside_downside_pct")),
        valuation_method=row.get("valuation_method"),
        valuation_quality=row.get("valuation_quality"),
        valuation_fetched_at=_datetime_or_none(row.get("valuation_fetched_at") or row.get("fetched_at")),
        fundamentals_as_of=_datetime_or_none(row.get("fundamentals_as_of")),
        sector_sample_count=_int_or_none(row.get("sector_sample_count")),
    )


def _portfolio_tool_result_from_request(
    request: AdvisoryDecisionRequest,
    *,
    as_of_timestamp: datetime,
    freshness: Freshness,
) -> PortfolioToolResult:
    snapshot, error_message = _portfolio_snapshot_from_metadata(request.metadata)
    status = ToolStatus.SUCCESS if snapshot is not None else ToolStatus.UNAVAILABLE
    return PortfolioToolResult(
        tool="PortfolioTool",
        status=status,
        request_id=request.request_id,
        as_of_timestamp=as_of_timestamp,
        freshness=freshness,
        source_refs=["request.metadata.holdings"] if snapshot is not None else [],
        error_message=error_message,
        data=snapshot,
    )


def _portfolio_snapshot_from_metadata(metadata: Mapping[str, Any]) -> tuple[PortfolioSnapshot | None, str | None]:
    raw = metadata.get("holdings") or metadata.get("portfolio")
    if isinstance(raw, Mapping):
        raw = raw.get("holdings") or raw.get("positions") or raw.get("assets")
    if not isinstance(raw, list):
        return None, "portfolio_snapshot is required: holdings metadata missing"

    holdings: list[HoldingSnapshot] = []
    cash_weight = 0.0
    invalid_items: list[str] = []
    for item in raw:
        if not isinstance(item, Mapping):
            invalid_items.append(str(item))
            continue
        symbol = str(item.get("symbol") or item.get("ticker") or "").strip().upper().replace(".", "-")
        weight = _float_or_none(
            item.get("weight_pct")
            if item.get("weight_pct") is not None
            else item.get("weight")
            if item.get("weight") is not None
            else item.get("current_weight")
        )
        if not symbol or weight is None:
            invalid_items.append(str(item))
            continue
        if weight < 0 or weight > 100:
            invalid_items.append(str(item))
            continue
        if symbol == "CASH":
            cash_weight += weight
            continue
        holdings.append(
            HoldingSnapshot(
                symbol=symbol,
                weight_pct=round(weight, 4),
                market_value=_positive_float_or_none(item.get("market_value")),
            )
        )

    if not holdings:
        return None, "portfolio_snapshot is required: no valid non-cash holdings metadata"
    if cash_weight > 100:
        return None, "portfolio_snapshot is required: cash weight exceeds 100"
    constraints = {
        key: metadata[key]
        for key in ("min_cash_weight", "max_single_asset_weight", "excluded_symbols", "target_sectors")
        if key in metadata
    }
    if invalid_items:
        constraints["ignored_invalid_holdings"] = len(invalid_items)
    return (
        PortfolioSnapshot(
            holdings=holdings,
            cash_weight_pct=round(cash_weight, 4),
            constraints=constraints,
        ),
        None,
    )


def _float(value: Any, default: float | None = None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        if default is None:
            raise
        return default


def _float_or_none(value: Any) -> float | None:
    return None if value is None else _float(value)


def _positive_float_or_none(value: Any) -> float | None:
    parsed = _float_or_none(value)
    return parsed if parsed is not None and parsed > 0 else None


def _int_or_none(value: Any) -> int | None:
    return None if value is None else max(0, int(_float(value, 0.0)))


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _datetime_or_none(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _enum_value(enum_type: Any, value: Any, default: Any) -> Any:
    try:
        return enum_type(str(value).upper())
    except ValueError:
        return default
