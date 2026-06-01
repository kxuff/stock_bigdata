from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from eod_inference.config import MARKET_CONTEXT_SYMBOL, PipelineConfig
from eod_inference.exceptions import PipelineValidationError
from eod_inference.iceberg import (
    build_spark,
    ensure_iceberg_tables,
    max_date_iso,
    merge_pandas_to_iceberg,
    stop_spark,
    symbol_state,
)
from eod_inference.utils import parse_date, read_parquet, stage_dir, write_json


def save_predictions(prediction_manifest: dict[str, Any]) -> dict[str, Any]:
    config = PipelineConfig.from_env()
    spark = build_spark()
    try:
        ensure_iceberg_tables(spark, config)
        target_date = parse_date(prediction_manifest["as_of_date"])
        predictions = read_parquet(Path(prediction_manifest["prediction_batch"]))
        if predictions.empty:
            raise PipelineValidationError("No predictions to save.")

        merge_pandas_to_iceberg(
            spark,
            predictions,
            config.ml_ready_prediction_table,
            keys=["Datetime", "Symbol", "model_version", "feature_version"],
        )
        # Save features to Iceberg so ORCA market screen can read r1/RSI14/RVOL20
        saved_feature_rows = _save_features_to_iceberg(
            spark, config, prediction_manifest.get("feature_batch", "")
        )
        _save_optional_context_table(
            spark,
            prediction_manifest,
            manifest_key="sentiment_context",
            table_name=config.ml_ready_sentiment_table,
            keys=["as_of_date", "Symbol"],
        )
        _save_optional_context_table(
            spark,
            prediction_manifest,
            manifest_key="valuation_context",
            table_name=config.ml_ready_valuation_table,
            keys=["as_of_date", "Symbol"],
        )

        state = {
            "initialized": True,
            "last_successful_as_of_date": target_date.isoformat(),
            "last_successful_market_date": max_date_iso(spark, config.curated_price_table),
            "symbols": symbol_state(spark, config.curated_price_table, sorted(set(config.symbols + [MARKET_CONTEXT_SYMBOL]))),
            "feature_version": prediction_manifest["feature_version"],
            "model_version": prediction_manifest["model_version"],
            "prediction_table": config.ml_ready_prediction_table,
            "prediction_table_location": config.ml_ready_prediction_location,
            "sentiment_table": config.ml_ready_sentiment_table,
            "sentiment_table_location": config.ml_ready_sentiment_location,
            "valuation_table": config.ml_ready_valuation_table,
            "valuation_table_location": config.ml_ready_valuation_location,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        write_json(config.data_dir / "state.json", state)
        manifest = {
            **prediction_manifest,
            "prediction_table": config.ml_ready_prediction_table,
            "prediction_table_location": config.ml_ready_prediction_location,
            "sentiment_table": config.ml_ready_sentiment_table,
            "sentiment_table_location": config.ml_ready_sentiment_location,
            "valuation_table": config.ml_ready_valuation_table,
            "valuation_table_location": config.ml_ready_valuation_location,
            "saved_prediction_rows": int(len(predictions)),
            "saved_feature_rows": saved_feature_rows,
            "state_path": str(config.data_dir / "state.json"),
        }
        write_json(stage_dir(config, target_date) / "save_manifest.json", manifest)
        return manifest
    finally:
        stop_spark(spark)


def _save_features_to_iceberg(spark, config: PipelineConfig, feature_batch_path: str) -> int:
    """Write feature batch to Iceberg ml_ready_feature_table (best-effort)."""
    try:
        if not feature_batch_path:
            return 0
        features = pd.read_parquet(feature_batch_path)
        if features.empty:
            return 0
        from eod_inference.feature_contract import PRICE_FEATURE_COLUMNS
        keep_cols = (
            ["Datetime", "Symbol"]
            + [c for c in PRICE_FEATURE_COLUMNS if c in features.columns]
            + [c for c in ["feature_version", "process_date"] if c in features.columns]
        )
        features = features[[c for c in keep_cols if c in features.columns]].copy()
        features["process_date"] = pd.Timestamp.utcnow().tz_localize(None)
        merge_pandas_to_iceberg(
            spark,
            features,
            config.ml_ready_feature_table,
            keys=["Datetime", "Symbol"],
        )
        return int(len(features))
    except Exception as exc:  # noqa: BLE001 - best-effort, don't break pipeline
        import logging
        logging.getLogger(__name__).warning("save_features skipped: %s", exc)
        return 0


def _save_optional_context_table(
    spark,
    prediction_manifest: dict[str, Any],
    *,
    manifest_key: str,
    table_name: str,
    keys: list[str],
) -> int:
    path = prediction_manifest.get(manifest_key)
    if not path:
        return 0
    context = read_parquet(Path(path))
    if context.empty:
        return 0
    context = _normalize_context_for_iceberg(context)
    merge_pandas_to_iceberg(spark, context, table_name, keys=keys)
    return int(len(context))


def _normalize_context_for_iceberg(context: pd.DataFrame) -> pd.DataFrame:
    normalized = context.copy()
    for column in ["as_of_date", "fundamentals_as_of"]:
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce").dt.date
    for column in [
        "latest_article_published_at",
        "oldest_article_published_at",
        "sentiment_scored_at",
        "valuation_fetched_at",
        "process_date",
    ]:
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce").dt.tz_localize(None)
    return normalized
