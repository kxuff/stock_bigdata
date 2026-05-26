from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

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

        state = {
            "initialized": True,
            "last_successful_as_of_date": target_date.isoformat(),
            "last_successful_market_date": max_date_iso(spark, config.curated_price_table),
            "symbols": symbol_state(spark, config.curated_price_table, sorted(set(config.symbols + [MARKET_CONTEXT_SYMBOL]))),
            "feature_version": prediction_manifest["feature_version"],
            "model_version": prediction_manifest["model_version"],
            "prediction_table": config.ml_ready_prediction_table,
            "prediction_table_location": config.ml_ready_prediction_location,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        write_json(config.data_dir / "state.json", state)
        manifest = {
            **prediction_manifest,
            "prediction_table": config.ml_ready_prediction_table,
            "prediction_table_location": config.ml_ready_prediction_location,
            "saved_prediction_rows": int(len(predictions)),
            "state_path": str(config.data_dir / "state.json"),
        }
        write_json(stage_dir(config, target_date) / "save_manifest.json", manifest)
        return manifest
    finally:
        stop_spark(spark)
