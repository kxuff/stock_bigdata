from __future__ import annotations

from typing import Any

from eod_inference.config import PipelineConfig
from eod_inference.exceptions import PipelineValidationError
from eod_inference.iceberg import (
    build_spark,
    clean_prices,
    ensure_iceberg_tables,
    max_date_iso,
    merge_spark_to_iceberg,
    spark_table,
    stop_spark,
    validate_price_history,
)
from eod_inference.utils import parse_date, stage_dir, write_json


def clean_validate_prices(extract_manifest: dict[str, Any]) -> dict[str, Any]:
    config = PipelineConfig.from_env()
    spark = build_spark()
    try:
        ensure_iceberg_tables(spark, config)
        target_date = parse_date(extract_manifest["as_of_date"])
        raw = spark_table(spark, config.raw_price_table)
        if raw.limit(1).count() == 0:
            raise PipelineValidationError("No EOD price rows are available after extraction.")

        clean = clean_prices(raw, target_date)
        if clean.limit(1).count() == 0:
            raise PipelineValidationError("No valid EOD price rows are available after cleaning.")

        merge_spark_to_iceberg(
            clean,
            config.curated_price_table,
            keys=["Datetime", "Symbol"],
        )
        validate_price_history(spark, config.curated_price_table, config.symbols, config.min_lookback_days)

        manifest = {
            **extract_manifest,
            "curated_table": config.curated_price_table,
            "curated_table_location": config.curated_price_location,
            "clean_rows": int(clean.count()),
            "max_clean_date": max_date_iso(spark, config.curated_price_table),
        }
        write_json(stage_dir(config, target_date) / "clean_manifest.json", manifest)
        return manifest
    finally:
        stop_spark(spark)
