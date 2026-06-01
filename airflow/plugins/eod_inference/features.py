from __future__ import annotations

from typing import Any

import pandas as pd

from eod_inference.config import MARKET_CONTEXT_SYMBOL, PipelineConfig
from eod_inference.exceptions import NoNewEodData, PipelineValidationError
from eod_inference.feature_contract import PRICE_FEATURE_COLUMNS, compute_price_features
from eod_inference.iceberg import build_spark, ensure_iceberg_tables, merge_pandas_to_iceberg, spark_table, stop_spark
from eod_inference.utils import parse_date, stage_dir, write_json


def engineer_features(clean_manifest: dict[str, Any]) -> dict[str, Any]:
    config = PipelineConfig.from_env()
    spark = build_spark()
    try:
        ensure_iceberg_tables(spark, config)
        target_date = parse_date(clean_manifest["as_of_date"])
        
        # 1. Tải dữ liệu từ bảng Iceberg
        clean = spark_table(spark, config.curated_price_table).toPandas()
        spy = clean[clean["Symbol"] == MARKET_CONTEXT_SYMBOL].sort_values("Datetime")
        prices = clean[clean["Symbol"].isin(config.symbols)].sort_values(["Symbol", "Datetime"])
        
        if spy.empty:
            raise PipelineValidationError(f"Missing {MARKET_CONTEXT_SYMBOL} market context history.")
        if prices.empty:
            raise PipelineValidationError("Missing target symbol price history.")

        # ---------------------------------------------------------------------
        # --- BEGIN FIX: Xử lý Data Bleeding & Đồng bộ Index Datetime ---
        # ---------------------------------------------------------------------
        spy_close = spy.set_index("Datetime")["Close"]
        all_symbol_features = []

        # Tách biệt tính toán cho từng mã cổ phiếu để không bị dính chéo dữ liệu
        for symbol, group in prices.groupby("Symbol"):
            # Ép Datetime làm Index giống hệt hàm yf.download()
            group_sorted = group.sort_values("Datetime").set_index("Datetime")
            
            # Tính features cho riêng mã này
            symbol_feats = compute_price_features(group_sorted, spy_close, drop_incomplete=True)
            
            if not symbol_feats.empty:
                # Trả Datetime từ Index về thành cột bình thường
                symbol_feats = symbol_feats.reset_index()
                # Đảm bảo cột Symbol được gắn lại chính xác
                symbol_feats["Symbol"] = symbol
                all_symbol_features.append(symbol_feats)

        if not all_symbol_features:
            raise PipelineValidationError("Feature engineering produced no complete rows for any symbol.")
            
        # Gộp tất cả kết quả lại thành DataFrame chuẩn
        features = pd.concat(all_symbol_features, ignore_index=True)
        # ---------------------------------------------------------------------
        # --- END FIX ---
        # ---------------------------------------------------------------------

        # 2. Xử lý logic theo ngày mục tiêu (Target Date)
        features["Datetime"] = pd.to_datetime(features["Datetime"]).dt.normalize()
        run_features = features[features["Datetime"].dt.date == target_date].copy()
        
        if run_features.empty:
            latest_date = pd.to_datetime(features["Datetime"]).max().date()
            if int(clean_manifest.get("new_rows", 0)) <= 0:
                raise NoNewEodData(f"No new EOD feature rows are available for {target_date.isoformat()}.")
            if latest_date <= target_date:
                run_features = features[features["Datetime"].dt.date == latest_date].copy()
            else:
                raise PipelineValidationError(f"No features found for {target_date.isoformat()}.")

        # 3. Validate cột feature
        missing_features = [name for name in PRICE_FEATURE_COLUMNS if name not in run_features.columns]
        if missing_features:
            raise PipelineValidationError(f"Missing feature columns: {missing_features}")

        # 4. Gắn metadata và lưu vào Iceberg
        run_features["feature_version"] = config.feature_version
        run_features["process_date"] = pd.Timestamp.utcnow().tz_localize(None)
        run_features["source_batch_id"] = int(pd.Timestamp.utcnow().timestamp())
        
        merge_pandas_to_iceberg(
            spark,
            run_features,
            config.ml_ready_feature_table,
            keys=["Datetime", "Symbol", "feature_version"],
        )

        # 5. Xuất file Parquet và ghi Manifest
        batch_path = stage_dir(config, target_date) / "features.parquet"
        run_features.to_parquet(batch_path, index=False)
        
        manifest = {
            **clean_manifest,
            "feature_table": config.ml_ready_feature_table,
            "feature_table_location": config.ml_ready_feature_location,
            "feature_batch": str(batch_path),
            "feature_rows": int(len(run_features)),
            "feature_version": config.feature_version,
            "feature_columns": list(PRICE_FEATURE_COLUMNS),
        }
        write_json(stage_dir(config, target_date) / "feature_manifest.json", manifest)
        
        return manifest
        
    finally:
        stop_spark(spark)