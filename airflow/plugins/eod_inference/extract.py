from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from eod_inference.config import MARKET_CONTEXT_SYMBOL, PipelineConfig
from eod_inference.iceberg import (
    build_spark,
    ensure_iceberg_tables,
    max_dates_by_symbol,
    merge_pandas_to_iceberg,
    stop_spark,
)
from eod_inference.utils import ensure_dirs, parse_date, stage_dir, write_json

logger = logging.getLogger(__name__)


def extract_eod_prices(as_of_date: str) -> dict[str, Any]:
    config = PipelineConfig.from_env()
    ensure_dirs(config)
    target_date = parse_date(as_of_date)
    manifest_path = stage_dir(config, target_date) / "extract_manifest.json"
    needed_symbols = sorted(set(config.symbols + [MARKET_CONTEXT_SYMBOL]))
    mode_by_symbol: dict[str, str] = {}
    max_dates: dict[str, date] = {}

    if not config.initial_load:
        spark = build_spark()
        try:
            ensure_iceberg_tables(spark, config)
            max_dates = max_dates_by_symbol(spark, config.raw_price_table)
        finally:
            stop_spark(spark)

    download_requests: list[tuple[str, date, date]] = []
    for symbol in needed_symbols:
        if config.initial_load:
            start = target_date - timedelta(days=config.backfill_calendar_days)
            mode_by_symbol[symbol] = "backfill"
        else:
            last_date = max_dates.get(symbol)
            start = last_date + timedelta(days=1) if last_date is not None else target_date
            mode_by_symbol[symbol] = "incremental"

        if start > target_date:
            continue

        download_requests.append((symbol, start, target_date + timedelta(days=1)))

    downloads = _download_symbols(download_requests)
    new_rows = pd.concat(downloads, ignore_index=True) if downloads else pd.DataFrame(columns=_price_columns())
    logger.info("Downloaded successfully. Starting Spark ...")
    spark = build_spark()
    try:
        logger.info("Spark Job running ...")
        ensure_iceberg_tables(spark, config)
        if not new_rows.empty:
            new_rows = _normalize_raw_price_columns(new_rows)
            merge_pandas_to_iceberg(
                spark,
                new_rows,
                config.raw_price_table,
                keys=["Datetime", "Symbol"],
            )

        manifest = {
            "as_of_date": target_date.isoformat(),
            "raw_table": config.raw_price_table,
            "raw_table_location": config.raw_price_location,
            "new_rows": int(len(new_rows)),
            "symbols": config.symbols,
            "context_symbol": MARKET_CONTEXT_SYMBOL,
            "mode_by_symbol": mode_by_symbol,
            "initial_load": config.initial_load,
            "stage_dir": str(manifest_path.parent),
        }
        write_json(manifest_path, manifest)
        return manifest
    finally:
        stop_spark(spark)


def _price_columns() -> list[str]:
    return ["Datetime", "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]


def _download_symbols(requests: list[tuple[str, date, date]]) -> list[pd.DataFrame]:
    if not requests:
        return []

    downloads: list[pd.DataFrame] = []
    for symbol, start, end in requests:
        try:
            frame = _download_symbol(symbol, start, end)
            if not frame.empty:
                downloads.append(frame)
        except Exception as exc:
            logger.warning("Unable to download %s: %s", symbol, exc)
            continue
            
    return downloads

def _download_symbol(symbol: str, start: date, end: date) -> pd.DataFrame:
    logger.info(f"Downloading {symbol} from {start} to {end}")

    # 2. Download dữ liệu (Để yfinance tự lo khoản chống block, KHÔNG truyền session)
    frame = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    
    if frame.empty:
        return pd.DataFrame(columns=_price_columns())

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")].copy()

    frame = frame.reset_index()
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")].copy()
    if "Date" in frame.columns:
        frame = frame.rename(columns={"Date": "Datetime"})
    frame["Datetime"] = pd.to_datetime(frame["Datetime"]).dt.tz_localize(None).dt.normalize()
    frame["Symbol"] = symbol
    
    for column in _price_columns():
        if column not in frame.columns:
            frame[column] = 0 if column in ["Dividends", "Stock Splits"] else np.nan

    return frame.reindex(columns=_price_columns())


def _normalize_raw_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.rename(columns={"Adj Close": "Adj_Close", "Stock Splits": "Stock_Splits"}).copy()
    output["Datetime"] = pd.to_datetime(output["Datetime"]).dt.tz_localize(None).dt.normalize()
    output["Symbol"] = output["Symbol"].astype(str).str.upper().str.strip()
    output["source"] = "yfinance"
    output["etl_load"] = pd.Timestamp.utcnow().tz_localize(None)
    numeric_columns = ["Open", "High", "Low", "Close", "Adj_Close", "Volume", "Dividends", "Stock_Splits"]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["Volume"] = output["Volume"].fillna(0).astype("int64")
    return output[
        [
            "Datetime",
            "Symbol",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj_Close",
            "Volume",
            "Dividends",
            "Stock_Splits",
            "source",
            "etl_load",
        ]
    ]
