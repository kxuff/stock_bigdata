from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SYMBOLS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AGN", "AIG", "ALL", "AMD", "AMGN",
    "AMZN", "AXP", "BA", "BAC", "BIIB", "BK", "BLK", "BMY", "BRK.B", "C",
    "CAT", "CELG", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DD", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FB",
    "FDX", "FOX", "FOXA", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS",
    "HAL", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI", "KO",
    "LLY", "LMT", "LOW", "LYFT", "MA", "MCD", "MDLZ", "MDT", "MET", "META",
    "MMM", "MO", "MON", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NUE",
    "NVDA", "ORCL", "OXY", "PCLN", "PEP", "PFE", "PG", "PLTR", "PM", "PYPL",
    "QCOM", "RTN", "SBUX", "SHOP", "SLB", "SNAP", "SO", "SPG", "SPOT", "T",
    "TGT", "TSLA", "TWX", "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "UTX",
    "V", "VZ", "WBA", "WFC", "WMT", "XOM", "ZM"
]

MARKET_CONTEXT_SYMBOL = "SPY"
FEATURE_VERSION = os.getenv("ML_FEATURE_VERSION", "price_v1_notebook_ac")
DEFAULT_DATA_DIR = Path(os.getenv("US_STOCK_EOD_DATA_DIR", "/opt/airflow/data/eod_batch"))
MIN_LOOKBACK_TRADING_DAYS = int(os.getenv("US_STOCK_MIN_LOOKBACK_DAYS", "260"))
BACKFILL_CALENDAR_DAYS = int(os.getenv("US_STOCK_BACKFILL_CALENDAR_DAYS", "500"))
ICEBERG_CATALOG = os.getenv("ICEBERG_CATALOG", "nessie")

RAW_PRICE_TABLE_NAME = os.getenv("US_STOCK_RAW_PRICE_TABLE", "raw.us_stock_eod_prices")
CURATED_PRICE_TABLE_NAME = os.getenv("US_STOCK_CURATED_PRICE_TABLE", "curated.us_stock_eod_prices")
ML_READY_FEATURE_TABLE_NAME = os.getenv("US_STOCK_ML_READY_FEATURE_TABLE", "ml_ready.stock_price_features")
ML_READY_PREDICTION_TABLE_NAME = os.getenv("US_STOCK_ML_READY_PREDICTION_TABLE", "ml_ready.stock_predictions_v2")
ML_READY_SENTIMENT_TABLE_NAME = os.getenv("US_STOCK_ML_READY_SENTIMENT_TABLE", "ml_ready.stock_sentiment_context")
ML_READY_VALUATION_TABLE_NAME = os.getenv("US_STOCK_ML_READY_VALUATION_TABLE", "ml_ready.stock_valuation_context")

RAW_PRICE_TABLE_LOCATION = os.getenv("US_STOCK_RAW_PRICE_LOCATION", "s3a://bronze/raw/us_stock_eod_prices")
CURATED_PRICE_TABLE_LOCATION = os.getenv("US_STOCK_CURATED_PRICE_LOCATION", "s3a://silver/curated/us_stock_eod_prices")
ML_READY_FEATURE_TABLE_LOCATION = os.getenv(
    "US_STOCK_ML_READY_FEATURE_LOCATION",
    "s3a://prediction/ml_ready/stock_price_features",
)
ML_READY_PREDICTION_TABLE_LOCATION = os.getenv(
    "US_STOCK_ML_READY_PREDICTION_LOCATION",
    "s3a://prediction/ml_ready/stock_predictions_v2",
)
ML_READY_SENTIMENT_TABLE_LOCATION = os.getenv(
    "US_STOCK_ML_READY_SENTIMENT_LOCATION",
    "s3a://prediction/ml_ready/stock_sentiment_context",
)
ML_READY_VALUATION_TABLE_LOCATION = os.getenv(
    "US_STOCK_ML_READY_VALUATION_LOCATION",
    "s3a://prediction/ml_ready/stock_valuation_context",
)

LEGACY_SYMBOL_ALIASES = {
    "AGN": "ABBV",
    "BRK.B": "BRK-B",
    "CELG": "BMY",
    "FB": "META",
    "MON": None,
    "PCLN": "BKNG",
    "RTN": "RTX",
    "TWX": None,
    "UTX": "RTX",
    "WBA": None,
}


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path
    symbols: list[str]
    min_lookback_days: int
    backfill_calendar_days: int
    feature_version: str
    model_a_path: Path
    model_c_path: Path | None
    require_risk_model: bool
    initial_load: bool
    iceberg_catalog: str
    raw_price_table: str
    curated_price_table: str
    ml_ready_feature_table: str
    ml_ready_prediction_table: str
    ml_ready_sentiment_table: str
    ml_ready_valuation_table: str
    raw_price_location: str
    curated_price_location: str
    ml_ready_feature_location: str
    ml_ready_prediction_location: str
    ml_ready_sentiment_location: str
    ml_ready_valuation_location: str

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        symbols = _parse_symbols(os.getenv("US_STOCK_EOD_SYMBOLS"))
        catalog = os.getenv("ICEBERG_CATALOG", ICEBERG_CATALOG)
        return cls(
            data_dir=Path(os.getenv("US_STOCK_EOD_DATA_DIR", str(DEFAULT_DATA_DIR))),
            symbols=symbols,
            min_lookback_days=int(os.getenv("US_STOCK_MIN_LOOKBACK_DAYS", str(MIN_LOOKBACK_TRADING_DAYS))),
            backfill_calendar_days=int(os.getenv("US_STOCK_BACKFILL_CALENDAR_DAYS", str(BACKFILL_CALENDAR_DAYS))),
            feature_version=os.getenv("ML_FEATURE_VERSION", FEATURE_VERSION),
            model_a_path=Path(os.getenv("US_STOCK_MODEL_A_PATH", "/opt/airflow/data/models/model_a.joblib")),
            model_c_path=_optional_path(os.getenv("US_STOCK_MODEL_C_PATH", "/opt/airflow/data/models/model_c.joblib")),
            require_risk_model=os.getenv("US_STOCK_REQUIRE_RISK_MODEL", "false").lower() == "true",
            initial_load=os.getenv("US_STOCK_INITIAL_LOAD", "false").lower() == "true",
            iceberg_catalog=catalog,
            raw_price_table=_qualified_table(catalog, os.getenv("US_STOCK_RAW_PRICE_TABLE", RAW_PRICE_TABLE_NAME)),
            curated_price_table=_qualified_table(
                catalog,
                os.getenv("US_STOCK_CURATED_PRICE_TABLE", CURATED_PRICE_TABLE_NAME),
            ),
            ml_ready_feature_table=_qualified_table(
                catalog,
                os.getenv("US_STOCK_ML_READY_FEATURE_TABLE", ML_READY_FEATURE_TABLE_NAME),
            ),
            ml_ready_prediction_table=_qualified_table(
                catalog,
                os.getenv("US_STOCK_ML_READY_PREDICTION_TABLE", ML_READY_PREDICTION_TABLE_NAME),
            ),
            ml_ready_sentiment_table=_qualified_table(
                catalog,
                os.getenv("US_STOCK_ML_READY_SENTIMENT_TABLE", ML_READY_SENTIMENT_TABLE_NAME),
            ),
            ml_ready_valuation_table=_qualified_table(
                catalog,
                os.getenv("US_STOCK_ML_READY_VALUATION_TABLE", ML_READY_VALUATION_TABLE_NAME),
            ),
            raw_price_location=os.getenv("US_STOCK_RAW_PRICE_LOCATION", RAW_PRICE_TABLE_LOCATION),
            curated_price_location=os.getenv("US_STOCK_CURATED_PRICE_LOCATION", CURATED_PRICE_TABLE_LOCATION),
            ml_ready_feature_location=os.getenv("US_STOCK_ML_READY_FEATURE_LOCATION", ML_READY_FEATURE_TABLE_LOCATION),
            ml_ready_prediction_location=os.getenv(
                "US_STOCK_ML_READY_PREDICTION_LOCATION",
                ML_READY_PREDICTION_TABLE_LOCATION,
            ),
            ml_ready_sentiment_location=os.getenv("US_STOCK_ML_READY_SENTIMENT_LOCATION", ML_READY_SENTIMENT_TABLE_LOCATION),
            ml_ready_valuation_location=os.getenv("US_STOCK_ML_READY_VALUATION_LOCATION", ML_READY_VALUATION_TABLE_LOCATION),
        )


def _parse_symbols(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_SYMBOLS.copy()
    symbols: list[str] = []
    for item in value.split(","):
        raw = item.strip().upper()
        if not raw:
            continue
        normalized = LEGACY_SYMBOL_ALIASES.get(raw, raw)
        if normalized is None:
            continue
        symbols.append(normalized)
    return sorted(set(symbols))


def _optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if str(path) else None


def _qualified_table(catalog: str, table_name: str) -> str:
    return table_name if table_name.startswith(f"{catalog}.") else f"{catalog}.{table_name}"
