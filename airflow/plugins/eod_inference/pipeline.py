from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


def _load_feature_contract():
    """Import the notebook-derived feature module mounted with the project."""
    candidate_paths = [
        Path("/opt/airflow/spark_jobs"),
        Path.cwd() / "spark_jobs",
        Path(__file__).resolve().parents[3] / "spark_jobs",
    ]
    for path in candidate_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from ml_features import PRICE_FEATURE_COLUMNS, compute_price_features

    return PRICE_FEATURE_COLUMNS, compute_price_features


PRICE_FEATURE_COLUMNS, compute_price_features = _load_feature_contract()


DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "META",
    "NFLX",
    "AMD",
    "INTC",
]

MARKET_CONTEXT_SYMBOL = "SPY"
FEATURE_VERSION = os.getenv("ML_FEATURE_VERSION", "price_v1_notebook_ac")
DEFAULT_DATA_DIR = Path(os.getenv("US_STOCK_EOD_DATA_DIR", "/opt/airflow/data/eod_batch"))
MIN_LOOKBACK_TRADING_DAYS = int(os.getenv("US_STOCK_MIN_LOOKBACK_DAYS", "260"))
BACKFILL_CALENDAR_DAYS = int(os.getenv("US_STOCK_BACKFILL_CALENDAR_DAYS", "430"))


class PipelineValidationError(ValueError):
    pass


class NoNewEodData(RuntimeError):
    pass


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

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        symbols = _parse_symbols(os.getenv("US_STOCK_EOD_SYMBOLS"))
        return cls(
            data_dir=Path(os.getenv("US_STOCK_EOD_DATA_DIR", str(DEFAULT_DATA_DIR))),
            symbols=symbols,
            min_lookback_days=int(os.getenv("US_STOCK_MIN_LOOKBACK_DAYS", str(MIN_LOOKBACK_TRADING_DAYS))),
            backfill_calendar_days=int(os.getenv("US_STOCK_BACKFILL_CALENDAR_DAYS", str(BACKFILL_CALENDAR_DAYS))),
            feature_version=os.getenv("ML_FEATURE_VERSION", FEATURE_VERSION),
            model_a_path=Path(os.getenv("US_STOCK_MODEL_A_PATH", "/opt/airflow/data/models/model_a.joblib")),
            model_c_path=_optional_path(os.getenv("US_STOCK_MODEL_C_PATH", "/opt/airflow/data/models/model_c.joblib")),
            require_risk_model=os.getenv("US_STOCK_REQUIRE_RISK_MODEL", "false").lower() == "true",
        )


def extract_eod_prices(as_of_date: str) -> dict[str, Any]:
    """Fetch historical backfill or incremental EOD prices from yfinance."""
    config = PipelineConfig.from_env()
    _ensure_dirs(config)
    target_date = _parse_date(as_of_date)
    raw_store = config.data_dir / "raw" / "prices.parquet"
    manifest_path = _stage_dir(config, target_date) / "extract_manifest.json"

    existing = _read_parquet(raw_store)
    needed_symbols = sorted(set(config.symbols + [MARKET_CONTEXT_SYMBOL]))
    downloads: list[pd.DataFrame] = []
    mode_by_symbol: dict[str, str] = {}

    for symbol in needed_symbols:
        symbol_history = existing[existing["Symbol"] == symbol] if not existing.empty else pd.DataFrame()
        if _needs_backfill(symbol_history, config.min_lookback_days):
            start = target_date - timedelta(days=config.backfill_calendar_days)
            mode_by_symbol[symbol] = "backfill"
        else:
            last_date = pd.to_datetime(symbol_history["Datetime"]).max().date()
            start = last_date + timedelta(days=1)
            mode_by_symbol[symbol] = "incremental"

        if start > target_date:
            continue

        downloaded = _download_symbol(symbol, start, target_date + timedelta(days=1))
        if not downloaded.empty:
            downloads.append(downloaded)

    new_rows = pd.concat(downloads, ignore_index=True) if downloads else pd.DataFrame(columns=_price_columns())
    combined = _upsert_prices(existing, new_rows)
    if not combined.empty:
        combined.to_parquet(raw_store, index=False)

    manifest = {
        "as_of_date": target_date.isoformat(),
        "raw_store": str(raw_store),
        "new_rows": int(len(new_rows)),
        "symbols": config.symbols,
        "context_symbol": MARKET_CONTEXT_SYMBOL,
        "mode_by_symbol": mode_by_symbol,
        "stage_dir": str(manifest_path.parent),
    }
    _write_json(manifest_path, manifest)
    return manifest


def clean_validate_prices(extract_manifest: dict[str, Any]) -> dict[str, Any]:
    """Clean OHLCV data and validate enough history exists for inference."""
    config = PipelineConfig.from_env()
    target_date = _parse_date(extract_manifest["as_of_date"])
    raw = _read_parquet(Path(extract_manifest["raw_store"]))
    if raw.empty:
        raise PipelineValidationError("No EOD price rows are available after extraction.")

    clean = _clean_prices(raw, target_date)
    _validate_price_history(clean, config.symbols, config.min_lookback_days)

    clean_store = config.data_dir / "clean" / "prices.parquet"
    clean.to_parquet(clean_store, index=False)
    manifest = {
        **extract_manifest,
        "clean_store": str(clean_store),
        "clean_rows": int(len(clean)),
        "max_clean_date": _max_date_iso(clean),
    }
    _write_json(_stage_dir(config, target_date) / "clean_manifest.json", manifest)
    return manifest


def engineer_features(clean_manifest: dict[str, Any]) -> dict[str, Any]:
    """Build inference features with the shared notebook-derived function."""
    config = PipelineConfig.from_env()
    target_date = _parse_date(clean_manifest["as_of_date"])
    clean = _read_parquet(Path(clean_manifest["clean_store"]))
    spy = clean[clean["Symbol"] == MARKET_CONTEXT_SYMBOL].sort_values("Datetime")
    prices = clean[clean["Symbol"].isin(config.symbols)].sort_values(["Symbol", "Datetime"])
    if spy.empty:
        raise PipelineValidationError(f"Missing {MARKET_CONTEXT_SYMBOL} market context history.")
    if prices.empty:
        raise PipelineValidationError("Missing target symbol price history.")

    features = compute_price_features(prices, spy.set_index("Datetime")["Close"], drop_incomplete=True)
    if features.empty:
        raise PipelineValidationError("Feature engineering produced no complete rows.")

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

    missing_features = [name for name in PRICE_FEATURE_COLUMNS if name not in run_features.columns]
    if missing_features:
        raise PipelineValidationError(f"Missing feature columns: {missing_features}")

    feature_store = config.data_dir / "features" / "stock_price_features.parquet"
    existing = _read_parquet(feature_store)
    run_features["feature_version"] = config.feature_version
    run_features["process_date"] = pd.Timestamp.utcnow().tz_localize(None)
    all_features = _upsert_by_keys(
        existing,
        run_features,
        keys=["Datetime", "Symbol", "feature_version"],
    )
    all_features.to_parquet(feature_store, index=False)

    batch_path = _stage_dir(config, target_date) / "features.parquet"
    run_features.to_parquet(batch_path, index=False)
    manifest = {
        **clean_manifest,
        "feature_store": str(feature_store),
        "feature_batch": str(batch_path),
        "feature_rows": int(len(run_features)),
        "feature_version": config.feature_version,
        "feature_columns": list(PRICE_FEATURE_COLUMNS),
    }
    _write_json(_stage_dir(config, target_date) / "feature_manifest.json", manifest)
    return manifest


def run_ml_inference(feature_manifest: dict[str, Any]) -> dict[str, Any]:
    """Load model artifacts, verify feature order, and score the feature batch."""
    config = PipelineConfig.from_env()
    target_date = _parse_date(feature_manifest["as_of_date"])
    features = _read_parquet(Path(feature_manifest["feature_batch"]))
    if features.empty:
        raise PipelineValidationError("No feature rows to score.")

    model_a = _load_model_artifact(config.model_a_path, required=True)
    model_c = _load_model_artifact(config.model_c_path, required=config.require_risk_model)
    _validate_model_columns(model_a, PRICE_FEATURE_COLUMNS, "Model A")
    if model_c is not None:
        _validate_model_columns(model_c, PRICE_FEATURE_COLUMNS, "Model C")

    x = features[list(PRICE_FEATURE_COLUMNS)].astype(float)
    pred_a = np.asarray(model_a["model"].predict(x), dtype=float)
    risk_prob = _predict_risk(model_c, x)
    if risk_prob is None:
        risk_prob = np.full(shape=len(pred_a), fill_value=np.nan, dtype=float)

    output = features[["Datetime", "Symbol"]].copy()
    output["model_version"] = _model_version(model_a, model_c)
    output["pred_a"] = pred_a
    output["risk_prob"] = risk_prob
    output["final_score"] = np.where(np.isnan(risk_prob), pred_a, pred_a * (1 - risk_prob))
    output["feature_version"] = feature_manifest["feature_version"]
    output["source_feature_process_date"] = features["process_date"]
    output["process_date"] = pd.Timestamp.utcnow().tz_localize(None)

    batch_path = _stage_dir(config, target_date) / "predictions.parquet"
    output.to_parquet(batch_path, index=False)
    manifest = {
        **feature_manifest,
        "prediction_batch": str(batch_path),
        "prediction_rows": int(len(output)),
        "model_version": output["model_version"].iloc[0],
    }
    _write_json(_stage_dir(config, target_date) / "inference_manifest.json", manifest)
    return manifest


def save_predictions(prediction_manifest: dict[str, Any]) -> dict[str, Any]:
    """Persist prediction results idempotently and update pipeline state."""
    config = PipelineConfig.from_env()
    target_date = _parse_date(prediction_manifest["as_of_date"])
    predictions = _read_parquet(Path(prediction_manifest["prediction_batch"]))
    if predictions.empty:
        raise PipelineValidationError("No predictions to save.")

    prediction_store = config.data_dir / "predictions" / "stock_predictions.parquet"
    existing = _read_parquet(prediction_store)
    all_predictions = _upsert_by_keys(
        existing,
        predictions,
        keys=["Datetime", "Symbol", "model_version", "feature_version"],
    )
    all_predictions.to_parquet(prediction_store, index=False)

    clean = _read_parquet(Path(prediction_manifest["clean_store"]))
    state = {
        "initialized": True,
        "last_successful_as_of_date": target_date.isoformat(),
        "last_successful_market_date": _max_date_iso(clean),
        "symbols": {
            symbol: {
                "rows": int(len(clean[clean["Symbol"] == symbol])),
                "last_date": _max_date_iso(clean[clean["Symbol"] == symbol]),
            }
            for symbol in sorted(set(config.symbols + [MARKET_CONTEXT_SYMBOL]))
        },
        "feature_version": prediction_manifest["feature_version"],
        "model_version": prediction_manifest["model_version"],
        "prediction_store": str(prediction_store),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    _write_json(config.data_dir / "state.json", state)
    manifest = {
        **prediction_manifest,
        "prediction_store": str(prediction_store),
        "saved_prediction_rows": int(len(predictions)),
        "state_path": str(config.data_dir / "state.json"),
    }
    _write_json(_stage_dir(config, target_date) / "save_manifest.json", manifest)
    return manifest


def _parse_symbols(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_SYMBOLS.copy()
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    return sorted(set(symbols))


def _optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if str(path) else None


def _ensure_dirs(config: PipelineConfig) -> None:
    for name in ["raw", "clean", "features", "predictions", "staging"]:
        (config.data_dir / name).mkdir(parents=True, exist_ok=True)


def _stage_dir(config: PipelineConfig, target_date: date) -> Path:
    path = config.data_dir / "staging" / target_date.strftime("%Y%m%d")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def _price_columns() -> list[str]:
    return ["Datetime", "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _needs_backfill(symbol_history: pd.DataFrame, min_lookback_days: int) -> bool:
    if symbol_history.empty:
        return True
    return len(symbol_history.drop_duplicates("Datetime")) < min_lookback_days


def _download_symbol(symbol: str, start: date, end: date) -> pd.DataFrame:
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

    frame = frame.reset_index()
    if "Date" in frame.columns:
        frame = frame.rename(columns={"Date": "Datetime"})
    frame["Datetime"] = pd.to_datetime(frame["Datetime"]).dt.tz_localize(None).dt.normalize()
    frame["Symbol"] = symbol
    for column in _price_columns():
        if column not in frame.columns:
            frame[column] = 0 if column in ["Dividends", "Stock Splits"] else np.nan
    return frame[_price_columns()]


def _upsert_prices(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        combined = new_rows.copy()
    elif new_rows.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    if combined.empty:
        return combined
    combined["Datetime"] = pd.to_datetime(combined["Datetime"]).dt.tz_localize(None).dt.normalize()
    combined = combined.drop_duplicates(["Datetime", "Symbol"], keep="last")
    return combined.sort_values(["Symbol", "Datetime"]).reset_index(drop=True)


def _clean_prices(raw: pd.DataFrame, target_date: date) -> pd.DataFrame:
    clean = raw.copy()
    clean["Datetime"] = pd.to_datetime(clean["Datetime"], errors="coerce").dt.tz_localize(None).dt.normalize()
    clean["Symbol"] = clean["Symbol"].astype(str).str.upper().str.strip()
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(subset=["Datetime", "Symbol", "Open", "High", "Low", "Close", "Volume"])
    clean = clean[clean["Datetime"].dt.date <= target_date]
    clean = clean[(clean["Open"] > 0) & (clean["High"] > 0) & (clean["Low"] > 0) & (clean["Close"] > 0)]
    clean = clean[clean["High"] >= clean["Low"]]
    clean = clean[clean["Volume"] >= 0]
    clean = clean.drop_duplicates(["Datetime", "Symbol"], keep="last")
    return clean.sort_values(["Symbol", "Datetime"]).reset_index(drop=True)


def _validate_price_history(clean: pd.DataFrame, symbols: list[str], min_lookback_days: int) -> None:
    needed_symbols = sorted(set(symbols + [MARKET_CONTEXT_SYMBOL]))
    missing_symbols = sorted(set(needed_symbols) - set(clean["Symbol"].unique()))
    if missing_symbols:
        raise PipelineValidationError(f"Missing symbols after cleaning: {missing_symbols}")

    short_history = {
        symbol: int(len(clean[clean["Symbol"] == symbol].drop_duplicates("Datetime")))
        for symbol in needed_symbols
        if len(clean[clean["Symbol"] == symbol].drop_duplicates("Datetime")) < min_lookback_days
    }
    if short_history:
        raise PipelineValidationError(
            f"Not enough lookback history for feature inference. Required {min_lookback_days}; got {short_history}"
        )


def _max_date_iso(frame: pd.DataFrame) -> str | None:
    if frame.empty or "Datetime" not in frame.columns:
        return None
    return pd.to_datetime(frame["Datetime"]).max().date().isoformat()


def _upsert_by_keys(existing: pd.DataFrame, updates: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing.empty:
        combined = updates.copy()
    else:
        combined = pd.concat([existing, updates], ignore_index=True)
    for key in keys:
        if key == "Datetime":
            combined[key] = pd.to_datetime(combined[key]).dt.tz_localize(None).dt.normalize()
    return combined.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def _load_model_artifact(path: Path | None, *, required: bool) -> dict[str, Any] | None:
    if path is None:
        if required:
            raise FileNotFoundError("Required model path is not configured.")
        return None
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required model artifact does not exist: {path}")
        return None

    try:
        import joblib

        artifact = joblib.load(path)
    except Exception:
        with path.open("rb") as file:
            artifact = pickle.load(file)

    if isinstance(artifact, dict):
        if "model" not in artifact:
            raise PipelineValidationError(f"Model artifact {path} is a dict but has no 'model' key.")
        artifact.setdefault("path", str(path))
        return artifact
    return {"model": artifact, "feature_columns": list(PRICE_FEATURE_COLUMNS), "model_version": path.stem, "path": str(path)}


def _validate_model_columns(artifact: dict[str, Any], expected_columns: list[str], name: str) -> None:
    artifact_columns = artifact.get("feature_columns")
    if artifact_columns is None:
        return
    if list(artifact_columns) != list(expected_columns):
        raise PipelineValidationError(
            f"{name} feature contract mismatch. Train and serve columns must match exactly."
        )


def _predict_risk(model_c: dict[str, Any] | None, x: pd.DataFrame) -> np.ndarray | None:
    if model_c is None:
        return None
    model = model_c["model"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return np.asarray(proba[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def _model_version(model_a: dict[str, Any], model_c: dict[str, Any] | None) -> str:
    version_a = str(model_a.get("model_version") or Path(model_a["path"]).stem)
    if model_c is None:
        return version_a
    version_c = str(model_c.get("model_version") or Path(model_c["path"]).stem)
    return f"{version_a}+{version_c}"
