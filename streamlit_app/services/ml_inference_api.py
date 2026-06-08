from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests


DEFAULT_LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "eod_batch"
LOCAL_CONTAINER_DATA_DIR = Path("/opt/airflow/data/eod_batch")
MIN_PRED_A = 0.06
MAX_RISK_PROB = 0.3
MAX_RISK_PROB_PCT = MAX_RISK_PROB * 100
MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 15)
AUTO_REFRESH_ENV = "ML_INFERENCE_AUTO_REFRESH"
PIPELINE_TIMEOUT_SECONDS = int(os.getenv("ML_INFERENCE_PIPELINE_TIMEOUT_SECONDS", "1800"))


@dataclass(frozen=True)
class InferenceAvailability:
    expected_signal_date: date
    prediction_path: Path | None
    refreshed: bool
    refresh_error: str | None = None


def ml_inference_url() -> str | None:
    value = os.getenv("ML_INFERENCE_PICKS_URL", "").strip()
    return value or None


def fetch_ml_inference_picks(limit: int = 25, timeout: float = 10.0) -> pd.DataFrame:
    """Return latest model picks in the display contract used by AI Stock Picks."""
    endpoint = ml_inference_url()
    if endpoint:
        response = requests.get(endpoint, params={"limit": limit}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("data", payload) if isinstance(payload, dict) else payload
        return normalize_ml_inference_picks(pd.DataFrame(records), limit=limit)

    availability = ensure_latest_ml_inference()
    if availability.refresh_error and availability.prediction_path is None:
        raise RuntimeError(availability.refresh_error)

    prediction_path = availability.prediction_path
    if prediction_path is None:
        return _empty_picks_frame()
    return normalize_ml_inference_picks(pd.read_parquet(prediction_path), limit=limit)


def ensure_latest_ml_inference(today: date | None = None) -> InferenceAvailability:
    expected_date = latest_completed_market_date(today=today)
    prediction_path = _prediction_path_for_date(expected_date)
    if prediction_path is not None:
        return InferenceAvailability(expected_date, prediction_path, refreshed=False)

    if os.getenv(AUTO_REFRESH_ENV, "true").lower() in {"0", "false", "no"}:
        return InferenceAvailability(expected_date, _latest_prediction_path(), refreshed=False)

    try:
        # Try direct inference first (lightweight, no pyspark needed)
        _run_direct_inference(expected_date)
    except Exception as exc:
        direct_error = exc
        try:
            _run_local_feature_generation(expected_date)
            _run_direct_inference(expected_date)
        except Exception as local_exc:
            direct_error = local_exc
        else:
            prediction_path = _prediction_path_for_date(expected_date)
            if prediction_path is not None:
                return InferenceAvailability(expected_date, prediction_path, refreshed=True)

        # If local feature generation fails, try full Spark/Iceberg pipeline.
        try:
            _run_local_pipeline(expected_date)
        except Exception as pipeline_exc:
            fallback = _latest_prediction_path()
            error_msg = f"Direct/local inference failed: {direct_error}. Pipeline failed: {pipeline_exc}"
            return InferenceAvailability(expected_date, fallback, refreshed=False, refresh_error=error_msg)

    # Verify predictions were created
    prediction_path = _prediction_path_for_date(expected_date)
    if prediction_path is not None:
        return InferenceAvailability(expected_date, prediction_path, refreshed=True)
    
    fallback = _latest_prediction_path()
    return InferenceAvailability(expected_date, fallback, refreshed=False, refresh_error="Predictions not created after inference")


def latest_completed_market_date(today: date | None = None, now: datetime | None = None) -> date:
    if today is None:
        current = now.astimezone(MARKET_TZ) if now is not None else datetime.now(MARKET_TZ)
        candidate = current.date()
        if current.time() < MARKET_CLOSE:
            candidate -= timedelta(days=1)
    else:
        candidate = today

    while not _is_market_session(candidate):
        candidate -= timedelta(days=1)
    return candidate


def normalize_ml_inference_picks(frame: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    if frame.empty:
        return _empty_picks_frame()

    normalized = frame.copy()
    normalized["Date"] = pd.to_datetime(
        _first_existing(normalized, ["Date", "Datetime", "date", "datetime"]),
        errors="coerce",
    ).dt.date.astype("string")
    normalized["Ticker"] = _first_existing(normalized, ["Ticker", "Symbol", "ticker", "symbol"]).astype("string")
    normalized["Entry_Price"] = pd.to_numeric(
        _first_existing(normalized, ["Entry_Price", "entry_price", "Close", "close"]),
        errors="coerce",
    )
    # Pred_A is in decimal format (0.05 = 5% upside)
    normalized["Pred_A"] = pd.to_numeric(_first_existing(normalized, ["Pred_A", "pred_a"]), errors="coerce")

    # Risk_Prob: ensure it's in percentage format (0-100)
    risk = pd.to_numeric(_first_existing(normalized, ["Risk_Prob_%", "risk_prob", "RiskProb"]), errors="coerce")
    if risk.dropna().le(1).all():
        # Convert from decimal [0,1] to percentage [0,100]
        risk = risk * 100
    normalized["Risk_Prob_%"] = risk

    # FinalScore: ensure it's in decimal format (0-1) to match Pred_A
    final_score = pd.to_numeric(
        _first_existing(normalized, ["FinalScore", "final_score"]),
        errors="coerce",
    )
    # If FinalScore is missing or in percentage format, recalculate from components
    if final_score.isna().all() or final_score.gt(1).any():
        # Recalculate: final_score = pred_a * (1 - risk_prob_decimal)
        final_score = normalized["Pred_A"] * (1 - normalized["Risk_Prob_%"] / 100)
    # Ensure FinalScore is in decimal format
    elif final_score.gt(1).any():
        # If values are > 1, assume they're in percentage format, convert to decimal
        final_score = final_score / 100
    normalized["FinalScore"] = final_score

    result = normalized[["Date", "Ticker", "Entry_Price", "Pred_A", "Risk_Prob_%", "FinalScore"]]
    result = result.dropna(subset=["Date", "Ticker", "Pred_A", "Risk_Prob_%", "FinalScore"])
    result = result[
        (result["Pred_A"] >= MIN_PRED_A)
        & (result["Risk_Prob_%"] <= MAX_RISK_PROB_PCT)
    ]
    result = result.sort_values(["Date", "FinalScore"], ascending=[False, False])
    return result.head(limit).reset_index(drop=True)


def _run_direct_inference(run_date: date) -> None:
    """Run ML inference directly without full pipeline (no pyspark needed)."""
    import joblib
    import pandas as pd
    
    feature_dir = _feature_stage_dir_for_date(run_date)
    feature_path = feature_dir / "features.parquet"
    manifest_path = feature_dir / "feature_manifest.json"
    if not feature_path.exists():
        latest_manifest = _latest_feature_manifest()
        latest_date = None
        if latest_manifest is not None:
            try:
                latest_payload = json.loads(latest_manifest.read_text(encoding="utf-8"))
                latest_date = latest_payload.get("as_of_date")
            except Exception:
                latest_date = latest_manifest.parent.name
        detail = f" Latest features are from {latest_date}." if latest_date else ""
        raise FileNotFoundError(f"No features found for {run_date.isoformat()}.{detail} Run feature engineering first.")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing feature_manifest.json in {feature_dir}")
    
    # Load features and manifest
    features_df = pd.read_parquet(feature_path)
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_as_of_date = feature_manifest['as_of_date']
    
    if pd.to_datetime(feature_as_of_date).date() != run_date:
        raise ValueError(
            f"Feature manifest date {feature_as_of_date} does not match requested predictions for {run_date}."
        )
    
    # Load models
    model_a = joblib.load(_repo_root() / "data" / "models" / "model_a.joblib")
    model_c = joblib.load(_repo_root() / "data" / "models" / "model_c.joblib")
    
    # Run inference
    expected_features = model_a['feature_columns']
    X_test = features_df.reindex(columns=expected_features).astype(float)
    pred_a = model_a['model'].predict(X_test)
    pred_c_proba = model_c['model'].predict_proba(X_test)[:, 1]
    
    # Create output
    output = features_df[["Datetime", "Symbol"]].copy()
    output["model_version"] = model_a['model_version'] + "+" + model_c['model_version']
    output["entry_price"] = pd.to_numeric(features_df["Close"], errors='coerce')
    output["pred_a"] = pred_a
    output["risk_prob"] = pred_c_proba
    output["final_score"] = pred_a * (1 - pred_c_proba)
    output["feature_version"] = feature_manifest["feature_version"]
    output["source_feature_process_date"] = features_df.get("process_date", pd.NaT)
    output["process_date"] = pd.Timestamp.now('UTC').tz_localize(None)
    
    # Save predictions using the feature date
    target_date = pd.to_datetime(feature_as_of_date).date()
    batch_dir = feature_dir.parent / target_date.isoformat().replace('-', '')
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = batch_dir / "predictions.parquet"
    
    output.to_parquet(batch_path, index=False)
    
    # Save manifest
    manifest = {
        **feature_manifest,
        "prediction_batch": str(batch_path),
        "prediction_rows": int(len(output)),
        "model_version": output["model_version"].iloc[0],
    }
    manifest_path = batch_dir / "inference_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)


def _feature_stage_dir_for_date(signal_date: date) -> Path:
    stage = _stage_dir(signal_date)
    if (stage / "features.parquet").exists() or (stage / "feature_manifest.json").exists():
        return stage
    if _data_dir() == LOCAL_CONTAINER_DATA_DIR:
        local_stage = DEFAULT_LOCAL_DATA_DIR / "staging" / signal_date.strftime("%Y%m%d")
        if (local_stage / "features.parquet").exists() or (local_stage / "feature_manifest.json").exists():
            return local_stage
    return stage


def _run_local_feature_generation(run_date: date) -> None:
    """Create a staging feature batch locally without Spark/Iceberg."""
    import numpy as np
    import yfinance as yf

    plugin_path = _repo_root() / "airflow" / "plugins"
    if str(plugin_path) not in sys.path:
        sys.path.insert(0, str(plugin_path))

    from eod_inference.config import (  # noqa: PLC0415
        DEFAULT_SYMBOLS,
        FEATURE_VERSION,
        LEGACY_SYMBOL_ALIASES,
        MARKET_CONTEXT_SYMBOL,
    )
    from eod_inference.feature_contract import PRICE_FEATURE_COLUMNS, compute_price_features  # noqa: PLC0415

    symbols = _local_feature_symbols(DEFAULT_SYMBOLS, LEGACY_SYMBOL_ALIASES)
    needed_symbols = sorted(set(symbols + [MARKET_CONTEXT_SYMBOL]))
    start_date = run_date - timedelta(days=int(os.getenv("US_STOCK_BACKFILL_CALENDAR_DAYS", "500")))
    end_date = run_date + timedelta(days=1)

    price_frames = _download_local_price_history(yf, needed_symbols, start_date, end_date)
    if not price_frames:
        raise RuntimeError(f"No EOD prices downloaded for local feature generation through {run_date.isoformat()}.")

    prices = pd.concat(price_frames, ignore_index=True)
    prices["Datetime"] = pd.to_datetime(prices["Datetime"], errors="coerce").dt.tz_localize(None).dt.normalize()
    prices["Symbol"] = prices["Symbol"].astype(str).str.upper().str.strip()
    prices = prices.dropna(subset=["Datetime", "Symbol", "Open", "High", "Low", "Close"])

    latest_price_date = pd.to_datetime(prices["Datetime"]).max().date()
    if latest_price_date < run_date:
        raise RuntimeError(
            f"Downloaded EOD prices only through {latest_price_date.isoformat()}, "
            f"so features for {run_date.isoformat()} cannot be generated yet."
        )

    spy = prices[prices["Symbol"] == MARKET_CONTEXT_SYMBOL].sort_values("Datetime")
    targets = prices[prices["Symbol"].isin(symbols)].sort_values(["Symbol", "Datetime"])
    if spy.empty:
        raise RuntimeError(f"Missing {MARKET_CONTEXT_SYMBOL} market context history.")
    if targets.empty:
        raise RuntimeError("Missing target symbol price history.")

    feature_frames: list[pd.DataFrame] = []
    spy_close = spy.set_index("Datetime")["Close"]
    for symbol, group in targets.groupby("Symbol"):
        symbol_features = compute_price_features(group.sort_values("Datetime"), spy_close, drop_incomplete=True)
        if not symbol_features.empty:
            symbol_features["Symbol"] = symbol
            feature_frames.append(symbol_features)

    if not feature_frames:
        raise RuntimeError("Local feature engineering produced no complete rows.")

    features = pd.concat(feature_frames, ignore_index=True)
    features["Datetime"] = pd.to_datetime(features["Datetime"], errors="coerce").dt.tz_localize(None).dt.normalize()
    run_features = features[features["Datetime"].dt.date == run_date].copy()
    if run_features.empty:
        latest_feature_date = pd.to_datetime(features["Datetime"]).max().date()
        raise RuntimeError(
            f"No local features found for {run_date.isoformat()}. "
            f"Latest generated features are from {latest_feature_date.isoformat()}."
        )

    missing_features = [name for name in PRICE_FEATURE_COLUMNS if name not in run_features.columns]
    if missing_features:
        raise RuntimeError(f"Missing feature columns: {missing_features}")

    run_features["feature_version"] = os.getenv("ML_FEATURE_VERSION", FEATURE_VERSION)
    run_features["process_date"] = pd.Timestamp.utcnow().tz_localize(None)
    run_features["source_batch_id"] = int(pd.Timestamp.utcnow().timestamp())
    run_features = run_features.replace([np.inf, -np.inf], np.nan).dropna(subset=PRICE_FEATURE_COLUMNS)
    if run_features.empty:
        raise RuntimeError(f"Local features for {run_date.isoformat()} are incomplete after validation.")

    stage = _stage_dir(run_date)
    stage.mkdir(parents=True, exist_ok=True)
    batch_path = stage / "features.parquet"
    run_features.to_parquet(batch_path, index=False)

    manifest = {
        "as_of_date": run_date.isoformat(),
        "feature_batch": str(batch_path),
        "feature_rows": int(len(run_features)),
        "feature_version": os.getenv("ML_FEATURE_VERSION", FEATURE_VERSION),
        "feature_columns": list(PRICE_FEATURE_COLUMNS),
        "symbols": symbols,
        "context_symbol": MARKET_CONTEXT_SYMBOL,
        "stage_dir": str(stage),
        "new_rows": int(len(prices)),
        "local_pandas_generated": True,
        "max_clean_date": latest_price_date.isoformat(),
    }
    (stage / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _local_feature_symbols(default_symbols: list[str], aliases: dict[str, str | None]) -> list[str]:
    configured = os.getenv("US_STOCK_EOD_SYMBOLS", "").strip()
    if configured:
        raw_symbols = [item.strip().upper() for item in configured.split(",") if item.strip()]
    else:
        latest_manifest = _latest_feature_manifest()
        if latest_manifest is not None:
            try:
                payload = json.loads(latest_manifest.read_text(encoding="utf-8"))
                raw_symbols = [str(item).strip().upper() for item in payload.get("symbols", []) if str(item).strip()]
            except Exception:
                raw_symbols = default_symbols
        else:
            raw_symbols = default_symbols

    normalized: list[str] = []
    for symbol in raw_symbols:
        alias = aliases.get(symbol, symbol)
        if alias:
            normalized.append(alias)
    return sorted(set(normalized))


def _latest_feature_manifest() -> Path | None:
    staging_dir = _data_dir() / "staging"
    manifests = sorted(staging_dir.glob("*/feature_manifest.json"), reverse=True)
    return manifests[0] if manifests else None


def _download_local_price_history(yf_module: Any, symbols: list[str], start: date, end: date) -> list[pd.DataFrame]:
    max_workers = max(1, min(8, int(os.getenv("ML_INFERENCE_DOWNLOAD_WORKERS", "6"))))
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_local_symbol, yf_module, symbol, start, end): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            try:
                frame = future.result()
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)
    return frames


def _download_local_symbol(yf_module: Any, symbol: str, start: date, end: date) -> pd.DataFrame:
    columns = ["Datetime", "Symbol", "Open", "High", "Low", "Close", "Volume"]
    try:
        frame = yf_module.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
            actions=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame(columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")].reset_index()
    if "Date" in frame.columns:
        frame = frame.rename(columns={"Date": "Datetime"})
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="coerce").dt.tz_localize(None).dt.normalize()
    frame["Symbol"] = symbol
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Volume"] = frame["Volume"].fillna(0)
    return frame.reindex(columns=columns)



def _run_local_pipeline(run_date: date) -> None:
    stage = _stage_dir(run_date)
    feature_manifest = stage / "feature_manifest.json"
    script = _repo_root() / "airflow" / "plugins" / "eod_inference" / (
        "run_ml_only.py" if feature_manifest.exists() else "run_eod_pipeline.py"
    )
    args = [sys.executable, str(script), "--run-date", run_date.isoformat()]
    env = _pipeline_env()

    completed = subprocess.run(
        args,
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        timeout=PIPELINE_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Unable to refresh ML inference for {run_date.isoformat()}: {detail}")


def _pipeline_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_repo_root() / "airflow" / "plugins"))
    env.setdefault("US_STOCK_EOD_DATA_DIR", str(_data_dir()))
    local_model_dir = _repo_root() / "data" / "models"
    env.setdefault("US_STOCK_MODEL_A_PATH", str(local_model_dir / "model_a.joblib"))
    env.setdefault("US_STOCK_MODEL_C_PATH", str(local_model_dir / "model_c.joblib"))
    return env


def _latest_prediction_path() -> Path | None:
    explicit_path = os.getenv("ML_INFERENCE_PREDICTIONS_PATH", "").strip()
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path
        fallback = _fallback_local_path(path)
        return fallback if fallback.exists() else None

    manifest_path = os.getenv("ML_INFERENCE_MANIFEST_PATH", "").strip()
    if manifest_path:
        path = Path(manifest_path)
        prediction_path = _prediction_path_from_manifest(path)
        if prediction_path is not None:
            return prediction_path
        fallback = _fallback_local_path(path)
        return _prediction_path_from_manifest(fallback) if fallback.exists() else None

    data_dir = Path(os.getenv("US_STOCK_EOD_DATA_DIR", str(DEFAULT_LOCAL_DATA_DIR)))
    staging_dir = data_dir / "staging"
    manifests = sorted(staging_dir.glob("*/inference_manifest.json"), reverse=True)
    for manifest in manifests:
        prediction_path = _prediction_path_from_manifest(manifest)
        if prediction_path is not None:
            return prediction_path

    if data_dir == LOCAL_CONTAINER_DATA_DIR:
        local_staging_dir = DEFAULT_LOCAL_DATA_DIR / "staging"
        manifests = sorted(local_staging_dir.glob("*/inference_manifest.json"), reverse=True)
        for manifest in manifests:
            prediction_path = _prediction_path_from_manifest(manifest)
            if prediction_path is not None:
                return prediction_path

    predictions = sorted(staging_dir.glob("*/predictions.parquet"), reverse=True)
    if predictions:
        return predictions[0]

    if data_dir == LOCAL_CONTAINER_DATA_DIR:
        local_predictions = sorted((DEFAULT_LOCAL_DATA_DIR / "staging").glob("*/predictions.parquet"), reverse=True)
        return local_predictions[0] if local_predictions else None

    return None


def _prediction_path_for_date(signal_date: date) -> Path | None:
    stage = _stage_dir(signal_date)
    manifest = stage / "inference_manifest.json"
    prediction_path = _prediction_path_from_manifest(manifest)
    if prediction_path is not None:
        return prediction_path

    parquet_path = stage / "predictions.parquet"
    if parquet_path.exists():
        return parquet_path

    if _data_dir() == LOCAL_CONTAINER_DATA_DIR:
        local_stage = DEFAULT_LOCAL_DATA_DIR / "staging" / signal_date.strftime("%Y%m%d")
        manifest = local_stage / "inference_manifest.json"
        prediction_path = _prediction_path_from_manifest(manifest)
        if prediction_path is not None:
            return prediction_path
        parquet_path = local_stage / "predictions.parquet"
        if parquet_path.exists():
            return parquet_path

    return None


def _prediction_path_from_manifest(path: Path) -> Path | None:
    if not path.exists():
        return None
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    prediction_batch = payload.get("prediction_batch")
    if not prediction_batch:
        return None
    candidate = Path(str(prediction_batch))
    if candidate.exists():
        return candidate
    fallback = _fallback_local_path(candidate)
    return fallback if fallback.exists() else None


def _fallback_local_path(path: Path) -> Path:
    parts = list(path.parts)
    if "/opt/airflow" not in path.as_posix():
        return path
    try:
        idx = parts.index("opt")
    except ValueError:
        return path
    if len(parts) <= idx + 2:
        return path
    suffix = Path(*parts[idx + 2 :])
    return Path(__file__).resolve().parents[2] / suffix


def _data_dir() -> Path:
    return Path(os.getenv("US_STOCK_EOD_DATA_DIR", str(DEFAULT_LOCAL_DATA_DIR)))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _stage_dir(signal_date: date) -> Path:
    return _data_dir() / "staging" / signal_date.strftime("%Y%m%d")


def _is_market_session(value: date) -> bool:
    if value.weekday() >= 5:
        return False
    return value not in _market_holidays(value.year)


def _market_holidays(year: int) -> set[date]:
    holidays = {
        _observed(date(year + 1, 1, 1)),
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _good_friday(year),
        _last_weekday(year, 5, 0),
        _observed(date(year, 6, 19)),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    return {holiday for holiday in holidays if holiday.year == year}


def _observed(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _good_friday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day) - timedelta(days=2)


def _first_existing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series([pd.NA] * len(frame), index=frame.index)


def _empty_picks_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["Date", "Ticker", "Entry_Price", "Pred_A", "Risk_Prob_%", "FinalScore"])
