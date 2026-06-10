from __future__ import annotations

import json
import os
import subprocess
import sys
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
        _run_local_pipeline(expected_date)
    except Exception as exc:
        fallback = _latest_prediction_path()
        return InferenceAvailability(expected_date, fallback, refreshed=False, refresh_error=str(exc))

    return InferenceAvailability(expected_date, _prediction_path_for_date(expected_date), refreshed=True)


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
    normalized["Pred_A"] = pd.to_numeric(_first_existing(normalized, ["Pred_A", "pred_a"]), errors="coerce")

    risk = pd.to_numeric(_first_existing(normalized, ["Risk_Prob_%", "risk_prob", "RiskProb"]), errors="coerce")
    if risk.dropna().le(1).all():
        risk = risk * 100
    normalized["Risk_Prob_%"] = risk

    normalized["FinalScore"] = pd.to_numeric(
        _first_existing(normalized, ["FinalScore", "final_score"]),
        errors="coerce",
    )
    if normalized["FinalScore"].isna().all():
        normalized["FinalScore"] = normalized["Pred_A"] * (1 - normalized["Risk_Prob_%"] / 100)

    result = normalized[["Date", "Ticker", "Entry_Price", "Pred_A", "Risk_Prob_%", "FinalScore"]]
    result = result.dropna(subset=["Date", "Ticker", "Pred_A", "Risk_Prob_%", "FinalScore"])
    result = result[
        (result["Pred_A"] >= MIN_PRED_A)
        & (result["Risk_Prob_%"] <= MAX_RISK_PROB_PCT)
    ]
    result = result.sort_values(["Date", "FinalScore"], ascending=[False, False])
    return result.head(limit).reset_index(drop=True)


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

    prediction_path = _prediction_path_from_manifest_signal_date(signal_date)
    if prediction_path is not None:
        return prediction_path

    return None


def _prediction_path_from_manifest(path: Path) -> Path | None:
    if not path.exists():
        return None
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return _prediction_path_from_manifest_payload(path, payload)


def _prediction_path_from_manifest_payload(path: Path, payload: dict[str, Any]) -> Path | None:
    prediction_batch = payload.get("prediction_batch")
    if not prediction_batch:
        return None
    candidate = Path(str(prediction_batch))
    if candidate.exists():
        return candidate
    fallback = _fallback_local_path(candidate)
    return fallback if fallback.exists() else None


def _prediction_path_from_manifest_signal_date(signal_date: date) -> Path | None:
    data_dirs = [_data_dir()]
    if data_dirs[0] == LOCAL_CONTAINER_DATA_DIR:
        data_dirs.append(DEFAULT_LOCAL_DATA_DIR)

    for data_dir in data_dirs:
        manifests = sorted((data_dir / "staging").glob("*/inference_manifest.json"), reverse=True)
        for manifest in manifests:
            try:
                payload: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            if _manifest_signal_date(payload) != signal_date:
                continue

            prediction_path = _prediction_path_from_manifest_payload(manifest, payload)
            if prediction_path is not None:
                return prediction_path

    return None


def _manifest_signal_date(payload: dict[str, Any]) -> date | None:
    for key in ("max_clean_date", "signal_date", "as_of_signal_date"):
        value = payload.get(key)
        if not value:
            continue
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            continue
    return None


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
