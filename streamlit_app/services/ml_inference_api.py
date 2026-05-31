from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "eod_batch"
LOCAL_CONTAINER_DATA_DIR = Path("/opt/airflow/data/eod_batch")
MIN_PRED_A = 0.06
MAX_RISK_PROB = 0.3
MAX_RISK_PROB_PCT = MAX_RISK_PROB * 100


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

    prediction_path = _latest_prediction_path()
    if prediction_path is None:
        return _empty_picks_frame()
    return normalize_ml_inference_picks(pd.read_parquet(prediction_path), limit=limit)


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


def _first_existing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series([pd.NA] * len(frame), index=frame.index)


def _empty_picks_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["Date", "Ticker", "Entry_Price", "Pred_A", "Risk_Prob_%", "FinalScore"])
