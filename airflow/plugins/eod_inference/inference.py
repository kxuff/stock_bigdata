from __future__ import annotations

import pickle
from pathlib import Path
import os
from typing import Any

import numpy as np
import pandas as pd

from eod_inference.config import PipelineConfig
from eod_inference.exceptions import PipelineValidationError
from eod_inference.feature_contract import PRICE_FEATURE_COLUMNS
from eod_inference.orca_context import write_orca_upstream_context
from eod_inference.utils import parse_date, read_parquet, stage_dir, write_json


def run_ml_inference(feature_manifest: dict[str, Any]) -> dict[str, Any]:
    """Score the feature batch and write the Streamlit stock-pick contract."""
    config = PipelineConfig.from_env()
    target_date = parse_date(feature_manifest["as_of_date"])
    feature_manifest = _with_agent_context(feature_manifest, target_date.isoformat())
    features = read_parquet(Path(feature_manifest["feature_batch"]))
    if features.empty:
        raise PipelineValidationError("No feature rows to score.")

    model_a = _load_model_artifact(config.model_a_path, required=True)
    model_c = _load_model_artifact(config.model_c_path, required=config.require_risk_model)

    expected_features = _artifact_feature_columns(model_a, "Model A")
    _validate_model_columns(model_a, expected_features, "Model A")
    if model_c is not None:
        _validate_model_columns(model_c, expected_features, "Model C")

    # Match the Kaggle notebook: one feature matrix, ordered by model A's
    # training contract, feeds both Model A and Model C.
    x_test_model = features.reindex(columns=expected_features).astype(float)
    pred_a = np.asarray(model_a["model"].predict(x_test_model), dtype=float)
    _validate_upside_output(pred_a, "Model A")

    risk_prob = _predict_risk(model_c, x_test_model)
    if risk_prob is None:
        risk_prob = np.full(shape=len(pred_a), fill_value=np.nan, dtype=float)

    output = features[["Datetime", "Symbol"]].copy()
    output["model_version"] = _model_version(model_a, model_c)
    output["entry_price"] = pd.to_numeric(features["Close"], errors="coerce")
    output["pred_a"] = pred_a
    output["risk_prob"] = risk_prob
    output["final_score"] = np.where(np.isnan(risk_prob), pred_a, pred_a * (1 - risk_prob))
    output["feature_version"] = feature_manifest["feature_version"]
    output["source_feature_process_date"] = features.get("process_date", pd.NaT)
    output["process_date"] = pd.Timestamp.utcnow().tz_localize(None)

    batch_path = stage_dir(config, target_date) / "predictions.parquet"
    output.to_parquet(batch_path, index=False)
    orca_context = write_orca_upstream_context(
        predictions=output,
        features=features,
        stage_path=stage_dir(config, target_date),
        source_ref_prefix=f"{config.ml_ready_prediction_table}:{target_date.isoformat()}",
        sentiment_path=_optional_manifest_path(feature_manifest, "sentiment_context"),
        valuation_path=_optional_manifest_path(feature_manifest, "valuation_context"),
    )
    manifest = {
        **feature_manifest,
        "prediction_batch": str(batch_path),
        "prediction_rows": int(len(output)),
        "model_version": output["model_version"].iloc[0],
        **orca_context,
    }
    write_json(stage_dir(config, target_date) / "inference_manifest.json", manifest)
    return manifest


def _artifact_feature_columns(artifact: dict[str, Any], name: str) -> list[str]:
    feature_columns = artifact.get("feature_columns")
    if not feature_columns:
        raise PipelineValidationError(f"{name} artifact must include feature_columns.")
    return list(feature_columns)


def _load_model_artifact(path: Path | None, *, required: bool) -> dict[str, Any] | None:
    if path is None:
        if required:
            raise FileNotFoundError("Required model path is not configured.")
        return None
    
    clean_path = Path(str(path).strip())

    if not clean_path.exists():
        if required:
            # Dùng repr() để in ra chính xác chuỗi (hiện rõ \r nếu có)
            raise FileNotFoundError(f"Required model artifact does not exist: {repr(str(clean_path))}")
        return None
    
    path = clean_path

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
        risk_prob = np.asarray(proba[:, 1], dtype=float)
    else:
        risk_prob = np.asarray(model.predict(x), dtype=float)
    _validate_probability_output(risk_prob, "Model C")
    return risk_prob


def _validate_probability_output(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise PipelineValidationError(f"{name} must output calibrated probabilities in [0, 1].")


def _validate_upside_output(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise PipelineValidationError(f"{name} must output finite decimal returns.")


def _model_version(model_a: dict[str, Any], model_c: dict[str, Any] | None) -> str:
    version_a = str(model_a.get("model_version") or Path(model_a["path"]).stem)
    if model_c is None:
        return version_a
    version_c = str(model_c.get("model_version") or Path(model_c["path"]).stem)
    return f"{version_a}+{version_c}"


def _optional_manifest_path(manifest: dict[str, Any], key: str) -> Path | None:
    value = manifest.get(key)
    if not value:
        return None
    return Path(str(value))


def _with_agent_context(manifest: dict[str, Any], as_of_date: str) -> dict[str, Any]:
    if manifest.get("skip_agent_context"):
        return manifest
    if manifest.get("sentiment_context") and manifest.get("valuation_context"):
        return manifest
    if not os.getenv("FINBERT_API_URL", "").strip():
        return manifest

    from eod_inference.agent_context import build_agent_context

    return {**manifest, **build_agent_context(as_of_date)}
