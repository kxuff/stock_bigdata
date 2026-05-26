from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eod_inference.config import PipelineConfig
from eod_inference.exceptions import PipelineValidationError
from eod_inference.feature_contract import PRICE_FEATURE_COLUMNS
from eod_inference.utils import parse_date, read_parquet, stage_dir, write_json


def run_ml_inference(feature_manifest: dict[str, Any]) -> dict[str, Any]:
    config = PipelineConfig.from_env()
    target_date = parse_date(feature_manifest["as_of_date"])
    features = read_parquet(Path(feature_manifest["feature_batch"]))
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

    batch_path = stage_dir(config, target_date) / "predictions.parquet"
    output.to_parquet(batch_path, index=False)
    manifest = {
        **feature_manifest,
        "prediction_batch": str(batch_path),
        "prediction_rows": int(len(output)),
        "model_version": output["model_version"].iloc[0],
    }
    write_json(stage_dir(config, target_date) / "inference_manifest.json", manifest)
    return manifest


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
