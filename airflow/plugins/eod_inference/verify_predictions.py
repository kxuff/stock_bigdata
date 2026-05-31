from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

for plugin_path in [Path("/opt/airflow/plugins"), Path(__file__).resolve().parents[1]]:
    if plugin_path.exists() and str(plugin_path) not in sys.path:
        sys.path.insert(0, str(plugin_path))

from eod_inference.config import PipelineConfig
from eod_inference.feature_contract import PRICE_FEATURE_COLUMNS
from eod_inference.inference import _load_model_artifact, _predict_risk, _validate_model_columns
from eod_inference.utils import parse_date, read_parquet, stage_dir


LOCAL_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "eod_batch"
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "models"


def recompute_predictions(features: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if features.empty:
        raise ValueError("Feature batch is empty.")

    model_a = _load_model_artifact(_resolve_model_path(config.model_a_path, required=True), required=True)
    model_c_path = _resolve_model_path(config.model_c_path, required=config.require_risk_model)
    model_c = _load_model_artifact(model_c_path, required=config.require_risk_model) if model_c_path is not None else None
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
    output["entry_price"] = pd.to_numeric(features["Close"], errors="coerce")
    output["pred_a"] = pred_a
    output["risk_prob"] = risk_prob
    output["final_score"] = np.where(np.isnan(risk_prob), pred_a, pred_a * (1 - risk_prob))
    output["feature_version"] = features.get("feature_version", pd.Series([pd.NA] * len(features))).astype("string")
    output["source_feature_process_date"] = features.get("process_date", pd.Series([pd.NA] * len(features)))
    output["process_date"] = pd.Timestamp.utcnow().tz_localize(None)
    return output


def summarize_predictions(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        print("No predictions to summarize.")
        return

    print("Prediction summary:")
    print(predictions[["pred_a", "risk_prob", "final_score"]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

    pred_a = pd.to_numeric(predictions["pred_a"], errors="coerce").dropna()
    if not pred_a.empty:
        print(
            "Pred_A buckets: "
            f"[0.00, 0.05)={(pred_a.lt(0.05).mean()):.2%}, "
            f"[0.05, 0.10]={((pred_a.ge(0.05) & pred_a.lt(0.10)).mean()):.2%}, "
            f"[0.10, 0.20]={((pred_a.ge(0.10) & pred_a.lt(0.20)).mean()):.2%}, "
            f">=0.50={(pred_a.ge(0.50).mean()):.2%}"
        )


def compare_predictions(expected: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    if expected.empty or actual.empty:
        return pd.DataFrame()

    left = expected.copy()
    right = actual.copy()
    for frame in (left, right):
        frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="coerce")
        frame["Symbol"] = frame["Symbol"].astype(str)

    merged = left.merge(
        right,
        on=["Datetime", "Symbol"],
        how="inner",
        suffixes=("_recomputed", "_stored"),
    )
    if merged.empty:
        return merged

    for column in ["pred_a", "risk_prob", "final_score"]:
        merged[f"abs_diff_{column}"] = (pd.to_numeric(merged[f"{column}_recomputed"], errors="coerce") - pd.to_numeric(merged[f"{column}_stored"], errors="coerce")).abs()
    return merged


def _model_version(model_a: dict[str, Any], model_c: dict[str, Any] | None) -> str:
    version_a = str(model_a.get("model_version") or Path(model_a["path"]).stem)
    if model_c is None:
        return version_a
    version_c = str(model_c.get("model_version") or Path(model_c["path"]).stem)
    return f"{version_a}+{version_c}"


def _resolve_stage_dir(config: PipelineConfig, run_date: str | None, stage_path: str | None) -> Path:
    if stage_path:
        return Path(stage_path)
    if run_date:
        candidate = config.data_dir / "staging" / parse_date(run_date).strftime("%Y%m%d")
        if candidate.exists() or config.data_dir.exists():
            return stage_dir(config, parse_date(run_date))
        return LOCAL_DATA_DIR / "staging" / parse_date(run_date).strftime("%Y%m%d")

    staging_dir = config.data_dir / "staging"
    manifests = sorted(staging_dir.glob("*/inference_manifest.json"), reverse=True)
    if manifests:
        return manifests[0].parent

    local_staging_dir = LOCAL_DATA_DIR / "staging"
    manifests = sorted(local_staging_dir.glob("*/inference_manifest.json"), reverse=True)
    if manifests:
        return manifests[0].parent

    feature_manifests = sorted(staging_dir.glob("*/feature_manifest.json"), reverse=True)
    if feature_manifests:
        return feature_manifests[0].parent

    feature_manifests = sorted(local_staging_dir.glob("*/feature_manifest.json"), reverse=True)
    if feature_manifests:
        return feature_manifests[0].parent

    raise FileNotFoundError(f"No staging batches found under {staging_dir}")


def _resolve_model_path(path: Path | None, *, required: bool) -> Path | None:
    if path is None:
        if required:
            raise FileNotFoundError("Required model path is not configured.")
        return None

    if path.exists():
        return path

    local_candidate = LOCAL_MODEL_DIR / path.name
    if local_candidate.exists():
        return local_candidate

    if required:
        raise FileNotFoundError(f"Model artifact not found at {path} or {local_candidate}")
    return None


def run(run_date: str | None, stage_path: str | None, output_path: str | None, show_rows: int) -> dict[str, Any]:
    config = PipelineConfig.from_env()
    batch_dir = _resolve_stage_dir(config, run_date, stage_path)
    feature_path = batch_dir / "features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature batch: {feature_path}")

    features = read_parquet(feature_path)
    predictions = recompute_predictions(features, config)
    summarize_predictions(predictions)

    stored_path = batch_dir / "predictions.parquet"
    stored = read_parquet(stored_path) if stored_path.exists() else pd.DataFrame()
    comparison = compare_predictions(predictions, stored)

    if not comparison.empty:
        print("\nComparison against stored predictions:")
        print(
            comparison[
                [
                    "Datetime",
                    "Symbol",
                    "pred_a_recomputed",
                    "pred_a_stored",
                    "abs_diff_pred_a",
                    "risk_prob_recomputed",
                    "risk_prob_stored",
                    "abs_diff_risk_prob",
                    "final_score_recomputed",
                    "final_score_stored",
                    "abs_diff_final_score",
                ]
            ]
            .head(show_rows)
            .to_string(index=False)
        )
        print("\nMax absolute diffs:")
        for column in ["pred_a", "risk_prob", "final_score"]:
            diff = pd.to_numeric(comparison[f"abs_diff_{column}"], errors="coerce")
            print(
                f"  {column}: max={diff.max():.8f}, mean={diff.mean():.8f}"
            )
    else:
        print("\nNo stored predictions available for comparison.")

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(output, index=False)
        print(f"\nWrote recomputed predictions to: {output}")

    return {
        "batch_dir": str(batch_dir),
        "feature_path": str(feature_path),
        "stored_prediction_path": str(stored_path) if stored_path.exists() else None,
        "recomputed_rows": int(len(predictions)),
        "comparison_rows": int(len(comparison)),
        "model_version": predictions["model_version"].iloc[0] if not predictions.empty else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute and verify latest EOD model predictions.")
    parser.add_argument("--run-date", default=None, help="As-of date for the batch, e.g. 2026-05-29.")
    parser.add_argument("--stage-dir", default=None, help="Explicit staging directory containing features.parquet.")
    parser.add_argument("--output", default=None, help="Optional path to write recomputed predictions.parquet.")
    parser.add_argument("--show-rows", type=int, default=10, help="Rows to display in the stored-vs-recomputed comparison.")
    args = parser.parse_args()

    result = run(args.run_date, args.stage_dir, args.output, args.show_rows)
    print("\nRun summary:")
    print(pd.Series(result).to_string())


if __name__ == "__main__":
    main()
