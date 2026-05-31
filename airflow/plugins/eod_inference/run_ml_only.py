from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

for plugin_path in [Path("/opt/airflow/plugins"), Path(__file__).resolve().parents[1]]:
    if plugin_path.exists() and str(plugin_path) not in sys.path:
        sys.path.insert(0, str(plugin_path))

from eod_inference.config import PipelineConfig
from eod_inference.inference import run_ml_inference
from eod_inference.save import save_predictions
from eod_inference.utils import parse_date, stage_dir


def _resolve_feature_manifest_path(config: PipelineConfig, run_date: str | None, stage_path: str | None) -> Path:
    if stage_path:
        candidate = Path(stage_path) / "feature_manifest.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Missing feature manifest: {candidate}")

    if run_date:
        candidate = stage_dir(config, parse_date(run_date)) / "feature_manifest.json"
        if candidate.exists():
            return candidate

    staging_dir = config.data_dir / "staging"
    manifests = sorted(staging_dir.glob("*/feature_manifest.json"), reverse=True)
    if manifests:
        return manifests[0]

    raise FileNotFoundError(f"No feature_manifest.json found under {staging_dir}")


def run(run_date: str | None, stage_path: str | None) -> dict[str, Any]:
    config = PipelineConfig.from_env()
    feature_manifest_path = _resolve_feature_manifest_path(config, run_date, stage_path)
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))

    inference_manifest = run_ml_inference(
        {
            **feature_manifest,
            "skip_agent_context": True,
            "sentiment_context": None,
            "valuation_context": None,
        }
    )
    save_manifest = save_predictions(inference_manifest)
    preview_csv = _write_preview_csv(inference_manifest, feature_manifest_path.parent)

    return {
        "feature_manifest": str(feature_manifest_path),
        "prediction_rows": inference_manifest.get("prediction_rows"),
        "prediction_batch": inference_manifest.get("prediction_batch"),
        "preview_csv": str(preview_csv) if preview_csv is not None else None,
        "prediction_table": save_manifest.get("prediction_table"),
        "saved_rows": save_manifest.get("saved_rows"),
        "model_version": inference_manifest.get("model_version"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML-only inference and save predictions from an existing feature batch.")
    parser.add_argument("--run-date", default=None, help="As-of date, e.g. 2026-05-29.")
    parser.add_argument("--stage-dir", default=None, help="Explicit staging directory containing feature_manifest.json.")
    args = parser.parse_args()

    print(json.dumps(run(args.run_date, args.stage_dir), indent=2, default=str))


def _write_preview_csv(inference_manifest: dict[str, Any], output_dir: Path) -> Path | None:
    prediction_batch = inference_manifest.get("prediction_batch")
    if not prediction_batch:
        return None

    prediction_path = Path(str(prediction_batch))
    if not prediction_path.exists():
        return None

    frame = pd.read_parquet(prediction_path)
    if frame.empty:
        return None

    preview = frame.copy()
    preview["Upside_pct"] = pd.to_numeric(preview["pred_a"], errors="coerce") * 100
    preview["Risk_Prob_pct"] = pd.to_numeric(preview["risk_prob"], errors="coerce") * 100
    preview["FinalScore_raw"] = pd.to_numeric(preview["final_score"], errors="coerce")
    preview["FinalScore_pct_display"] = preview["Upside_pct"] * (1 - preview["Risk_Prob_pct"] / 100)
    preview = preview[
        [
            "Datetime",
            "Symbol",
            "entry_price",
            "pred_a",
            "Upside_pct",
            "risk_prob",
            "Risk_Prob_pct",
            "final_score",
            "FinalScore_raw",
            "FinalScore_pct_display",
        ]
    ].sort_values(["Datetime", "FinalScore_raw"], ascending=[False, False])

    preview_path = output_dir / "predictions_preview.csv"
    preview.to_csv(preview_path, index=False)
    return preview_path


if __name__ == "__main__":
    main()
