from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

for plugin_path in [Path("/opt/airflow/plugins"), Path(__file__).resolve().parents[1]]:
    if plugin_path.exists() and str(plugin_path) not in sys.path:
        sys.path.insert(0, str(plugin_path))

from eod_inference.pipeline import (
    build_agent_context,
    clean_validate_prices,
    engineer_features,
    extract_eod_prices,
    run_ml_inference,
    save_predictions,
)


def run(run_date: str) -> dict[str, Any]:
    extract_manifest = extract_eod_prices(run_date)
    clean_manifest = clean_validate_prices(extract_manifest)
    feature_manifest = engineer_features(clean_manifest)
    context_manifest = (
        {**feature_manifest, **build_agent_context(feature_manifest["as_of_date"])}
        if os.getenv("FINBERT_API_URL", "").strip()
        else feature_manifest
    )
    inference_manifest = run_ml_inference(context_manifest)
    save_manifest = save_predictions(inference_manifest)

    return {
        "run_date": run_date,
        "prediction_rows": inference_manifest.get("prediction_rows"),
        "orca_context_rows": inference_manifest.get("orca_context_rows"),
        "orca_context_includes": inference_manifest.get("orca_context_includes"),
        "orca_context_excludes": inference_manifest.get("orca_context_excludes"),
        "sentiment_rows": inference_manifest.get("sentiment_rows"),
        "valuation_rows": inference_manifest.get("valuation_rows"),
        "sentiment_context": inference_manifest.get("sentiment_context"),
        "valuation_context": inference_manifest.get("valuation_context"),
        "orca_upstream_context": inference_manifest.get("orca_upstream_context"),
        "prediction_batch": inference_manifest.get("prediction_batch"),
        "prediction_table": save_manifest.get("prediction_table"),
        "saved_rows": save_manifest.get("saved_rows"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EOD inference and ORCA upstream context pipeline.")
    parser.add_argument("--run-date", required=True, help="Airflow ds/as-of date, e.g. 2026-05-29")
    args = parser.parse_args()

    print(json.dumps(run(args.run_date), indent=2, default=str))


if __name__ == "__main__":
    main()
