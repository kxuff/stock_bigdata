#!/usr/bin/env python3
"""
Simplified direct inference for testing - without using helper functions.
You can modify and test this directly without the abstraction layer.
"""
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import joblib
import pickle

sys.path.insert(0, str(Path(__file__).parent / "airflow" / "plugins"))
from eod_inference.utils import read_parquet, write_json, stage_dir, parse_date
from eod_inference.config import PipelineConfig

def run_ml_inference_direct(feature_manifest: dict[str, Any], 
                             model_a_path: Path = None,
                             model_c_path: Path = None) -> dict[str, Any]:
    """
    Direct inference without helper functions - for testing and debugging.
    This is easier to modify and test than the abstracted version.
    
    Args:
        feature_manifest: Feature batch manifest
        model_a_path: Override Model A path (for testing)
        model_c_path: Override Model C path (for testing)
    """
    # Configuration
    config = PipelineConfig.from_env()
    
    # Use provided paths or config defaults
    if model_a_path is None:
        model_a_path = config.model_a_path
    if model_c_path is None:
        model_c_path = config.model_c_path
    
    target_date = parse_date(feature_manifest["as_of_date"])
    
    # 1. Load features
    print(f"Loading features from: {feature_manifest['feature_batch']}")
    features = read_parquet(Path(feature_manifest["feature_batch"]))
    print(f"Features shape: {features.shape}")
    
    if features.empty:
        raise ValueError("No feature rows to score.")
    
    # 2. Load Model A directly (no helper)
    print(f"\nLoading Model A from: {model_a_path}")
    model_a_path = Path(str(model_a_path).strip())
    
    try:
        model_a_artifact = joblib.load(model_a_path)
    except Exception as e:
        print(f"Joblib failed: {e}, trying pickle...")
        with open(model_a_path, "rb") as f:
            model_a_artifact = pickle.load(f)
    
    model_a = model_a_artifact["model"] if isinstance(model_a_artifact, dict) else model_a_artifact
    model_a_features = model_a_artifact.get("feature_columns", []) if isinstance(model_a_artifact, dict) else []
    
    print(f"Model A type: {type(model_a)}")
    print(f"Model A features needed: {len(model_a_features)}")
    
    # 3. Load Model C directly (no helper)  
    print(f"\nLoading Model C from: {model_c_path}")
    model_c_path = Path(str(model_c_path).strip()) if model_c_path else None
    model_c = None
    model_c_artifact = None
    
    if model_c_path and model_c_path.exists():
        try:
            model_c_artifact = joblib.load(model_c_path)
        except Exception as e:
            print(f"Joblib failed: {e}, trying pickle...")
            with open(model_c_path, "rb") as f:
                model_c_artifact = pickle.load(f)
        
        model_c = model_c_artifact["model"] if isinstance(model_c_artifact, dict) else model_c_artifact
        print(f"Model C loaded, type: {type(model_c)}")
    else:
        print("Model C not found, using NaN for risk_prob")
    
    # 4. Prepare features for Model A
    print(f"\nPreparing features...")
    print(f"Available features: {list(features.columns)}")
    print(f"Expected by Model A: {model_a_features[:5]}... (showing first 5)")
    
    # CRITICAL: Reindex to exact column order and cast to float64
    x_test_model = features.reindex(columns=model_a_features).astype(np.float64)
    print(f"Feature matrix shape: {x_test_model.shape}")
    print(f"NaN count in features: {x_test_model.isna().sum().sum()}")
    
    # 5. Make predictions with Model A
    print(f"\n=== PREDICTING WITH MODEL A ===")
    pred_a = np.asarray(model_a.predict(x_test_model), dtype=float)
    print(f"Predictions dtype: {pred_a.dtype}")
    print(f"Predictions range: [{pred_a.min():.6f}, {pred_a.max():.6f}]")
    print(f"Predictions mean: {pred_a.mean():.6f}")
    print(f"Expected: approximately [0.03, 0.10] (3-10%)")
    print(f"✅ PASS" if 0.01 < pred_a.min() and pred_a.max() < 0.20 else "❌ FAIL - Out of range!")
    
    # 6. Make predictions with Model C (if available)
    print(f"\n=== PREDICTING WITH MODEL C (RISK) ===")
    if model_c is not None:
        if hasattr(model_c, "predict_proba"):
            proba = model_c.predict_proba(x_test_model)
            risk_prob = np.asarray(proba[:, 1], dtype=float)
            print("Using predict_proba (for classifier)")
        else:
            risk_prob = np.asarray(model_c.predict(x_test_model), dtype=float)
            print("Using predict (for regressor)")
        
        # Validate risk probabilities
        if risk_prob.min() < 0 or risk_prob.max() > 1:
            print(f"⚠️  Warning: Risk probabilities [{risk_prob.min():.4f}, {risk_prob.max():.4f}] outside [0, 1]")
        else:
            print(f"✅ Risk probabilities in valid range [0, 1]")
        
        print(f"Risk prob range: [{risk_prob.min():.6f}, {risk_prob.max():.6f}]")
    else:
        risk_prob = np.full(shape=len(pred_a), fill_value=np.nan, dtype=float)
        print("No Model C: risk_prob set to NaN")
    
    # 7. Calculate final scores
    print(f"\n=== CALCULATING FINAL SCORES ===")
    final_score = np.where(np.isnan(risk_prob), pred_a, pred_a * (1 - risk_prob))
    print(f"Final score range: [{final_score.min():.6f}, {final_score.max():.6f}]")
    print(f"Final score mean: {final_score.mean():.6f}")
    
    # 8. Build output dataframe
    print(f"\n=== BUILDING OUTPUT ===")
    output = features[["Datetime", "Symbol"]].copy()
    output["model_version"] = model_a_artifact.get("model_version", "unknown") if isinstance(model_a_artifact, dict) else "unknown"
    output["entry_price"] = pd.to_numeric(features.get("Close", 0), errors="coerce")
    output["pred_a"] = pred_a
    output["risk_prob"] = risk_prob
    output["final_score"] = final_score
    output["feature_version"] = feature_manifest.get("feature_version", "unknown")
    output["source_feature_process_date"] = features.get("process_date", pd.NaT)
    output["process_date"] = pd.Timestamp.utcnow().tz_localize(None)
    
    print(f"Output shape: {output.shape}")
    print(f"\n=== SAMPLE OUTPUT ===")
    print(output[["Symbol", "pred_a", "risk_prob", "final_score"]].head(10))
    
    # 9. Save predictions
    batch_path = stage_dir(config, target_date) / "predictions.parquet"
    print(f"\n=== SAVING PREDICTIONS ===")
    print(f"Saving to: {batch_path}")
    output.to_parquet(batch_path, index=False)
    
    # 10. Create manifest
    manifest = {
        **feature_manifest,
        "prediction_batch": str(batch_path),
        "prediction_rows": int(len(output)),
        "model_version": output["model_version"].iloc[0],
    }
    
    manifest_path = stage_dir(config, target_date) / "inference_manifest.json"
    print(f"Saving manifest to: {manifest_path}")
    write_json(manifest_path, manifest)
    
    print(f"\n✅ Inference complete!")
    return manifest

# Test with real data
if __name__ == "__main__":
    # Use most recent feature batch - LOCAL PATH
    feature_file = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/eod_batch/staging/20260529/features.parquet")
    manifest_file = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/eod_batch/staging/20260529/feature_manifest.json")
    
    # Local model paths
    local_model_a = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/models/model_a.joblib")
    local_model_c = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/models/model_c.joblib")
    
    if manifest_file.exists():
        import json
        with open(manifest_file) as f:
            manifest = json.load(f)
        # Override to use local paths
        manifest["feature_batch"] = str(feature_file)
    else:
        manifest = {
            "as_of_date": "2026-05-29",
            "feature_batch": str(feature_file),
            "feature_version": "price_v1_notebook_ac",
        }
    
    result = run_ml_inference_direct(manifest, model_a_path=local_model_a, model_c_path=local_model_c)
    print(f"\nManifest keys: {list(result.keys())}")
