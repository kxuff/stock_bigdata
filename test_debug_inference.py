#!/usr/bin/env python3
"""Comprehensive test to debug run_ml_inference behavior"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent / "airflow" / "plugins"))

from eod_inference.config import PipelineConfig
from eod_inference.feature_contract import PRICE_FEATURE_COLUMNS
from eod_inference.utils import read_parquet

def debug_inference():
    """Replicate the exact logic from run_ml_inference"""
    
    # Load feature data
    feature_file = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/eod_batch/staging/20260529/features.parquet")
    features = read_parquet(feature_file)
    print(f"Features shape: {features.shape}")
    print(f"Features[['Datetime', 'Symbol']]: \n{features[['Datetime', 'Symbol']].head(3)}")
    
    # Load model  
    model_a_path = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/models/model_a.joblib")
    artifact = joblib.load(model_a_path)
    model_a = artifact
    
    print(f"\nModel A keys: {model_a.keys()}")
    print(f"Model A['model']: {model_a['model']}")
    print(f"Model A['feature_columns'][:5]: {model_a['feature_columns'][:5]}")
    
    # This is what inference.py does:
    expected_features = model_a.get("feature_columns", list(PRICE_FEATURE_COLUMNS))
    print(f"\n=== Reindexing to expected features ===")
    print(f"Expected features count: {len(expected_features)}")
    print(f"Expected features: {expected_features}")
    
    x_test_model = features.reindex(columns=expected_features).astype(float)
    print(f"x_test_model shape: {x_test_model.shape}")
    print(f"x_test_model dtypes unique: {x_test_model.dtypes.unique()}")
    print(f"x_test_model NaN count: {x_test_model.isna().sum().sum()}")
    print(f"x_test_model sample:\n{x_test_model.head(3)}")
    
    # Make prediction
    print(f"\n=== Making Prediction ===")
    pred_a = np.asarray(model_a["model"].predict(x_test_model), dtype=float)
    print(f"pred_a dtype: {pred_a.dtype}")
    print(f"pred_a shape: {pred_a.shape}")
    print(f"pred_a range: [{pred_a.min():.6f}, {pred_a.max():.6f}]")
    print(f"pred_a mean: {pred_a.mean():.6f}")
    print(f"pred_a values (first 10): {pred_a[:10]}")
    
    # Create output
    print(f"\n=== Creating Output ===")
    output = features[["Datetime", "Symbol"]].copy()
    output["pred_a"] = pred_a
    print(f"output shape: {output.shape}")
    print(f"output['pred_a'] values (first 10):\n{output['pred_a'].head(10)}")
    print(f"output['pred_a'] range: [{output['pred_a'].min():.6f}, {output['pred_a'].max():.6f}]")
    
    # Compare with stored predictions
    print(f"\n=== Comparing with Stored ===")
    stored_file = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/eod_batch/staging/20260529/predictions.parquet")
    stored = read_parquet(stored_file)
    print(f"Stored pred_a range: [{stored['pred_a'].min():.6f}, {stored['pred_a'].max():.6f}]")
    print(f"Stored pred_a mean: {stored['pred_a'].mean():.6f}")
    
    # Compare values
    our_vals = output['pred_a'].values
    stored_vals = stored['pred_a'].values
    
    print(f"\nOur predictions (first 5): {our_vals[:5]}")
    print(f"Stored predictions (first 5): {stored_vals[:5]}")
    print(f"Difference: {(our_vals[:5] - stored_vals[:5])}")
    print(f"Max difference: {np.abs(our_vals - stored_vals).max():.10f}")
    print(f"Are they approximately equal? {np.allclose(our_vals, stored_vals, atol=1e-6)}")

if __name__ == "__main__":
    debug_inference()
