#!/usr/bin/env python3
"""Test complete prediction flow to verify final_score calculation"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent / "airflow" / "plugins"))

from eod_inference.utils import read_parquet

def test_complete_flow():
    """Test complete prediction flow including risk model"""
    
    # Load models
    model_a_path = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/models/model_a.joblib")
    model_c_path = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/models/model_c.joblib")
    
    model_a = joblib.load(model_a_path)
    model_c = joblib.load(model_c_path)
    
    # Load features
    feature_file = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/eod_batch/staging/20260529/features.parquet")
    features = read_parquet(feature_file)
    
    # Get feature columns
    expected_features = model_a["feature_columns"]
    x_test = features.reindex(columns=expected_features).astype(float)
    
    # Model A predictions
    print("=== Model A ===")
    pred_a = np.asarray(model_a["model"].predict(x_test), dtype=float)
    print(f"pred_a range: [{pred_a.min():.6f}, {pred_a.max():.6f}]")
    print(f"pred_a mean: {pred_a.mean():.6f}")
    print(f"pred_a (first 5): {pred_a[:5]}")
    
    # Model C predictions (risk probabilities)
    print(f"\n=== Model C ===")
    model_c_obj = model_c["model"]
    if hasattr(model_c_obj, "predict_proba"):
        proba = model_c_obj.predict_proba(x_test)
        risk_prob = np.asarray(proba[:, 1], dtype=float)
        print("Model C uses predict_proba")
    else:
        risk_prob = np.asarray(model_c_obj.predict(x_test), dtype=float)
        print("Model C uses predict")
    
    print(f"risk_prob range: [{risk_prob.min():.6f}, {risk_prob.max():.6f}]")
    print(f"risk_prob mean: {risk_prob.mean():.6f}")
    print(f"risk_prob (first 5): {risk_prob[:5]}")
    
    # Final score calculation (as in inference.py)
    print(f"\n=== Final Score Calculation ===")
    final_score = np.where(np.isnan(risk_prob), pred_a, pred_a * (1 - risk_prob))
    print(f"final_score range: [{final_score.min():.6f}, {final_score.max():.6f}]")
    print(f"final_score mean: {final_score.mean():.6f}")
    print(f"final_score (first 5): {final_score[:5]}")
    
    # Compare with stored
    print(f"\n=== Comparing with Stored ===")
    stored_file = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/eod_batch/staging/20260529/predictions.parquet")
    stored = read_parquet(stored_file)
    print(f"Stored risk_prob range: [{stored['risk_prob'].min():.6f}, {stored['risk_prob'].max():.6f}]")
    print(f"Stored final_score range: [{stored['final_score'].min():.6f}, {stored['final_score'].max():.6f}]")
    print(f"Match pred_a: {np.allclose(pred_a, stored['pred_a'].values)}")
    print(f"Match risk_prob: {np.allclose(risk_prob, stored['risk_prob'].values)}")
    print(f"Match final_score: {np.allclose(final_score, stored['final_score'].values)}")
    
    # Display a sample
    print(f"\n=== Sample Predictions ===")
    output = pd.DataFrame({
        'Symbol': features['Symbol'].values,
        'pred_a': pred_a,
        'pred_a_%': pred_a * 100,
        'risk_prob': risk_prob,
        'risk_prob_%': risk_prob * 100,
        'final_score': final_score,
        'final_score_%': final_score * 100,
    })
    print(output.head(10))

if __name__ == "__main__":
    test_complete_flow()
