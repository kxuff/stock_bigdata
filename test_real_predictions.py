#!/usr/bin/env python3
"""Test predictions with real feature data from staging"""
import sys
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "airflow" / "plugins"))

from eod_inference.feature_contract import PRICE_FEATURE_COLUMNS
from eod_inference.utils import read_parquet

def test_real_predictions():
    """Test with real feature data"""
    # Load real features from staging
    feature_file = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/eod_batch/staging/20260529/features.parquet")
    if not feature_file.exists():
        print(f"❌ Feature file doesn't exist: {feature_file}")
        return
    
    print("=== Loading Real Feature Data ===")
    features = read_parquet(feature_file)
    print(f"Features shape: {features.shape}")
    print(f"Features columns: {list(features.columns)[:10]}")
    print(f"Features dtypes:\n{features.dtypes}")
    print(f"\nFirst few rows:")
    print(features.head(3))
    
    # Load model
    model_a_path = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/models/model_a.joblib")
    print(f"\n=== Loading Model A ===")
    artifact = joblib.load(model_a_path)
    model = artifact["model"]
    feature_cols = artifact.get("feature_columns", list(PRICE_FEATURE_COLUMNS))
    
    print(f"Model feature columns required: {len(feature_cols)}")
    print(f"First 5: {feature_cols[:5]}")
    
    # Prepare feature matrix
    print(f"\n=== Preparing Feature Matrix ===")
    x_test = features.reindex(columns=feature_cols).astype(float)
    print(f"X_test shape: {x_test.shape}")
    print(f"X_test dtypes: {x_test.dtypes.unique()}")
    print(f"X_test NaN count: {x_test.isna().sum().sum()}")
    print(f"X_test sample:\n{x_test.head(3)}")
    
    # Make predictions
    print(f"\n=== Making Predictions ===")
    predictions = model.predict(x_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions dtype: {predictions.dtype}")
    print(f"Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"Predictions mean: {predictions.mean():.6f}")
    print(f"Predictions std: {predictions.std():.6f}")
    print(f"Predictions (first 10): {predictions[:10]}")
    
    # Compare with stored predictions
    print(f"\n=== Comparing with Stored Predictions ===")
    predictions_file = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/eod_batch/staging/20260529/predictions.parquet")
    if predictions_file.exists():
        stored_preds = read_parquet(predictions_file)
        print(f"Stored predictions shape: {stored_preds.shape}")
        print(f"Stored pred_a range: [{stored_preds['pred_a'].min():.6f}, {stored_preds['pred_a'].max():.6f}]")
        print(f"Stored pred_a mean: {stored_preds['pred_a'].mean():.6f}")
        print(f"Stored pred_a (first 5):")
        print(stored_preds[['Symbol', 'pred_a']].head(5))
        
        # Check if our predictions match
        print(f"\n📊 Do predictions match?")
        print(f"New min: {predictions.min():.6f} vs Stored min: {stored_preds['pred_a'].min():.6f}")
        print(f"New max: {predictions.max():.6f} vs Stored max: {stored_preds['pred_a'].max():.6f}")
        print(f"New mean: {predictions.mean():.6f} vs Stored mean: {stored_preds['pred_a'].mean():.6f}")

if __name__ == "__main__":
    test_real_predictions()
