#!/usr/bin/env python3
"""Test script to diagnose model inference issue"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import pickle

# Add airflow plugins to path
sys.path.insert(0, str(Path(__file__).parent / "airflow" / "plugins"))

from eod_inference.config import PipelineConfig
from eod_inference.feature_contract import PRICE_FEATURE_COLUMNS

def test_model_loading():
    """Test loading model artifacts"""
    # Use local paths for testing
    model_a_path = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/models/model_a.joblib")
    model_c_path = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/models/model_c.joblib")
    print(f"Model A path: {model_a_path}")
    print(f"Model C path: {model_c_path}")
    
    # Load Model A
    try:
        print("\n=== Loading Model A ===")
        print(f"Clean path: {model_a_path}")
        print(f"Path exists: {model_a_path.exists()}")
        
        if model_a_path.exists():
            with open(model_a_path, "rb") as f:
                model_a = joblib.load(f)
            print(f"Model A type: {type(model_a)}")
            print(f"Model A keys (if dict): {model_a.keys() if isinstance(model_a, dict) else 'Not a dict'}")
            
            # Check structure
            if isinstance(model_a, dict):
                print(f"Model artifact keys: {list(model_a.keys())}")
                if "model" in model_a:
                    print(f"Inner model type: {type(model_a['model'])}")
                if "feature_columns" in model_a:
                    print(f"Feature columns count: {len(model_a['feature_columns'])}")
                    print(f"First 5 features: {model_a['feature_columns'][:5]}")
        else:
            print("❌ Model A file does not exist!")
    except Exception as e:
        print(f"❌ Error loading Model A: {e}")
    
    # Load Model C
    try:
        print("\n=== Loading Model C ===")
        print(f"Clean path: {model_c_path}")
        print(f"Path exists: {model_c_path.exists()}")
        
        if model_c_path.exists():
            with open(model_c_path, "rb") as f:
                model_c = joblib.load(f)
            print(f"Model C type: {type(model_c)}")
            print(f"Model C keys (if dict): {model_c.keys() if isinstance(model_c, dict) else 'Not a dict'}")
        else:
            print("⚠️  Model C file does not exist")
    except Exception as e:
        print(f"⚠️  Error loading Model C: {e}")

def test_prediction():
    """Test making predictions with sample data"""
    model_a_path = Path("/Volumes/SSD-WDBlue/tohuy/y3s2/stock_bigdata/data/models/model_a.joblib")
    
    if not model_a_path.exists():
        print("❌ Cannot test prediction - model file doesn't exist")
        return
    
    try:
        print("\n=== Testing Prediction ===")
        
        # Load model
        artifact = joblib.load(model_a_path)
        if isinstance(artifact, dict):
            model = artifact["model"]
            features_cols = artifact.get("feature_columns", list(PRICE_FEATURE_COLUMNS))
        else:
            model = artifact
            features_cols = list(PRICE_FEATURE_COLUMNS)
        
        # Create sample feature data
        n_samples = 5
        X_test = pd.DataFrame(
            np.random.randn(n_samples, len(features_cols)),
            columns=features_cols
        )
        
        print(f"Sample X shape: {X_test.shape}")
        print(f"Sample X dtypes: {X_test.dtypes.unique()}")
        
        # Make predictions
        predictions = model.predict(X_test)
        print(f"\nRaw predictions: {predictions}")
        print(f"Predictions dtype: {predictions.dtype}")
        print(f"Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"Predictions mean: {predictions.mean():.6f}")
        
        # Check if range matches expected [0.03, 0.10]
        print(f"\n📊 Analysis:")
        print(f"Min: {predictions.min():.6f}")
        print(f"Max: {predictions.max():.6f}")
        print(f"Mean: {predictions.mean():.6f}")
        print(f"Std: {predictions.std():.6f}")
        
        # If in expected range as percentage
        if predictions.min() < 0.01 and predictions.max() < 0.5:
            print("✅ Predictions appear to be in decimal format (0.0X)")
            print(f"As percentages: {(predictions * 100).min():.2f}% - {(predictions * 100).max():.2f}%")
        elif predictions.min() > 3 and predictions.max() < 20:
            print("⚠️  Predictions appear to be in percentage format already (3-10)")
            print(f"As decimals: {(predictions / 100).min():.6f} - {(predictions / 100).max():.6f}")
        else:
            print("⚠️  Predictions are NOT in expected range [3,10]%")
            
    except Exception as e:
        print(f"❌ Error in prediction test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()
    test_prediction()
