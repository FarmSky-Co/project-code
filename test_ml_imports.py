#!/usr/bin/env python3
"""
Test ML package imports to verify Conda environment setup.
Run this script in the farmer-credit-scoring environment.
"""

def test_imports():
    """Test all ML package imports."""
    tests = []
    
    # Test Django imports
    try:
        import django
        from django.conf import settings
        from rest_framework import status
        tests.append(f"✅ Django {django.VERSION}")
    except ImportError as e:
        tests.append(f"❌ Django: {e}")
    
    # Test ML core packages
    try:
        import numpy as np
        tests.append(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        tests.append(f"❌ NumPy: {e}")
    
    try:
        import pandas as pd
        tests.append(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        tests.append(f"❌ Pandas: {e}")
    
    try:
        import sklearn
        tests.append(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        tests.append(f"❌ Scikit-learn: {e}")
    
    # Test visualization packages
    try:
        import matplotlib
        tests.append(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        tests.append(f"❌ Matplotlib: {e}")
    
    try:
        import seaborn as sns
        tests.append(f"✅ Seaborn {sns.__version__}")
    except ImportError as e:
        tests.append(f"❌ Seaborn: {e}")
    
    try:
        import plotly
        tests.append(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        tests.append(f"❌ Plotly: {e}")
    
    # Test advanced ML packages
    try:
        import xgboost as xgb
        tests.append(f"✅ XGBoost {xgb.__version__}")
    except ImportError as e:
        tests.append(f"❌ XGBoost: {e}")
    
    try:
        import lightgbm as lgb
        tests.append(f"✅ LightGBM {lgb.__version__}")
    except ImportError as e:
        tests.append(f"❌ LightGBM: {e}")
    
    # Print results
    print("ML Package Import Test Results:")
    print("=" * 40)
    for test in tests:
        print(test)
    print("=" * 40)
    
    # Count successes
    successes = sum(1 for test in tests if test.startswith("✅"))
    total = len(tests)
    print(f"\nSummary: {successes}/{total} packages imported successfully")
    
    if successes == total:
        print("🎉 All packages imported successfully! Environment is ready.")
        return True
    else:
        print("⚠️  Some packages failed to import. Check environment setup.")
        return False

if __name__ == "__main__":
    test_imports()