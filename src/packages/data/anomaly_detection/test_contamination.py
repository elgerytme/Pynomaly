#!/usr/bin/env python3
import sys
import os
import importlib

# Clear any existing imports
if 'pynomaly_detection' in sys.modules:
    del sys.modules['pynomaly_detection']

# Set up path
sys.path.insert(0, os.path.abspath('src'))

# Import fresh
import pynomaly_detection
import numpy as np

# Test
print("Testing contamination parameter...")
X = np.random.randn(100, 5)
detector = pynomaly_detection.AnomalyDetector()

try:
    detector.fit(X, contamination=0.1)
    print("✓ Contamination parameter works!")
    predictions = detector.predict(X)
    print(f"✓ Predictions: {len(predictions)} samples")
    anomaly_count = np.sum(predictions) if hasattr(predictions, 'sum') else sum(predictions)
    print(f"✓ Anomalies detected: {anomaly_count}")
except Exception as e:
    print(f"✗ Error: {e}")
    
    # Debug: Check the actual method signature
    import inspect
    sig = inspect.signature(detector.fit)
    print(f"Fit signature: {sig}")
    
    # Try without contamination
    print("Trying without contamination...")
    try:
        detector.fit(X)
        print("✓ Basic fit works")
    except Exception as e2:
        print(f"✗ Basic fit error: {e2}")