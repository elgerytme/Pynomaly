#!/usr/bin/env python3

import sys
sys.path.append('/mnt/c/Users/andre/Pynomaly/src/packages/data/anomaly_detection/src')

import numpy as np
from pynomaly_detection import AnomalyDetector, get_default_detector

def test_basic_functionality():
    print("Testing basic anomaly detection functionality...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X[0:5] += 3  # Add some outliers
    
    # Test detector initialization
    detector = AnomalyDetector()
    print("✓ AnomalyDetector initialized successfully")
    
    # Test fit method
    detector.fit(X)
    print("✓ Detector fitted successfully")
    
    # Test predict method
    predictions = detector.predict(X)
    print(f"✓ Predictions generated: {len(predictions)} predictions")
    print(f"✓ Predictions type: {type(predictions)}")
    print(f"✓ Anomalies detected: {sum(predictions) if isinstance(predictions, (list, tuple)) else predictions.sum()}")
    
    # Test detect method (fit + predict in one call)
    detector2 = AnomalyDetector()
    anomalies = detector2.detect(X)
    print(f"✓ Direct detection: {sum(anomalies) if isinstance(anomalies, (list, tuple)) else anomalies.sum()} anomalies detected")
    
    # Test get_default_detector
    default_detector = get_default_detector()
    print("✓ Default detector created successfully")
    
    print("\n🎉 All basic functionality tests passed!")

if __name__ == "__main__":
    test_basic_functionality()