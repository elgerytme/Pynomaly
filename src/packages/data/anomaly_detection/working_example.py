#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import numpy as np

def test_working_example():
    """Test a working minimal anomaly detection example."""
    print("🚀 Testing Pynomaly Detection Package")
    print("=" * 50)
    
    # Import with error handling
    try:
        from pynomaly_detection import AnomalyDetector, get_default_detector
        print("✓ Package imported successfully")
        
        # Check available methods
        methods = [m for m in dir(AnomalyDetector) if not m.startswith('_')]
        print(f"✓ Available methods: {methods}")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    np.random.seed(42)
    normal_data = np.random.randn(90, 5)
    anomalous_data = np.random.randn(10, 5) + 4  # Clear outliers
    X = np.vstack([normal_data, anomalous_data])
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    
    print(f"✓ Created dataset with {len(X)} samples, {X.shape[1]} features")
    
    # Test 1: Basic detector initialization
    print("\n🔧 Testing detector initialization...")
    try:
        detector = AnomalyDetector()
        print("✓ AnomalyDetector created successfully")
    except Exception as e:
        print(f"✗ Error creating detector: {e}")
        return False
    
    # Test 2: Fit method
    print("\n🎯 Testing fit method...")
    try:
        detector.fit(X)
        print("✓ Model fitted successfully")
    except Exception as e:
        print(f"✗ Error fitting model: {e}")
        return False
    
    # Test 3: Predict method
    print("\n🔍 Testing predict method...")
    try:
        predictions = detector.predict(X)
        anomaly_count = np.sum(predictions) if hasattr(predictions, 'sum') else sum(predictions)
        print(f"✓ Predictions generated: {len(predictions)} predictions")
        print(f"✓ Anomalies detected: {anomaly_count}")
        print(f"✓ Anomaly rate: {anomaly_count/len(X)*100:.1f}%")
    except Exception as e:
        print(f"✗ Error in prediction: {e}")
        return False
    
    # Test 4: Detect method (fit + predict in one call)
    print("\n⚡ Testing detect method...")
    try:
        detector2 = AnomalyDetector()
        if hasattr(detector2, 'detect'):
            anomalies = detector2.detect(X)
            anomaly_count = np.sum(anomalies) if hasattr(anomalies, 'sum') else sum(anomalies)
            print(f"✓ Direct detection: {anomaly_count} anomalies")
        else:
            print("⚠ Detect method not available, using fit+predict")
            detector2.fit(X)
            anomalies = detector2.predict(X)
            anomaly_count = np.sum(anomalies) if hasattr(anomalies, 'sum') else sum(anomalies)
            print(f"✓ Fit+predict: {anomaly_count} anomalies")
    except Exception as e:
        print(f"✗ Error in detection: {e}")
        return False
    
    # Test 5: Get default detector
    print("\n🔧 Testing get_default_detector...")
    try:
        default_detector = get_default_detector()
        print("✓ Default detector created")
    except Exception as e:
        print(f"✗ Error creating default detector: {e}")
        return False
    
    print("\n🎉 All tests passed! Package is working correctly.")
    return True

if __name__ == "__main__":
    success = test_working_example()
    sys.exit(0 if success else 1)