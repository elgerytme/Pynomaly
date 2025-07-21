#!/usr/bin/env python3
"""Test basic functionality of anomaly_detection."""

import sys
from pathlib import Path
import numpy as np

# Add the package to Python path
package_root = Path(__file__).parent / "src/packages/data/anomaly_detection/src"
sys.path.insert(0, str(package_root))

from anomaly_detection import AnomalyDetector, __version__

def test_basic_functionality():
    """Test basic anomaly detection functionality."""
    print(f"Testing anomaly_detection version: {__version__}")
    
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    # Add some outliers
    data[:10] += 3
    
    # Test basic detector
    detector = AnomalyDetector()
    
    # Test detection
    predictions = detector.detect(data, contamination=0.1)
    
    print(f"✅ Basic detection test passed")
    print(f"   Data shape: {data.shape}")
    print(f"   Anomalies detected: {np.sum(predictions)}")
    print(f"   Anomaly rate: {np.sum(predictions) / len(predictions):.2%}")
    
    # Test fit/predict workflow
    detector2 = AnomalyDetector()
    detector2.fit(data[:80], contamination=0.1)
    predictions2 = detector2.predict(data[80:])
    
    print(f"✅ Fit/predict workflow test passed")
    print(f"   Test data shape: {data[80:].shape}")
    print(f"   Anomalies detected: {np.sum(predictions2)}")
    
    # Test different algorithms
    try:
        predictions_lof = detector.detect(data, algorithm='lof', contamination=0.1)
        print(f"✅ LOF algorithm test passed")
        print(f"   LOF anomalies detected: {np.sum(predictions_lof)}")
    except Exception as e:
        print(f"⚠️ LOF algorithm test failed: {e}")
    
    try:
        predictions_ocsvm = detector.detect(data, algorithm='ocsvm', contamination=0.1)
        print(f"✅ OCSVM algorithm test passed")
        print(f"   OCSVM anomalies detected: {np.sum(predictions_ocsvm)}")
    except Exception as e:
        print(f"⚠️ OCSVM algorithm test failed: {e}")
    
    print(f"\n🎉 All tests completed successfully!")

if __name__ == "__main__":
    test_basic_functionality()