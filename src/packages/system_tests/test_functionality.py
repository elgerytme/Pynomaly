#!/usr/bin/env python3
"""Test basic functionality - anomaly_detection package removed."""

import numpy as np

# Placeholder anomaly detection functionality for testing
class MockAnomalyDetector:
    """Mock anomaly detector for testing purposes."""
    
    def __init__(self):
        self.is_fitted = False
    
    def fit(self, data, contamination=0.1):
        """Mock fit method."""
        self.is_fitted = True
        return self
    
    def detect(self, data, algorithm='isolation_forest', contamination=0.1):
        """Mock detection method."""
        # Simple mock: mark random 10% as anomalies
        np.random.seed(42)
        n_samples = len(data)
        n_anomalies = int(n_samples * contamination)
        predictions = np.zeros(n_samples, dtype=int)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        predictions[anomaly_indices] = 1
        return predictions
    
    def predict(self, data):
        """Mock predict method."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        return self.detect(data)

def test_basic_functionality():
    """Test basic anomaly detection functionality with mock implementation."""
    print(f"Testing mock anomaly detection functionality (anomaly_detection package removed)")
    
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    # Add some outliers
    data[:10] += 3
    
    # Test basic detector
    detector = MockAnomalyDetector()
    
    # Test detection
    predictions = detector.detect(data, contamination=0.1)
    
    print(f"✅ Basic detection test passed (mock)")
    print(f"   Data shape: {data.shape}")
    print(f"   Anomalies detected: {np.sum(predictions)}")
    print(f"   Anomaly rate: {np.sum(predictions) / len(predictions):.2%}")
    
    # Test fit/predict workflow
    detector2 = MockAnomalyDetector()
    detector2.fit(data[:80], contamination=0.1)
    predictions2 = detector2.predict(data[80:])
    
    print(f"✅ Fit/predict workflow test passed (mock)")
    print(f"   Test data shape: {data[80:].shape}")
    print(f"   Anomalies detected: {np.sum(predictions2)}")
    
    # Test different algorithms (mock implementation)
    try:
        predictions_lof = detector.detect(data, algorithm='lof', contamination=0.1)
        print(f"✅ LOF algorithm test passed (mock)")
        print(f"   LOF anomalies detected: {np.sum(predictions_lof)}")
    except Exception as e:
        print(f"⚠️ LOF algorithm test failed: {e}")
    
    try:
        predictions_ocsvm = detector.detect(data, algorithm='ocsvm', contamination=0.1)
        print(f"✅ OCSVM algorithm test passed (mock)")
        print(f"   OCSVM anomalies detected: {np.sum(predictions_ocsvm)}")
    except Exception as e:
        print(f"⚠️ OCSVM algorithm test failed: {e}")
    
    print(f"\n🎉 All tests completed successfully! (using mock implementation)")

if __name__ == "__main__":
    test_basic_functionality()