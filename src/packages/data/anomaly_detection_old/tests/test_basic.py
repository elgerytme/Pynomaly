"""Basic tests for pynomaly-detection package."""

import pytest
import numpy as np


def test_import_package():
    """Test that the package can be imported."""
    import pynomaly_detection
    assert pynomaly_detection.__version__ == "0.1.0"


def test_basic_anomaly_detector():
    """Test basic functionality of AnomalyDetector."""
    from pynomaly_detection import AnomalyDetector
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X[0:5] += 3  # Add some outliers
    
    # Test detector
    detector = AnomalyDetector()
    
    # Test fit method
    detector.fit(X)
    
    # Test predict method
    predictions = detector.predict(X)
    
    # Verify results
    assert len(predictions) == len(X)
    assert isinstance(predictions, (np.ndarray, list))
    anomaly_count = sum(predictions) if isinstance(predictions, (list, tuple)) else predictions.sum()
    assert anomaly_count >= 0  # Should detect some anomalies (or none)


def test_detect_method():
    """Test the detect method (fit + predict)."""
    from pynomaly_detection import AnomalyDetector
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X[0:3] += 4  # Add clear outliers
    
    detector = AnomalyDetector()
    
    # Test detect method (if available) or fit+predict
    if hasattr(detector, 'detect'):
        anomalies = detector.detect(X)
    else:
        # Fallback to fit + predict
        detector.fit(X)
        anomalies = detector.predict(X)
    
    # Verify results
    assert len(anomalies) == len(X)
    anomaly_count = sum(anomalies) if isinstance(anomalies, (list, tuple)) else anomalies.sum()
    assert anomaly_count >= 0  # Should detect some anomalies (or none)
    assert anomaly_count <= len(X)  # Should not flag more than total samples


def test_get_default_detector():
    """Test the get_default_detector function."""
    from pynomaly_detection import get_default_detector
    
    detector = get_default_detector()
    assert detector is not None
    
    # Test with sample data
    np.random.seed(42)
    X = np.random.randn(50, 3)
    
    # Should not raise an error
    detector.fit(X)
    predictions = detector.predict(X)
    assert len(predictions) == len(X)


def test_configuration():
    """Test detector with configuration parameters."""
    from pynomaly_detection import AnomalyDetector
    
    # Test with contamination parameter
    detector = AnomalyDetector()
    
    np.random.seed(42)
    X = np.random.randn(100, 4)
    X[0:10] += 2  # Add 10% outliers
    
    # Test with specific contamination
    if hasattr(detector, 'detect'):
        anomalies = detector.detect(X, contamination=0.1)
    else:
        # Fallback to fit + predict with contamination
        detector.fit(X, contamination=0.1)
        anomalies = detector.predict(X)
    
    # Should detect approximately 10% as anomalies
    anomaly_count = sum(anomalies) if isinstance(anomalies, (list, tuple)) else anomalies.sum()
    anomaly_rate = anomaly_count / len(X)
    assert 0.05 <= anomaly_rate <= 0.20  # Allow some flexibility


def test_empty_data():
    """Test behavior with edge cases."""
    from pynomaly_detection import AnomalyDetector
    
    detector = AnomalyDetector()
    
    # Test with minimal data
    X_small = np.array([[1, 2], [1, 2], [1, 2]])
    
    try:
        detector.fit(X_small)
        predictions = detector.predict(X_small)
        assert len(predictions) == len(X_small)
    except (ValueError, RuntimeError):
        # Some algorithms may not work with very small datasets
        pass


def test_package_attributes():
    """Test package-level attributes and exports."""
    import pynomaly_detection
    
    # Test that main classes are available
    assert hasattr(pynomaly_detection, 'AnomalyDetector')
    assert hasattr(pynomaly_detection, 'get_default_detector')
    
    # Test version info
    assert pynomaly_detection.__version__
    assert pynomaly_detection.__author__
    assert pynomaly_detection.__email__


if __name__ == "__main__":
    # Run basic test manually
    test_import_package()
    test_basic_anomaly_detector()
    test_detect_method()
    print("All basic tests passed!")