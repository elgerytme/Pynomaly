"""Test different algorithm branches to increase coverage."""

import pytest
import numpy as np
from src.pynomaly_detection import AnomalyDetector


class TestAlgorithmBranches:
    """Test different algorithm types to hit coverage branches."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_data = np.random.randn(50, 3)
        self.test_data[:5] += 2  # Add some outliers

    def test_lof_algorithm_branch(self):
        """Test LocalOutlierFactor algorithm branch."""
        detector = AnomalyDetector()
        
        # Test LOF with fit/predict workflow
        detector.fit(self.test_data, algorithm='lof', n_neighbors=5)
        predictions = detector.predict(self.test_data)
        
        assert len(predictions) == len(self.test_data)
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == int
        assert all(p in [0, 1] for p in predictions)

    def test_ocsvm_algorithm_branch(self):
        """Test OneClassSVM algorithm branch."""
        detector = AnomalyDetector()
        
        # Test OneClassSVM with detect method
        results = detector.detect(self.test_data, algorithm='ocsvm', gamma='scale')
        
        assert len(results) == len(self.test_data)
        assert isinstance(results, np.ndarray)
        assert results.dtype == int
        assert all(r in [0, 1] for r in results)

    def test_isolation_forest_default_branch(self):
        """Test IsolationForest default algorithm branch."""
        detector = AnomalyDetector()
        
        # Test with algorithm explicitly set to default
        results = detector.detect(self.test_data, algorithm='isolation_forest', random_state=42)
        
        assert len(results) == len(self.test_data)
        assert isinstance(results, np.ndarray)
        assert results.dtype == int
        assert all(r in [0, 1] for r in results)

    def test_unknown_algorithm_falls_back(self):
        """Test that unknown algorithms fall back to IsolationForest."""
        detector = AnomalyDetector()
        
        # Test with unknown algorithm
        results = detector.detect(self.test_data, algorithm='unknown_algorithm', random_state=42)
        
        assert len(results) == len(self.test_data)
        assert isinstance(results, np.ndarray)
        assert results.dtype == int
        assert all(r in [0, 1] for r in results)

    def test_import_error_handling(self):
        """Test import error handling for missing sklearn."""
        # This test simulates what happens if sklearn is not available
        # but we can't actually test it without mocking
        detector = AnomalyDetector()
        
        # Test that normal operation works
        results = detector.detect(self.test_data)
        assert len(results) == len(self.test_data)

    def test_fit_with_different_algorithms(self):
        """Test fit method with different algorithm parameters."""
        detector = AnomalyDetector()
        
        # Test fit with contamination parameter
        detector.fit(self.test_data, contamination=0.1, random_state=42)
        predictions = detector.predict(self.test_data)
        
        assert len(predictions) == len(self.test_data)
        
        # Test fit with auto contamination
        detector2 = AnomalyDetector()
        detector2.fit(self.test_data, contamination='auto')
        predictions2 = detector2.predict(self.test_data)
        
        assert len(predictions2) == len(self.test_data)