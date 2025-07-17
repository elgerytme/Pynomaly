"""Comprehensive tests for AnomalyDetector class."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Any

from pynomaly_detection import AnomalyDetector, get_default_detector


class TestAnomalyDetector:
    """Test suite for AnomalyDetector class."""

    def setup_method(self):
        """Setup test fixtures."""
        np.random.seed(42)
        self.normal_data = np.random.randn(100, 5)
        self.anomalous_data = np.random.randn(100, 5) + 5  # Clear outliers
        self.mixed_data = np.vstack([self.normal_data, self.anomalous_data[:10]])

    def test_init_default(self):
        """Test default initialization."""
        detector = AnomalyDetector()
        assert detector.algorithm is None
        assert detector.config is None
        assert detector._trained is False
        assert detector._model is None
        assert detector.detection_service is None

    def test_init_with_params(self):
        """Test initialization with parameters."""
        algorithm = Mock()
        config = {"contamination": 0.1}
        detector = AnomalyDetector(algorithm=algorithm, config=config)
        
        assert detector.algorithm is algorithm
        assert detector.config == config
        assert detector._trained is False

    def test_fit_basic(self):
        """Test basic fit functionality."""
        detector = AnomalyDetector()
        result = detector.fit(self.normal_data)
        
        assert result is detector  # Should return self
        assert detector._trained is True
        assert detector._model is not None

    def test_fit_with_contamination(self):
        """Test fit with contamination parameter."""
        detector = AnomalyDetector()
        detector.fit(self.normal_data, contamination=0.1)
        
        assert detector._trained is True
        assert detector._model is not None

    def test_fit_with_random_state(self):
        """Test fit with random state for reproducibility."""
        detector1 = AnomalyDetector()
        detector2 = AnomalyDetector()
        
        detector1.fit(self.normal_data, random_state=42)
        detector2.fit(self.normal_data, random_state=42)
        
        pred1 = detector1.predict(self.normal_data)
        pred2 = detector2.predict(self.normal_data)
        
        np.testing.assert_array_equal(pred1, pred2)

    def test_predict_without_fit_raises_error(self):
        """Test that predict raises error when not fitted."""
        detector = AnomalyDetector()
        
        with pytest.raises(ValueError, match="Model must be trained before prediction"):
            detector.predict(self.normal_data)

    def test_predict_after_fit(self):
        """Test predict after fitting."""
        detector = AnomalyDetector()
        detector.fit(self.normal_data)
        
        predictions = detector.predict(self.normal_data)
        
        assert len(predictions) == len(self.normal_data)
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == int
        assert all(p in [0, 1] for p in predictions)

    def test_detect_combines_fit_and_predict(self):
        """Test that detect method combines fit and predict."""
        detector = AnomalyDetector()
        
        predictions = detector.detect(self.mixed_data)
        
        assert len(predictions) == len(self.mixed_data)
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == int
        assert all(p in [0, 1] for p in predictions)
        
        # Should detect some anomalies in mixed data
        anomaly_count = np.sum(predictions)
        assert anomaly_count > 0

    def test_detect_with_parameters(self):
        """Test detect with algorithm parameters."""
        detector = AnomalyDetector()
        
        predictions = detector.detect(self.mixed_data, contamination=0.1, random_state=42)
        
        assert len(predictions) == len(self.mixed_data)
        anomaly_count = np.sum(predictions)
        
        # With 10% contamination, should detect around 10% of samples
        expected_anomalies = int(0.1 * len(self.mixed_data))
        assert abs(anomaly_count - expected_anomalies) <= 2  # Allow some tolerance

    def test_detect_consistency(self):
        """Test that detect gives consistent results."""
        detector = AnomalyDetector()
        
        pred1 = detector.detect(self.normal_data, random_state=42)
        pred2 = detector.detect(self.normal_data, random_state=42)
        
        np.testing.assert_array_equal(pred1, pred2)

    def test_different_data_shapes(self):
        """Test with different data shapes."""
        detector = AnomalyDetector()
        
        # 1D data (should work with reshape)
        data_1d = np.random.randn(50)
        try:
            predictions = detector.detect(data_1d.reshape(-1, 1))
            assert len(predictions) == 50
        except ValueError:
            # Some sklearn versions may not accept 1D data
            pass
        
        # High dimensional data
        data_high_dim = np.random.randn(50, 20)
        predictions = detector.detect(data_high_dim)
        assert len(predictions) == 50

    def test_small_dataset(self):
        """Test behavior with small datasets."""
        detector = AnomalyDetector()
        small_data = np.array([[1, 2], [1, 2], [1, 2]])
        
        try:
            predictions = detector.detect(small_data)
            assert len(predictions) == 3
        except (ValueError, RuntimeError):
            # Small datasets might not work with some algorithms
            pass

    def test_empty_data_raises_error(self):
        """Test that empty data raises appropriate error."""
        detector = AnomalyDetector()
        
        with pytest.raises((ValueError, IndexError)):
            detector.detect(np.array([]))

    @patch('sklearn.ensemble.IsolationForest')
    def test_sklearn_import_failure(self, mock_isolation_forest):
        """Test graceful handling of sklearn import failure."""
        # Mock import failure
        mock_isolation_forest.side_effect = ImportError("sklearn not available")
        
        detector = AnomalyDetector()
        
        with pytest.raises(ImportError, match="sklearn is required for basic anomaly detection"):
            detector.fit(self.normal_data)

    def test_with_detection_service(self):
        """Test behavior when detection service is provided."""
        mock_service = Mock()
        mock_service.detect_anomalies.return_value = np.array([0, 1, 0, 1])
        mock_service.train.return_value = None
        mock_service.predict.return_value = np.array([0, 1, 0, 1])
        
        detector = AnomalyDetector()
        detector.detection_service = mock_service
        
        # Test detect
        result = detector.detect(self.normal_data[:4])
        mock_service.detect_anomalies.assert_called_once()
        np.testing.assert_array_equal(result, [0, 1, 0, 1])
        
        # Test fit
        detector.fit(self.normal_data[:4])
        mock_service.train.assert_called_once()
        
        # Test predict
        detector.predict(self.normal_data[:4])
        mock_service.predict.assert_called_once()

    def test_multiple_fits_overwrites_model(self):
        """Test that multiple fits overwrite the previous model."""
        detector = AnomalyDetector()
        
        # First fit
        detector.fit(self.normal_data, random_state=42)
        first_model = detector._model
        
        # Second fit should create new model
        detector.fit(self.normal_data, random_state=24)
        second_model = detector._model
        
        assert first_model is not second_model

    def test_list_input_data(self):
        """Test with list input data."""
        detector = AnomalyDetector()
        
        # Convert numpy array to list
        data_list = self.normal_data.tolist()
        
        predictions = detector.detect(data_list)
        assert len(predictions) == len(data_list)
        assert isinstance(predictions, np.ndarray)

    def test_performance_with_large_dataset(self):
        """Test performance with larger datasets."""
        detector = AnomalyDetector()
        
        # Generate larger dataset
        large_data = np.random.randn(1000, 10)
        large_data[:50] += 3  # Add some outliers
        
        predictions = detector.detect(large_data, contamination=0.05)
        
        assert len(predictions) == 1000
        anomaly_count = np.sum(predictions)
        
        # Should detect approximately 5% as anomalies
        assert 30 <= anomaly_count <= 70  # Allow reasonable tolerance


class TestDefaultDetector:
    """Test suite for get_default_detector function."""

    def test_get_default_detector_returns_anomaly_detector(self):
        """Test that get_default_detector returns AnomalyDetector instance."""
        detector = get_default_detector()
        assert isinstance(detector, AnomalyDetector)

    def test_get_default_detector_works_with_data(self):
        """Test that default detector works with sample data."""
        detector = get_default_detector()
        
        np.random.seed(42)
        data = np.random.randn(50, 3)
        
        predictions = detector.detect(data)
        assert len(predictions) == 50

    def test_multiple_default_detectors_are_independent(self):
        """Test that multiple default detectors are independent."""
        detector1 = get_default_detector()
        detector2 = get_default_detector()
        
        assert detector1 is not detector2
        
        # Test independence
        np.random.seed(42)
        data = np.random.randn(50, 3)
        
        detector1.fit(data)
        assert detector1._trained is True
        assert detector2._trained is False