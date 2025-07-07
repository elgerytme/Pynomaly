"""Tests for Detector domain entity."""

from datetime import datetime
from unittest.mock import Mock
from uuid import UUID

import pandas as pd
import pytest

from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.training_result import TrainingResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate


class TestDetectorTraining:
    """Test detector training functionality."""

    def test_train_with_valid_dataset_returns_training_result(self):
        """Test that training with valid dataset returns comprehensive training metrics."""
        # Arrange
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.1, "n_estimators": 100},
        )

        # Create test dataset
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 100],  # 100 is outlier
                "feature2": [2, 4, 6, 8, 10, 200],  # 200 is outlier
            }
        )
        dataset = Dataset(name="Test Data", data=data)

        # Mock algorithm adapter
        mock_adapter = Mock()
        mock_adapter.fit.return_value = None
        mock_adapter.score.return_value = [
            0.1,
            0.2,
            0.15,
            0.18,
            0.12,
            0.9,
        ]  # High score for outlier

        # Act
        result = detector.train(dataset, algorithm_adapter=mock_adapter)

        # Assert
        assert isinstance(result, TrainingResult)
        assert result.success is True
        assert detector.is_fitted is True
        assert detector.trained_at is not None
        assert isinstance(detector.trained_at, datetime)
        assert result.metrics is not None
        assert "training_time" in result.metrics
        assert "n_samples" in result.metrics
        assert result.metrics["n_samples"] == 6

        # Verify adapter was called correctly
        mock_adapter.fit.assert_called_once()
        call_args = mock_adapter.fit.call_args[0]
        pd.testing.assert_frame_equal(call_args[0], data)

    def test_train_with_empty_dataset_raises_error(self):
        """Test that training with empty dataset raises appropriate error."""
        # Arrange
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")
        # Create a dataset with some data first, then make it empty to bypass Dataset validation
        data = pd.DataFrame({"feature1": [1, 2, 3]})
        dataset = Dataset(name="Test Data", data=data)
        # Now make the data empty to test our training logic
        dataset.data = pd.DataFrame()
        mock_adapter = Mock()

        # Act & Assert
        with pytest.raises(ValueError, match="Dataset cannot be empty"):
            detector.train(dataset, algorithm_adapter=mock_adapter)

    def test_train_updates_detector_state(self):
        """Test that training properly updates detector state."""
        # Arrange
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")
        data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        dataset = Dataset(name="Test Data", data=data)
        mock_adapter = Mock()
        mock_adapter.fit.return_value = None
        mock_adapter.score.return_value = [0.1, 0.2, 0.15, 0.18, 0.12]

        initial_trained_at = detector.trained_at
        initial_fitted_state = detector.is_fitted

        # Act
        detector.train(dataset, algorithm_adapter=mock_adapter)

        # Assert
        assert detector.is_fitted != initial_fitted_state
        assert detector.trained_at != initial_trained_at
        assert detector.is_fitted is True
        assert isinstance(detector.trained_at, datetime)

    def test_train_with_algorithm_error_handles_gracefully(self):
        """Test that algorithm errors during training are handled gracefully."""
        # Arrange
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")
        data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        dataset = Dataset(name="Test Data", data=data)
        mock_adapter = Mock()
        mock_adapter.fit.side_effect = RuntimeError("Algorithm training failed")

        # Act
        result = detector.train(dataset, algorithm_adapter=mock_adapter)

        # Assert - should return failure result, not raise exception
        assert isinstance(result, TrainingResult)
        assert result.success is False
        assert "Algorithm training failed" in result.error_message
        assert detector.is_fitted is False
        assert detector.trained_at is None


class TestDetectorDetection:
    """Test detector anomaly detection functionality."""

    def test_detect_with_fitted_detector_returns_detection_result(self):
        """Test that detection with fitted detector returns anomaly scores and predictions."""
        # Arrange
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            is_fitted=True,
            trained_at=datetime.utcnow(),
        )

        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 100],  # 100 is outlier
                "feature2": [2, 4, 6, 200],  # 200 is outlier
            }
        )
        dataset = Dataset(name="Test Data", data=data)

        # Mock algorithm adapter
        mock_adapter = Mock()
        mock_adapter.predict.return_value = [0, 0, 0, 1]  # Last sample is anomaly
        mock_adapter.score.return_value = [
            0.1,
            0.15,
            0.12,
            0.95,
        ]  # High score for outlier

        # Act
        result = detector.detect(dataset, algorithm_adapter=mock_adapter)

        # Assert
        assert isinstance(result, DetectionResult)
        assert len(result.labels) == 4
        assert len(result.scores) == 4
        assert list(result.labels) == [0, 0, 0, 1]
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert result.n_anomalies == 1
        assert result.anomaly_rate == 0.25

        # Verify adapter was called correctly
        mock_adapter.predict.assert_called_once()
        mock_adapter.score.assert_called_once()

    def test_detect_with_unfitted_detector_raises_error(self):
        """Test that detection with unfitted detector raises appropriate error."""
        # Arrange
        detector = Detector(
            name="Test Detector", algorithm_name="IsolationForest", is_fitted=False
        )
        data = pd.DataFrame({"feature1": [1, 2, 3, 4]})
        dataset = Dataset(name="Test Data", data=data)
        mock_adapter = Mock()

        # Act & Assert
        with pytest.raises(
            ValueError, match="Detector must be fitted before detection"
        ):
            detector.detect(dataset, algorithm_adapter=mock_adapter)

    def test_detect_with_empty_dataset_handles_gracefully(self):
        """Test that detection with empty dataset returns empty results."""
        # Arrange
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            is_fitted=True,
            trained_at=datetime.utcnow(),
        )
        # Create dataset with data then make it empty to bypass Dataset validation
        data = pd.DataFrame({"feature1": [1, 2, 3]})
        dataset = Dataset(name="Test Data", data=data)
        dataset.data = pd.DataFrame()  # Make it empty for testing
        mock_adapter = Mock()
        mock_adapter.predict.return_value = []
        mock_adapter.score.return_value = []

        # Act
        result = detector.detect(dataset, algorithm_adapter=mock_adapter)

        # Assert
        assert isinstance(result, DetectionResult)
        assert len(result.labels) == 0
        assert len(result.scores) == 0
        assert result.n_anomalies == 0
        assert result.anomaly_rate == 0.0

    def test_detect_provides_confidence_intervals(self):
        """Test that detection provides confidence intervals for scores."""
        # Arrange
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            is_fitted=True,
            trained_at=datetime.utcnow(),
        )

        data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        dataset = Dataset(name="Test Data", data=data)

        mock_adapter = Mock()
        mock_adapter.predict.return_value = [0, 0, 0, 0, 1]
        mock_adapter.score.return_value = [0.1, 0.15, 0.12, 0.18, 0.85]

        # Act
        result = detector.detect(dataset, algorithm_adapter=mock_adapter)

        # Assert
        assert hasattr(result, "confidence_intervals")
        if result.confidence_intervals:
            assert "lower_bound" in result.confidence_intervals
            assert "upper_bound" in result.confidence_intervals


class TestDetectorValidation:
    """Test detector validation and edge cases."""

    def test_detector_requires_algorithm_adapter(self):
        """Test that detector methods require algorithm adapter."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")
        data = pd.DataFrame({"feature1": [1, 2, 3]})
        dataset = Dataset(name="Test Data", data=data)

        # Act & Assert
        with pytest.raises(TypeError):
            detector.train(dataset)  # Missing algorithm_adapter

        with pytest.raises(TypeError):
            detector.detect(dataset)  # Missing algorithm_adapter

    def test_train_validates_dataset_compatibility(self):
        """Test that training validates dataset compatibility."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")

        # Test with non-numeric data that might cause issues
        data = pd.DataFrame({"text_feature": ["a", "b", "c"]})
        dataset = Dataset(name="Text Data", data=data)
        mock_adapter = Mock()
        mock_adapter.fit.side_effect = ValueError("Cannot fit on non-numeric data")

        # Act
        result = detector.train(dataset, algorithm_adapter=mock_adapter)

        # Assert - should return failure result
        assert isinstance(result, TrainingResult)
        assert result.success is False
        assert "Cannot fit on non-numeric data" in result.error_message
