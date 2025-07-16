"""Comprehensive tests for Detector domain entity."""

from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import pandas as pd
import pytest

from monorepo.domain.entities.dataset import Dataset
from monorepo.domain.entities.detection_result import DetectionResult
from monorepo.domain.entities.detector import Detector
from monorepo.domain.entities.training_result import TrainingResult
from monorepo.domain.value_objects import AnomalyScore, ContaminationRate


class TestDetectorInitialization:
    """Test detector initialization and validation."""

    def test_detector_initialization_with_required_fields(self):
        """Test detector initialization with only required fields."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")

        assert detector.name == "Test Detector"
        assert detector.algorithm_name == "IsolationForest"
        assert isinstance(detector.contamination_rate, ContaminationRate)
        assert detector.contamination_rate.value == 0.1  # Default auto value
        assert detector.parameters == {}
        assert detector.metadata == {}
        assert detector.is_fitted is False
        assert detector.trained_at is None
        assert isinstance(detector.created_at, datetime)
        assert isinstance(detector.id, type(uuid4()))

    def test_detector_initialization_with_all_fields(self):
        """Test detector initialization with all fields specified."""
        contamination_rate = ContaminationRate(0.2)
        parameters = {"n_estimators": 100, "contamination": 0.2}
        metadata = {"version": "1.0", "author": "test"}

        detector = Detector(
            name="Advanced Detector",
            algorithm_name="ExtendedIsolationForest",
            contamination_rate=contamination_rate,
            parameters=parameters,
            metadata=metadata,
        )

        assert detector.name == "Advanced Detector"
        assert detector.algorithm_name == "ExtendedIsolationForest"
        assert detector.contamination_rate == contamination_rate
        assert detector.parameters == parameters
        assert detector.metadata == metadata

    def test_detector_validation_empty_name(self):
        """Test detector validation with empty name."""
        with pytest.raises(ValueError, match="Detector name cannot be empty"):
            Detector(name="", algorithm_name="IsolationForest")

    def test_detector_validation_empty_algorithm_name(self):
        """Test detector validation with empty algorithm name."""
        with pytest.raises(ValueError, match="Algorithm name cannot be empty"):
            Detector(name="Test Detector", algorithm_name="")

    def test_detector_validation_invalid_contamination_rate(self):
        """Test detector validation with invalid contamination rate."""
        with pytest.raises(
            TypeError, match="Contamination rate must be ContaminationRate instance"
        ):
            Detector(
                name="Test Detector",
                algorithm_name="IsolationForest",
                contamination_rate=0.1,  # Should be ContaminationRate instance
            )


class TestDetectorProperties:
    """Test detector properties and metadata."""

    def test_requires_fitting_default(self):
        """Test requires_fitting property with default value."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")
        assert detector.requires_fitting is True

    def test_requires_fitting_custom(self):
        """Test requires_fitting property with custom value."""
        detector = Detector(
            name="Test Detector",
            algorithm_name="StatisticalTest",
            metadata={"requires_fitting": False},
        )
        assert detector.requires_fitting is False

    def test_supports_streaming_default(self):
        """Test supports_streaming property with default value."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")
        assert detector.supports_streaming is False

    def test_supports_streaming_custom(self):
        """Test supports_streaming property with custom value."""
        detector = Detector(
            name="Test Detector",
            algorithm_name="StreamingDetector",
            metadata={"supports_streaming": True},
        )
        assert detector.supports_streaming is True

    def test_supports_multivariate_default(self):
        """Test supports_multivariate property with default value."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")
        assert detector.supports_multivariate is True

    def test_supports_multivariate_custom(self):
        """Test supports_multivariate property with custom value."""
        detector = Detector(
            name="Test Detector",
            algorithm_name="UnivariateDetector",
            metadata={"supports_multivariate": False},
        )
        assert detector.supports_multivariate is False

    def test_time_complexity_property(self):
        """Test time_complexity property."""
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            metadata={"time_complexity": "O(n log n)"},
        )
        assert detector.time_complexity == "O(n log n)"

    def test_time_complexity_none(self):
        """Test time_complexity property when not specified."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")
        assert detector.time_complexity is None

    def test_space_complexity_property(self):
        """Test space_complexity property."""
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            metadata={"space_complexity": "O(n)"},
        )
        assert detector.space_complexity == "O(n)"

    def test_space_complexity_none(self):
        """Test space_complexity property when not specified."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")
        assert detector.space_complexity is None


class TestDetectorMethods:
    """Test detector methods for metadata and parameter management."""

    def test_update_metadata(self):
        """Test updating detector metadata."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")

        detector.update_metadata("version", "1.0")
        detector.update_metadata("author", "test")

        assert detector.metadata["version"] == "1.0"
        assert detector.metadata["author"] == "test"

    def test_update_metadata_overwrite(self):
        """Test overwriting existing metadata."""
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            metadata={"version": "1.0"},
        )

        detector.update_metadata("version", "2.0")
        assert detector.metadata["version"] == "2.0"

    def test_update_parameters(self):
        """Test updating detector parameters."""
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            is_fitted=True,
            trained_at=datetime.utcnow(),
        )

        detector.update_parameters(n_estimators=200, contamination=0.2)

        assert detector.parameters["n_estimators"] == 200
        assert detector.parameters["contamination"] == 0.2
        # Should reset fitted state
        assert detector.is_fitted is False
        assert detector.trained_at is None

    def test_update_parameters_partial(self):
        """Test partial parameter updates."""
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            parameters={"n_estimators": 100, "max_samples": 0.5},
        )

        detector.update_parameters(n_estimators=200)

        assert detector.parameters["n_estimators"] == 200
        assert detector.parameters["max_samples"] == 0.5  # Should remain unchanged

    def test_get_info(self):
        """Test getting comprehensive detector information."""
        contamination_rate = ContaminationRate(0.15)
        parameters = {"n_estimators": 100}
        metadata = {"version": "1.0", "supports_streaming": True}

        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            contamination_rate=contamination_rate,
            parameters=parameters,
            metadata=metadata,
            is_fitted=True,
        )

        info = detector.get_info()

        assert info["name"] == "Test Detector"
        assert info["algorithm"] == "IsolationForest"
        assert info["contamination_rate"] == 0.15
        assert info["is_fitted"] is True
        assert info["parameters"] == parameters
        assert info["metadata"] == metadata
        assert info["requires_fitting"] is True
        assert info["supports_streaming"] is True
        assert info["supports_multivariate"] is True
        assert "id" in info
        assert "created_at" in info
        assert "trained_at" in info

    def test_repr(self):
        """Test string representation of detector."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")

        repr_str = repr(detector)
        assert "Detector(" in repr_str
        assert "name='Test Detector'" in repr_str
        assert "algorithm='IsolationForest'" in repr_str
        assert "is_fitted=False" in repr_str


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

    def test_train_with_score_error_handles_gracefully(self):
        """Test that score errors during training are handled gracefully."""
        detector = Detector(name="Test Detector", algorithm_name="IsolationForest")
        data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        dataset = Dataset(name="Test Data", data=data)

        mock_adapter = Mock()
        mock_adapter.fit.return_value = None
        mock_adapter.score.side_effect = RuntimeError("Scoring failed")

        result = detector.train(dataset, algorithm_adapter=mock_adapter)

        assert isinstance(result, TrainingResult)
        assert result.success is False
        assert "Scoring failed" in result.error_message
        assert detector.is_fitted is False

    def test_train_comprehensive_metrics_calculation(self):
        """Test that training calculates comprehensive metrics correctly."""
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            parameters={"n_estimators": 100, "contamination": 0.1},
        )

        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "feature2": [2, 4, 6, 8, 10, 12],
                "feature3": [1, 1, 1, 1, 1, 1],
            }
        )
        dataset = Dataset(name="Test Data", data=data)

        mock_adapter = Mock()
        mock_adapter.fit.return_value = None
        mock_adapter.score.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        result = detector.train(dataset, algorithm_adapter=mock_adapter)

        assert result.success is True
        assert result.metrics["n_samples"] == 6
        assert result.metrics["n_features"] == 3
        assert result.metrics["algorithm"] == "IsolationForest"
        assert result.metrics["parameters"] == {
            "n_estimators": 100,
            "contamination": 0.1,
        }
        assert result.metrics["mean_score"] == 0.35  # (0.1+0.2+0.3+0.4+0.5+0.6)/6
        assert result.metrics["max_score"] == 0.6
        assert result.metrics["min_score"] == 0.1
        assert "training_time" in result.metrics
        assert result.metrics["training_time"] > 0

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

        # Verify anomaly objects are created correctly
        assert len(result.anomalies) == 1
        anomaly = result.anomalies[0]
        assert anomaly.detector_name == "Test Detector"
        assert anomaly.data_point == {"feature1": 100, "feature2": 200}
        assert "index" in anomaly.metadata
        assert "detector_id" in anomaly.metadata

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

    def test_detect_with_algorithm_error_returns_failed_result(self):
        """Test that algorithm errors during detection return failed result."""
        detector = Detector(
            name="Test Detector", algorithm_name="IsolationForest", is_fitted=True
        )

        data = pd.DataFrame({"feature1": [1, 2, 3, 4]})
        dataset = Dataset(name="Test Data", data=data)

        mock_adapter = Mock()
        mock_adapter.predict.side_effect = RuntimeError("Algorithm failed")

        result = detector.detect(dataset, algorithm_adapter=mock_adapter)

        assert isinstance(result, DetectionResult)
        assert len(result.anomalies) == 0
        assert len(result.scores) == 0
        assert len(result.labels) == 0
        assert "error" in result.metadata
        assert "Algorithm failed" in result.metadata["error"]

    def test_detect_creates_anomaly_objects_correctly(self):
        """Test that detection creates anomaly objects with correct attributes."""
        detector = Detector(
            name="Test Detector", algorithm_name="IsolationForest", is_fitted=True
        )

        data = pd.DataFrame({"feature1": [1, 2, 100], "feature2": [2, 4, 200]})
        dataset = Dataset(name="Test Data", data=data)

        mock_adapter = Mock()
        mock_adapter.predict.return_value = [0, 1, 1]  # Two anomalies
        mock_adapter.score.return_value = [0.1, 0.8, 0.9]

        result = detector.detect(dataset, algorithm_adapter=mock_adapter)

        assert len(result.anomalies) == 2

        # Check first anomaly
        anomaly1 = result.anomalies[0]
        assert anomaly1.detector_name == "Test Detector"
        assert anomaly1.data_point == {"feature1": 2, "feature2": 4}
        assert anomaly1.metadata["index"] == 1
        assert str(detector.id) in anomaly1.metadata["detector_id"]

        # Check second anomaly
        anomaly2 = result.anomalies[1]
        assert anomaly2.detector_name == "Test Detector"
        assert anomaly2.data_point == {"feature1": 100, "feature2": 200}
        assert anomaly2.metadata["index"] == 2

    def test_detect_with_no_anomalies_returns_empty_list(self):
        """Test detection when no anomalies are found."""
        detector = Detector(
            name="Test Detector", algorithm_name="IsolationForest", is_fitted=True
        )

        data = pd.DataFrame({"feature1": [1, 2, 3, 4]})
        dataset = Dataset(name="Test Data", data=data)

        mock_adapter = Mock()
        mock_adapter.predict.return_value = [0, 0, 0, 0]  # No anomalies
        mock_adapter.score.return_value = [0.1, 0.2, 0.15, 0.18]

        result = detector.detect(dataset, algorithm_adapter=mock_adapter)

        assert len(result.anomalies) == 0
        assert len(result.scores) == 4
        assert len(result.labels) == 4
        assert result.n_anomalies == 0
        assert result.anomaly_rate == 0.0

    def test_contamination_rate_used_as_threshold(self):
        """Test that contamination rate is used as detection threshold."""
        contamination_rate = ContaminationRate(0.25)
        detector = Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            contamination_rate=contamination_rate,
            is_fitted=True,
        )

        data = pd.DataFrame({"feature1": [1, 2, 3, 4]})
        dataset = Dataset(name="Test Data", data=data)

        mock_adapter = Mock()
        mock_adapter.predict.return_value = [0, 0, 0, 0]
        mock_adapter.score.return_value = [0.1, 0.2, 0.15, 0.18]

        result = detector.detect(dataset, algorithm_adapter=mock_adapter)

        assert result.threshold == 0.25
        assert result.metadata["contamination_rate"] == 0.25

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

    def test_detect_metadata_includes_all_required_fields(self):
        """Test that detection metadata includes all required fields."""
        detector = Detector(
            name="Test Detector", algorithm_name="IsolationForest", is_fitted=True
        )

        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [2, 4, 6], "feature3": [1, 1, 1]}
        )
        dataset = Dataset(name="Test Data", data=data)

        mock_adapter = Mock()
        mock_adapter.predict.return_value = [0, 0, 0]
        mock_adapter.score.return_value = [0.1, 0.2, 0.15]

        result = detector.detect(dataset, algorithm_adapter=mock_adapter)

        assert "n_samples" in result.metadata
        assert "n_features" in result.metadata
        assert "algorithm" in result.metadata
        assert "detection_time" in result.metadata
        assert "contamination_rate" in result.metadata
        assert result.metadata["n_samples"] == 3
        assert result.metadata["n_features"] == 3
        assert result.metadata["algorithm"] == "IsolationForest"
        assert result.metadata["detection_time"] > 0

    def test_execution_time_tracking(self):
        """Test that execution time is tracked correctly."""
        detector = Detector(
            name="Test Detector", algorithm_name="IsolationForest", is_fitted=True
        )

        data = pd.DataFrame({"feature1": [1, 2, 3]})
        dataset = Dataset(name="Test Data", data=data)

        mock_adapter = Mock()
        mock_adapter.predict.return_value = [0, 0, 0]
        mock_adapter.score.return_value = [0.1, 0.2, 0.15]

        result = detector.detect(dataset, algorithm_adapter=mock_adapter)

        assert result.execution_time_ms > 0
        assert isinstance(result.execution_time_ms, float)
        assert result.metadata["detection_time"] > 0

    def test_anomaly_score_objects_creation(self):
        """Test that AnomalyScore objects are created correctly."""
        detector = Detector(
            name="Test Detector", algorithm_name="IsolationForest", is_fitted=True
        )

        data = pd.DataFrame({"feature1": [1, 2, 3]})
        dataset = Dataset(name="Test Data", data=data)

        mock_adapter = Mock()
        mock_adapter.predict.return_value = [0, 0, 0]
        mock_adapter.score.return_value = [0.1, 0.2, 0.15]

        result = detector.detect(dataset, algorithm_adapter=mock_adapter)

        assert len(result.scores) == 3
        for score in result.scores:
            assert isinstance(score, AnomalyScore)
            assert 0.0 <= score.value <= 1.0

        assert result.scores[0].value == 0.1
        assert result.scores[1].value == 0.2
        assert result.scores[2].value == 0.15
