"""
Comprehensive unit tests for anomaly detection functionality.

This module demonstrates:
- Standard unit tests with mocking
- Property-based testing with Hypothesis
- Parametrized testing
- Test fixtures
- Coverage optimization
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynomaly.application.services.anomaly_detection_service import (
    AnomalyDetectionService,
)
from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.services.advanced_detection_service import AdvancedDetectionService
from pynomaly.domain.value_objects import AnomalyScore


class TestAnomalyDetectionService:
    """Unit tests for AnomalyDetectionService."""

    def test_init_creates_service(self):
        """Test service initialization."""
        service = AnomalyDetectionService()
        assert service is not None
        assert hasattr(service, "detect_anomalies")

    def test_detect_anomalies_with_valid_data(self, sample_dataset, sample_detector):
        """Test anomaly detection with valid data."""
        service = AnomalyDetectionService()

        # Mock the underlying detection
        with patch.object(service, "_run_detection") as mock_detection:
            mock_detection.return_value = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=sample_dataset.id,
                scores=[AnomalyScore(0.1), AnomalyScore(0.9)],
                metadata={"test": True},
            )

            result = service.detect_anomalies(sample_dataset, sample_detector)

            assert result is not None
            assert len(result.scores) == 2
            assert result.detector_id == sample_detector.id
            mock_detection.assert_called_once()

    def test_detect_anomalies_with_empty_dataset(self, sample_detector):
        """Test anomaly detection with empty dataset."""
        empty_dataset = Dataset(
            name="Empty Dataset",
            data=pd.DataFrame(),
            description="Empty dataset for testing",
        )

        service = AnomalyDetectionService()

        with pytest.raises(ValueError, match="Dataset cannot be empty"):
            service.detect_anomalies(empty_dataset, sample_detector)

    def test_detect_anomalies_with_invalid_detector(self, sample_dataset):
        """Test anomaly detection with invalid detector."""
        invalid_detector = Detector(
            algorithm_name="NonExistentAlgorithm", parameters={}
        )

        service = AnomalyDetectionService()

        with pytest.raises(ValueError, match="Unsupported algorithm"):
            service.detect_anomalies(sample_dataset, invalid_detector)

    @pytest.mark.parametrize(
        "algorithm,expected_params",
        [
            ("IsolationForest", {"contamination": 0.1}),
            ("LocalOutlierFactor", {"n_neighbors": 20}),
            ("OneClassSVM", {"nu": 0.05}),
        ],
    )
    def test_detect_anomalies_with_different_algorithms(
        self, sample_dataset, algorithm, expected_params
    ):
        """Test anomaly detection with different algorithms."""
        detector = Detector(algorithm_name=algorithm, parameters=expected_params)

        service = AnomalyDetectionService()

        with patch.object(service, "_run_detection") as mock_detection:
            mock_detection.return_value = DetectionResult(
                detector_id=detector.id,
                dataset_id=sample_dataset.id,
                scores=[AnomalyScore(0.5)],
                metadata={"algorithm": algorithm},
            )

            result = service.detect_anomalies(sample_dataset, detector)

            assert result.metadata["algorithm"] == algorithm
            mock_detection.assert_called_once()

    @pytest.mark.slow
    def test_detect_anomalies_performance(self, large_dataset, sample_detector):
        """Test anomaly detection performance with large dataset."""
        service = AnomalyDetectionService()

        import time

        start_time = time.time()

        # Mock for performance test
        with patch.object(service, "_run_detection") as mock_detection:
            mock_detection.return_value = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=large_dataset.id,
                scores=[AnomalyScore(0.5)] * len(large_dataset.data),
                metadata={"performance_test": True},
            )

            result = service.detect_anomalies(large_dataset, sample_detector)

            end_time = time.time()
            execution_time = end_time - start_time

            assert execution_time < 5.0  # Should complete within 5 seconds
            assert len(result.scores) == len(large_dataset.data)


class TestAdvancedDetectionServicePropertyBased:
    """Property-based tests for DetectionService using Hypothesis."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=10, max_value=1000),  # rows
                st.integers(min_value=2, max_value=50),  # columns
            ),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False),
        ),
        contamination=st.floats(min_value=0.01, max_value=0.5),
    )
    @settings(max_examples=50, deadline=10000)
    def test_detection_service_properties(self, data, contamination):
        """Property-based test for detection service invariants."""
        assume(not np.any(np.isnan(data)))
        assume(not np.any(np.isinf(data)))

        # Create dataset
        df = pd.DataFrame(data)
        dataset = Dataset(
            name="Property Test Dataset",
            data=df,
            description="Generated dataset for property testing",
        )

        # Create detector
        detector = Detector(
            algorithm_name="IsolationForest",
            parameters={"contamination": contamination, "random_state": 42},
        )

        service = AdvancedDetectionService()

        # Mock the actual detection
        with patch.object(service, "_get_adapter") as mock_adapter:
            mock_model = Mock()
            mock_model.decision_function.return_value = np.random.random(len(data))
            mock_adapter.return_value.create_model.return_value = mock_model
            mock_adapter.return_value.fit.return_value = None
            mock_adapter.return_value.predict.return_value = np.random.choice(
                [0, 1], len(data)
            )

            result = service.detect_anomalies(dataset, detector)

            # Property 1: Result should have same number of scores as input samples
            assert len(result.scores) == len(data)

            # Property 2: All scores should be valid numbers
            for score in result.scores:
                assert isinstance(score, AnomalyScore)
                assert not np.isnan(score.value)
                assert not np.isinf(score.value)

            # Property 3: Result should have correct detector and dataset IDs
            assert result.detector_id == detector.id
            assert result.dataset_id == dataset.id

    @given(
        n_samples=st.integers(min_value=5, max_value=100),
        n_features=st.integers(min_value=2, max_value=20),
        noise_level=st.floats(min_value=0.0, max_value=2.0),
    )
    @example(n_samples=10, n_features=5, noise_level=0.1)  # Known good example
    def test_detection_service_with_synthetic_data(
        self, n_samples, n_features, noise_level
    ):
        """Test detection service with synthetic data generation."""
        # Generate synthetic data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (n_samples, n_features))
        noise = np.random.normal(0, noise_level, (n_samples, n_features))
        data = normal_data + noise

        df = pd.DataFrame(data)
        dataset = Dataset(
            name="Synthetic Dataset",
            data=df,
            description="Synthetic dataset for testing",
        )

        detector = Detector(
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.1, "random_state": 42},
        )

        service = AdvancedDetectionService()

        # Mock the detection
        with patch.object(service, "_get_adapter") as mock_adapter:
            mock_model = Mock()
            mock_model.decision_function.return_value = np.random.random(n_samples)
            mock_adapter.return_value.create_model.return_value = mock_model
            mock_adapter.return_value.fit.return_value = None
            mock_adapter.return_value.predict.return_value = np.random.choice(
                [0, 1], n_samples
            )

            result = service.detect_anomalies(dataset, detector)

            # Properties should hold regardless of data characteristics
            assert len(result.scores) == n_samples
            assert all(isinstance(score, AnomalyScore) for score in result.scores)
            assert result.detector_id == detector.id

    @given(
        algorithm=st.sampled_from(
            ["IsolationForest", "LocalOutlierFactor", "OneClassSVM", "EllipticEnvelope"]
        )
    )
    def test_detection_service_algorithm_invariants(self, algorithm):
        """Test that detection service maintains invariants across different algorithms."""
        # Create standard test data
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 5))
        df = pd.DataFrame(data)

        dataset = Dataset(
            name="Algorithm Test Dataset",
            data=df,
            description="Dataset for algorithm testing",
        )

        detector = Detector(algorithm_name=algorithm, parameters={"random_state": 42})

        service = AdvancedDetectionService()

        # Mock the detection
        with patch.object(service, "_get_adapter") as mock_adapter:
            mock_model = Mock()
            mock_model.decision_function.return_value = np.random.random(100)
            mock_adapter.return_value.create_model.return_value = mock_model
            mock_adapter.return_value.fit.return_value = None
            mock_adapter.return_value.predict.return_value = np.random.choice(
                [0, 1], 100
            )

            result = service.detect_anomalies(dataset, detector)

            # Algorithm-independent invariants
            assert len(result.scores) == 100
            assert result.detector_id == detector.id
            assert result.dataset_id == dataset.id
            assert all(isinstance(score, AnomalyScore) for score in result.scores)


class TestAnomalyScoreValueObject:
    """Unit tests for AnomalyScore value object."""

    def test_anomaly_score_creation(self):
        """Test AnomalyScore creation with valid values."""
        score = AnomalyScore(0.5)
        assert score.value == 0.5
        assert isinstance(score.value, float)

    def test_anomaly_score_validation(self):
        """Test AnomalyScore validation."""
        # Valid scores
        AnomalyScore(0.0)
        AnomalyScore(1.0)
        AnomalyScore(0.5)

        # Invalid scores should raise ValueError
        with pytest.raises(ValueError):
            AnomalyScore(float("nan"))

        with pytest.raises(ValueError):
            AnomalyScore(float("inf"))

    @given(
        score_value=st.floats(
            min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
        )
    )
    def test_anomaly_score_properties(self, score_value):
        """Property-based test for AnomalyScore."""
        score = AnomalyScore(score_value)

        # Property 1: Value should be preserved
        assert score.value == score_value

        # Property 2: Value should be a float
        assert isinstance(score.value, float)

        # Property 3: Score should be comparable
        other_score = AnomalyScore(score_value)
        assert score.value == other_score.value

    def test_anomaly_score_comparison(self):
        """Test AnomalyScore comparison operations."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)
        score3 = AnomalyScore(0.3)

        assert score1.value < score2.value
        assert score2.value > score1.value
        assert score1.value == score3.value

    def test_anomaly_score_edge_cases(self):
        """Test AnomalyScore edge cases."""
        # Test with very small values
        small_score = AnomalyScore(1e-10)
        assert small_score.value == 1e-10

        # Test with very large values
        large_score = AnomalyScore(1e10)
        assert large_score.value == 1e10

        # Test with negative values
        negative_score = AnomalyScore(-0.5)
        assert negative_score.value == -0.5


class TestDatasetEntityUnit:
    """Unit tests for Dataset entity."""

    def test_dataset_creation_with_minimal_data(self):
        """Test Dataset creation with minimal required data."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        dataset = Dataset(name="Test Dataset", data=df, description="Test dataset")

        assert dataset.name == "Test Dataset"
        assert dataset.description == "Test dataset"
        assert len(dataset.data) == 3
        assert list(dataset.data.columns) == ["feature1", "feature2"]

    def test_dataset_with_metadata(self):
        """Test Dataset with metadata."""
        df = pd.DataFrame({"feature1": [1, 2, 3]})
        metadata = {"source": "test", "created_by": "unit_test"}

        dataset = Dataset(
            name="Test Dataset", data=df, description="Test dataset", metadata=metadata
        )

        assert dataset.metadata == metadata
        assert dataset.metadata["source"] == "test"

    @pytest.mark.parametrize("data_size", [1, 10, 100, 1000])
    def test_dataset_with_different_sizes(self, data_size):
        """Test Dataset with different data sizes."""
        df = pd.DataFrame(
            {
                "feature1": list(range(data_size)),
                "feature2": list(range(data_size, 2 * data_size)),
            }
        )

        dataset = Dataset(
            name=f"Test Dataset {data_size}",
            data=df,
            description=f"Test dataset with {data_size} samples",
        )

        assert len(dataset.data) == data_size
        assert dataset.name == f"Test Dataset {data_size}"

    def test_dataset_immutability(self):
        """Test that Dataset maintains data integrity."""
        original_df = pd.DataFrame({"feature1": [1, 2, 3]})
        dataset = Dataset(
            name="Test Dataset", data=original_df, description="Test dataset"
        )

        # Modifying original DataFrame should not affect dataset
        original_df.loc[0, "feature1"] = 999

        assert dataset.data.iloc[0]["feature1"] == 1  # Should remain unchanged


@pytest.mark.unit
class TestDetectorEntityUnit:
    """Unit tests for Detector entity."""

    def test_detector_creation(self):
        """Test Detector creation."""
        detector = Detector(
            algorithm_name="IsolationForest", parameters={"contamination": 0.1}
        )

        assert detector.algorithm_name == "IsolationForest"
        assert detector.parameters["contamination"] == 0.1
        assert detector.is_fitted is False

    def test_detector_with_metadata(self):
        """Test Detector with metadata."""
        metadata = {"version": "1.0", "author": "test"}
        detector = Detector(
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.1},
            metadata=metadata,
        )

        assert detector.metadata == metadata

    def test_detector_fitting_state(self):
        """Test Detector fitting state management."""
        detector = Detector(
            algorithm_name="IsolationForest", parameters={"contamination": 0.1}
        )

        assert detector.is_fitted is False

        # Simulate fitting
        detector.is_fitted = True
        assert detector.is_fitted is True

    @pytest.mark.parametrize(
        "algorithm,params",
        [
            ("IsolationForest", {"contamination": 0.1}),
            ("LocalOutlierFactor", {"n_neighbors": 20}),
            ("OneClassSVM", {"nu": 0.05, "gamma": "scale"}),
        ],
    )
    def test_detector_with_different_algorithms(self, algorithm, params):
        """Test Detector with different algorithms and parameters."""
        detector = Detector(algorithm_name=algorithm, parameters=params)

        assert detector.algorithm_name == algorithm
        assert detector.parameters == params


# Property-based test for entire detection pipeline
@pytest.mark.property
class TestDetectionPipelineProperties:
    """Property-based tests for the entire detection pipeline."""

    @given(
        n_samples=st.integers(min_value=10, max_value=200),
        n_features=st.integers(min_value=2, max_value=10),
        contamination=st.floats(min_value=0.01, max_value=0.3),
    )
    @settings(max_examples=20, deadline=15000)
    def test_end_to_end_detection_properties(
        self, n_samples, n_features, contamination
    ):
        """Test end-to-end detection pipeline properties."""
        # Generate synthetic data
        np.random.seed(42)
        data = np.random.normal(0, 1, (n_samples, n_features))
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])

        dataset = Dataset(
            name="Property Test Dataset",
            data=df,
            description="Generated for property testing",
        )

        detector = Detector(
            algorithm_name="IsolationForest",
            parameters={"contamination": contamination, "random_state": 42},
        )

        service = AnomalyDetectionService()

        # Mock the detection
        with patch.object(service, "_run_detection") as mock_detection:
            expected_anomalies = int(n_samples * contamination)
            mock_scores = [AnomalyScore(np.random.random()) for _ in range(n_samples)]

            mock_detection.return_value = DetectionResult(
                detector_id=detector.id,
                dataset_id=dataset.id,
                scores=mock_scores,
                metadata={"n_samples": n_samples, "n_features": n_features},
            )

            result = service.detect_anomalies(dataset, detector)

            # Pipeline properties
            assert len(result.scores) == n_samples
            assert result.detector_id == detector.id
            assert result.dataset_id == dataset.id
            assert all(isinstance(score, AnomalyScore) for score in result.scores)
            assert result.metadata["n_samples"] == n_samples
            assert result.metadata["n_features"] == n_features
