"""
Comprehensive tests for detection service.
Tests core detection orchestration, ensemble coordination, and result processing.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.application.services import DetectionService
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.exceptions import DetectorError
from pynomaly.domain.value_objects import AnomalyScore


class TestDetectionService:
    """Test suite for DetectionService application service."""

    @pytest.fixture
    def mock_detector_repository(self):
        """Mock detector repository."""
        mock_repo = AsyncMock()
        mock_repo.find_by_id.return_value = None
        mock_repo.save.return_value = None
        mock_repo.delete.return_value = None
        return mock_repo

    @pytest.fixture
    def mock_dataset_repository(self):
        """Mock dataset repository."""
        mock_repo = AsyncMock()
        mock_repo.find_by_id.return_value = None
        mock_repo.save.return_value = None
        return mock_repo

    @pytest.fixture
    def mock_result_repository(self):
        """Mock detection result repository."""
        mock_repo = AsyncMock()
        mock_repo.save.return_value = None
        mock_repo.find_by_detector_id.return_value = []
        return mock_repo

    @pytest.fixture
    def mock_algorithm_adapter(self):
        """Mock algorithm adapter."""
        adapter = Mock()
        adapter.fit.return_value = None
        adapter.predict.return_value = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
        adapter.predict_proba.return_value = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
        adapter.is_fitted = True
        return adapter

    @pytest.fixture
    def mock_algorithm_registry(self, mock_algorithm_adapter):
        """Mock algorithm registry."""
        registry = Mock()
        registry.get_adapter.return_value = mock_algorithm_adapter
        registry.list_algorithms.return_value = [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
        ]
        return registry

    @pytest.fixture
    def sample_detector(self):
        """Create sample detector."""
        return Detector(
            id=uuid4(),
            name="test-detector",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100, "contamination": 0.1},
            is_fitted=True,
        )

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        return Dataset(
            id=uuid4(),
            name="test-dataset",
            file_path="/tmp/test.csv",
            features=["feature1", "feature2", "feature3"],
            feature_types={"feature1": "numeric", "feature2": "numeric", "feature3": "categorical"},
            target_column=None,
            data_shape=(100, 3),
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample detection data."""
        return np.random.randn(100, 3)

    @pytest.fixture
    def detection_service(
        self,
        mock_detector_repository,
        mock_dataset_repository,
        mock_result_repository,
        mock_algorithm_registry,
    ):
        """Create detection service with mocked dependencies."""
        return DetectionService(
            detector_repository=mock_detector_repository,
            dataset_repository=mock_dataset_repository,
            result_repository=mock_result_repository,
            algorithm_registry=mock_algorithm_registry,
        )

    @pytest.mark.asyncio
    async def test_detect_anomalies_basic(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
        mock_algorithm_adapter,
    ):
        """Test basic anomaly detection."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        result = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=sample_data,
        )

        # Verify
        assert result is not None
        assert isinstance(result, DetectionResult)
        assert result.detector_id == detector_id
        assert len(result.anomaly_scores) == len(sample_data)
        assert all(isinstance(score, AnomalyScore) for score in result.anomaly_scores)

        # Verify algorithm adapter was called
        mock_algorithm_adapter.predict.assert_called_once()
        mock_detector_repository.find_by_id.assert_called_once_with(detector_id)

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_threshold(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
        mock_algorithm_adapter,
    ):
        """Test anomaly detection with custom threshold."""
        # Setup
        detector_id = sample_detector.id
        threshold = 0.5
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        result = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=sample_data,
            threshold=threshold,
        )

        # Verify
        assert result is not None
        assert result.threshold == threshold
        assert len(result.anomalies) >= 0  # Should have some anomalies based on threshold

        # Verify anomalies are properly classified
        for anomaly in result.anomalies:
            assert isinstance(anomaly, Anomaly)
            assert anomaly.score.value >= threshold

    @pytest.mark.asyncio
    async def test_detect_anomalies_detector_not_found(
        self,
        detection_service,
        sample_data,
        mock_detector_repository,
    ):
        """Test detection with non-existent detector."""
        # Setup
        detector_id = uuid4()
        mock_detector_repository.find_by_id.return_value = None

        # Execute & Verify
        with pytest.raises(DetectorError, match="Detector not found"):
            await detection_service.detect_anomalies(
                detector_id=detector_id,
                data=sample_data,
            )

    @pytest.mark.asyncio
    async def test_detect_anomalies_unfitted_detector(
        self,
        detection_service,
        sample_data,
        mock_detector_repository,
        mock_algorithm_adapter,
    ):
        """Test detection with unfitted detector."""
        # Setup
        unfitted_detector = Detector(
            id=uuid4(),
            name="unfitted-detector",
            algorithm_name="IsolationForest",
            hyperparameters={},
            is_fitted=False,
        )
        mock_detector_repository.find_by_id.return_value = unfitted_detector
        mock_algorithm_adapter.is_fitted = False

        # Execute & Verify
        with pytest.raises(DetectorError, match="Detector must be fitted"):
            await detection_service.detect_anomalies(
                detector_id=unfitted_detector.id,
                data=sample_data,
            )

    @pytest.mark.asyncio
    async def test_detect_anomalies_invalid_data(
        self,
        detection_service,
        sample_detector,
        mock_detector_repository,
    ):
        """Test detection with invalid data."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Test with None data
        with pytest.raises(ValueError, match="Data cannot be None"):
            await detection_service.detect_anomalies(
                detector_id=detector_id,
                data=None,
            )

        # Test with empty data
        with pytest.raises(ValueError, match="Data cannot be empty"):
            await detection_service.detect_anomalies(
                detector_id=detector_id,
                data=np.array([]),
            )

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_feature_names(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
    ):
        """Test detection with feature names."""
        # Setup
        detector_id = sample_detector.id
        feature_names = ["feature1", "feature2", "feature3"]
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        result = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=sample_data,
            feature_names=feature_names,
        )

        # Verify
        assert result is not None
        assert result.feature_names == feature_names

        # Verify anomalies have feature information
        for anomaly in result.anomalies:
            assert anomaly.feature_names == feature_names

    @pytest.mark.asyncio
    async def test_detect_anomalies_batch_processing(
        self,
        detection_service,
        sample_detector,
        mock_detector_repository,
        mock_algorithm_adapter,
    ):
        """Test batch processing of large datasets."""
        # Setup
        detector_id = sample_detector.id
        large_data = np.random.randn(10000, 3)  # Large dataset
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        result = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=large_data,
            batch_size=1000,
        )

        # Verify
        assert result is not None
        assert len(result.anomaly_scores) == len(large_data)
        assert result.processing_info["batch_size"] == 1000
        assert result.processing_info["total_batches"] == 10

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_confidence_intervals(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
        mock_algorithm_adapter,
    ):
        """Test detection with confidence intervals."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Mock algorithm adapter to return confidence intervals
        mock_algorithm_adapter.predict_with_confidence.return_value = (
            np.array([0.1, 0.8, 0.3, 0.9, 0.2]),
            np.array([0.05, 0.75, 0.25, 0.85, 0.15]),
            np.array([0.15, 0.85, 0.35, 0.95, 0.25]),
        )

        # Execute
        result = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=sample_data,
            include_confidence=True,
        )

        # Verify
        assert result is not None
        assert result.has_confidence_intervals

        # Verify confidence intervals for anomaly scores
        for score in result.anomaly_scores:
            assert score.confidence_lower is not None
            assert score.confidence_upper is not None
            assert score.confidence_lower <= score.value <= score.confidence_upper

    @pytest.mark.asyncio
    async def test_detect_streaming_anomalies(
        self,
        detection_service,
        sample_detector,
        mock_detector_repository,
        mock_algorithm_adapter,
    ):
        """Test streaming anomaly detection."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Create streaming data generator
        async def data_generator():
            for i in range(10):
                yield np.random.randn(1, 3)

        # Execute
        results = []
        async for result in detection_service.detect_streaming_anomalies(
            detector_id=detector_id,
            data_stream=data_generator(),
        ):
            results.append(result)

        # Verify
        assert len(results) == 10
        for result in results:
            assert isinstance(result, DetectionResult)
            assert result.detector_id == detector_id
            assert len(result.anomaly_scores) == 1

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_preprocessing(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
    ):
        """Test detection with preprocessing pipeline."""
        # Setup
        detector_id = sample_detector.id
        preprocessing_config = {
            "standardize": True,
            "remove_outliers": True,
            "handle_missing": "mean",
        }
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        result = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=sample_data,
            preprocessing_config=preprocessing_config,
        )

        # Verify
        assert result is not None
        assert result.preprocessing_applied is True
        assert result.preprocessing_config == preprocessing_config

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_explanation(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
        mock_algorithm_adapter,
    ):
        """Test detection with anomaly explanation."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Mock explanation functionality
        mock_algorithm_adapter.explain_anomaly.return_value = {
            "feature_importance": [0.5, 0.3, 0.2],
            "feature_contributions": [0.1, 0.2, 0.05],
            "explanation_text": "High values in feature1 and feature2",
        }

        # Execute
        result = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=sample_data,
            include_explanation=True,
        )

        # Verify
        assert result is not None
        assert result.has_explanations

        # Verify explanations for anomalies
        for anomaly in result.anomalies:
            if anomaly.explanation:
                assert "feature_importance" in anomaly.explanation
                assert "feature_contributions" in anomaly.explanation
                assert "explanation_text" in anomaly.explanation

    @pytest.mark.asyncio
    async def test_detect_anomalies_result_persistence(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
        mock_result_repository,
    ):
        """Test detection result persistence."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        result = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=sample_data,
            save_results=True,
        )

        # Verify
        assert result is not None
        mock_result_repository.save.assert_called_once()

        # Verify saved result
        saved_result = mock_result_repository.save.call_args[0][0]
        assert isinstance(saved_result, DetectionResult)
        assert saved_result.detector_id == detector_id

    @pytest.mark.asyncio
    async def test_detect_anomalies_performance_monitoring(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
    ):
        """Test performance monitoring during detection."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        result = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=sample_data,
            monitor_performance=True,
        )

        # Verify
        assert result is not None
        assert result.performance_metrics is not None
        assert "execution_time" in result.performance_metrics
        assert "memory_usage" in result.performance_metrics
        assert "throughput" in result.performance_metrics

    @pytest.mark.asyncio
    async def test_detect_anomalies_concurrent_requests(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
    ):
        """Test concurrent anomaly detection requests."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute multiple concurrent requests
        tasks = []
        for i in range(5):
            task = detection_service.detect_anomalies(
                detector_id=detector_id,
                data=sample_data,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Verify
        assert len(results) == 5
        for result in results:
            assert isinstance(result, DetectionResult)
            assert result.detector_id == detector_id

    @pytest.mark.asyncio
    async def test_detect_anomalies_error_handling(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
        mock_algorithm_adapter,
    ):
        """Test error handling during detection."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Mock algorithm adapter to raise error
        mock_algorithm_adapter.predict.side_effect = RuntimeError("Algorithm failed")

        # Execute & Verify
        with pytest.raises(DetectorError, match="Detection failed"):
            await detection_service.detect_anomalies(
                detector_id=detector_id,
                data=sample_data,
            )

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_custom_scoring(
        self,
        detection_service,
        sample_detector,
        sample_data,
        mock_detector_repository,
    ):
        """Test detection with custom scoring function."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Define custom scoring function
        def custom_scoring(scores):
            return np.exp(scores) / np.sum(np.exp(scores))

        # Execute
        result = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=sample_data,
            custom_scoring=custom_scoring,
        )

        # Verify
        assert result is not None
        assert result.custom_scoring_applied is True

        # Verify scores are transformed
        for score in result.anomaly_scores:
            assert 0 <= score.value <= 1  # Softmax normalization

    @pytest.mark.asyncio
    async def test_detect_anomalies_integration_workflow(
        self,
        detection_service,
        sample_detector,
        sample_dataset,
        mock_detector_repository,
        mock_dataset_repository,
    ):
        """Test complete integration workflow."""
        # Setup
        detector_id = sample_detector.id
        dataset_id = sample_dataset.id
        mock_detector_repository.find_by_id.return_value = sample_detector
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute complete workflow
        result = await detection_service.detect_anomalies_from_dataset(
            detector_id=detector_id,
            dataset_id=dataset_id,
            include_confidence=True,
            include_explanation=True,
            save_results=True,
            monitor_performance=True,
        )

        # Verify
        assert result is not None
        assert isinstance(result, DetectionResult)
        assert result.detector_id == detector_id
        assert result.dataset_id == dataset_id
        assert result.has_confidence_intervals
        assert result.has_explanations
        assert result.performance_metrics is not None

        # Verify all repositories were called
        mock_detector_repository.find_by_id.assert_called_once_with(detector_id)
        mock_dataset_repository.find_by_id.assert_called_once_with(dataset_id)