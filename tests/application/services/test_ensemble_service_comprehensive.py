"""
Comprehensive tests for ensemble service.
Tests ensemble model management, aggregation strategies, and ensemble optimization.
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.application.services import EnsembleService
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.exceptions import DetectorError
from pynomaly.domain.value_objects import AnomalyScore, PerformanceMetrics


class TestEnsembleService:
    """Test suite for EnsembleService application service."""

    @pytest.fixture
    def mock_detector_repository(self):
        """Mock detector repository."""
        mock_repo = AsyncMock()
        mock_repo.find_by_id.return_value = None
        mock_repo.save.return_value = None
        mock_repo.find_by_ensemble_id.return_value = []
        return mock_repo

    @pytest.fixture
    def mock_ensemble_repository(self):
        """Mock ensemble repository."""
        mock_repo = AsyncMock()
        mock_repo.find_by_id.return_value = None
        mock_repo.save.return_value = None
        mock_repo.delete.return_value = None
        return mock_repo

    @pytest.fixture
    def mock_detection_service(self):
        """Mock detection service."""
        service = AsyncMock()
        service.detect_anomalies.return_value = DetectionResult(
            detector_id=uuid4(),
            anomaly_scores=[AnomalyScore(0.5)],
            anomalies=[],
            threshold=0.5,
        )
        return service

    @pytest.fixture
    def mock_algorithm_registry(self):
        """Mock algorithm registry."""
        registry = Mock()
        registry.get_adapter.return_value = Mock()
        registry.list_algorithms.return_value = [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
        ]
        return registry

    @pytest.fixture
    def sample_detectors(self):
        """Create sample detectors for ensemble."""
        return [
            Detector(
                id=uuid4(),
                name="detector-1",
                algorithm_name="IsolationForest",
                hyperparameters={"n_estimators": 100},
                is_fitted=True,
            ),
            Detector(
                id=uuid4(),
                name="detector-2",
                algorithm_name="LocalOutlierFactor",
                hyperparameters={"n_neighbors": 20},
                is_fitted=True,
            ),
            Detector(
                id=uuid4(),
                name="detector-3",
                algorithm_name="OneClassSVM",
                hyperparameters={"gamma": "scale"},
                is_fitted=True,
            ),
        ]

    @pytest.fixture
    def sample_ensemble(self, sample_detectors):
        """Create sample ensemble."""
        return {
            "id": uuid4(),
            "name": "test-ensemble",
            "detectors": sample_detectors,
            "aggregation_method": "average",
            "weights": [0.4, 0.3, 0.3],
            "is_fitted": True,
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample data for ensemble detection."""
        return np.random.randn(100, 5)

    @pytest.fixture
    def ensemble_service(
        self,
        mock_detector_repository,
        mock_ensemble_repository,
        mock_detection_service,
        mock_algorithm_registry,
    ):
        """Create ensemble service with mocked dependencies."""
        return EnsembleService(
            detector_repository=mock_detector_repository,
            ensemble_repository=mock_ensemble_repository,
            detection_service=mock_detection_service,
            algorithm_registry=mock_algorithm_registry,
        )

    @pytest.mark.asyncio
    async def test_create_ensemble_basic(
        self,
        ensemble_service,
        sample_detectors,
        mock_detector_repository,
        mock_ensemble_repository,
    ):
        """Test basic ensemble creation."""
        # Setup
        detector_ids = [detector.id for detector in sample_detectors]
        for detector in sample_detectors:
            mock_detector_repository.find_by_id.return_value = detector

        # Execute
        ensemble = await ensemble_service.create_ensemble(
            name="test-ensemble",
            detector_ids=detector_ids,
            aggregation_method="average",
        )

        # Verify
        assert ensemble is not None
        assert ensemble.name == "test-ensemble"
        assert len(ensemble.detectors) == len(detector_ids)
        assert ensemble.aggregation_method == "average"
        assert ensemble.weights is None  # Default for average

        # Verify ensemble was saved
        mock_ensemble_repository.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_ensemble_with_weights(
        self,
        ensemble_service,
        sample_detectors,
        mock_detector_repository,
        mock_ensemble_repository,
    ):
        """Test ensemble creation with custom weights."""
        # Setup
        detector_ids = [detector.id for detector in sample_detectors]
        weights = [0.5, 0.3, 0.2]
        for detector in sample_detectors:
            mock_detector_repository.find_by_id.return_value = detector

        # Execute
        ensemble = await ensemble_service.create_ensemble(
            name="weighted-ensemble",
            detector_ids=detector_ids,
            aggregation_method="weighted",
            weights=weights,
        )

        # Verify
        assert ensemble is not None
        assert ensemble.aggregation_method == "weighted"
        assert ensemble.weights == weights

    @pytest.mark.asyncio
    async def test_create_ensemble_validation_errors(
        self,
        ensemble_service,
        sample_detectors,
        mock_detector_repository,
    ):
        """Test ensemble creation validation."""
        # Test with empty detector list
        with pytest.raises(ValueError, match="At least 2 detectors required"):
            await ensemble_service.create_ensemble(
                name="empty-ensemble",
                detector_ids=[],
                aggregation_method="average",
            )

        # Test with single detector
        with pytest.raises(ValueError, match="At least 2 detectors required"):
            await ensemble_service.create_ensemble(
                name="single-ensemble",
                detector_ids=[uuid4()],
                aggregation_method="average",
            )

        # Test with non-existent detector
        mock_detector_repository.find_by_id.return_value = None
        with pytest.raises(DetectorError, match="Detector not found"):
            await ensemble_service.create_ensemble(
                name="invalid-ensemble",
                detector_ids=[uuid4(), uuid4()],
                aggregation_method="average",
            )

    @pytest.mark.asyncio
    async def test_create_ensemble_weight_validation(
        self,
        ensemble_service,
        sample_detectors,
        mock_detector_repository,
    ):
        """Test ensemble weight validation."""
        # Setup
        detector_ids = [detector.id for detector in sample_detectors]
        for detector in sample_detectors:
            mock_detector_repository.find_by_id.return_value = detector

        # Test with mismatched weight length
        with pytest.raises(ValueError, match="Weights must match detector count"):
            await ensemble_service.create_ensemble(
                name="mismatched-ensemble",
                detector_ids=detector_ids,
                aggregation_method="weighted",
                weights=[0.5, 0.5],  # Wrong length
            )

        # Test with negative weights
        with pytest.raises(ValueError, match="Weights must be positive"):
            await ensemble_service.create_ensemble(
                name="negative-ensemble",
                detector_ids=detector_ids,
                aggregation_method="weighted",
                weights=[0.5, -0.2, 0.7],
            )

        # Test with weights not summing to 1
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            await ensemble_service.create_ensemble(
                name="invalid-sum-ensemble",
                detector_ids=detector_ids,
                aggregation_method="weighted",
                weights=[0.5, 0.5, 0.5],  # Sum > 1
            )

    @pytest.mark.asyncio
    async def test_detect_with_ensemble_average(
        self,
        ensemble_service,
        sample_ensemble,
        sample_data,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test ensemble detection with average aggregation."""
        # Setup
        ensemble_id = sample_ensemble["id"]
        mock_ensemble_repository.find_by_id.return_value = sample_ensemble

        # Mock individual detector results
        mock_results = []
        for i, detector in enumerate(sample_ensemble["detectors"]):
            scores = [AnomalyScore(0.1 + i * 0.2) for _ in range(len(sample_data))]
            result = DetectionResult(
                detector_id=detector.id,
                anomaly_scores=scores,
                anomalies=[],
                threshold=0.5,
            )
            mock_results.append(result)

        mock_detection_service.detect_anomalies.side_effect = mock_results

        # Execute
        result = await ensemble_service.detect_with_ensemble(
            ensemble_id=ensemble_id,
            data=sample_data,
        )

        # Verify
        assert result is not None
        assert isinstance(result, DetectionResult)
        assert result.ensemble_id == ensemble_id
        assert len(result.anomaly_scores) == len(sample_data)

        # Verify average aggregation
        # Expected: (0.1 + 0.3 + 0.5) / 3 = 0.3
        expected_avg = (0.1 + 0.3 + 0.5) / 3
        assert abs(result.anomaly_scores[0].value - expected_avg) < 0.001

        # Verify all detectors were called
        assert mock_detection_service.detect_anomalies.call_count == 3

    @pytest.mark.asyncio
    async def test_detect_with_ensemble_weighted(
        self,
        ensemble_service,
        sample_data,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test ensemble detection with weighted aggregation."""
        # Setup weighted ensemble
        weighted_ensemble = {
            "id": uuid4(),
            "name": "weighted-ensemble",
            "detectors": [
                Detector(
                    id=uuid4(),
                    name="detector-1",
                    algorithm_name="IsolationForest",
                    hyperparameters={},
                    is_fitted=True,
                ),
                Detector(
                    id=uuid4(),
                    name="detector-2",
                    algorithm_name="LocalOutlierFactor",
                    hyperparameters={},
                    is_fitted=True,
                ),
            ],
            "aggregation_method": "weighted",
            "weights": [0.7, 0.3],
            "is_fitted": True,
        }

        mock_ensemble_repository.find_by_id.return_value = weighted_ensemble

        # Mock individual detector results
        mock_results = [
            DetectionResult(
                detector_id=weighted_ensemble["detectors"][0].id,
                anomaly_scores=[AnomalyScore(0.8)],
                anomalies=[],
                threshold=0.5,
            ),
            DetectionResult(
                detector_id=weighted_ensemble["detectors"][1].id,
                anomaly_scores=[AnomalyScore(0.2)],
                anomalies=[],
                threshold=0.5,
            ),
        ]

        mock_detection_service.detect_anomalies.side_effect = mock_results

        # Execute
        result = await ensemble_service.detect_with_ensemble(
            ensemble_id=weighted_ensemble["id"],
            data=sample_data[:1],  # Single sample for simplicity
        )

        # Verify
        assert result is not None
        # Expected: 0.8 * 0.7 + 0.2 * 0.3 = 0.56 + 0.06 = 0.62
        expected_weighted = 0.8 * 0.7 + 0.2 * 0.3
        assert abs(result.anomaly_scores[0].value - expected_weighted) < 0.001

    @pytest.mark.asyncio
    async def test_detect_with_ensemble_max_aggregation(
        self,
        ensemble_service,
        sample_data,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test ensemble detection with max aggregation."""
        # Setup max ensemble
        max_ensemble = {
            "id": uuid4(),
            "name": "max-ensemble",
            "detectors": [
                Detector(
                    id=uuid4(),
                    name="detector-1",
                    algorithm_name="IsolationForest",
                    hyperparameters={},
                    is_fitted=True,
                ),
                Detector(
                    id=uuid4(),
                    name="detector-2",
                    algorithm_name="LocalOutlierFactor",
                    hyperparameters={},
                    is_fitted=True,
                ),
            ],
            "aggregation_method": "max",
            "weights": None,
            "is_fitted": True,
        }

        mock_ensemble_repository.find_by_id.return_value = max_ensemble

        # Mock individual detector results
        mock_results = [
            DetectionResult(
                detector_id=max_ensemble["detectors"][0].id,
                anomaly_scores=[AnomalyScore(0.3)],
                anomalies=[],
                threshold=0.5,
            ),
            DetectionResult(
                detector_id=max_ensemble["detectors"][1].id,
                anomaly_scores=[AnomalyScore(0.7)],
                anomalies=[],
                threshold=0.5,
            ),
        ]

        mock_detection_service.detect_anomalies.side_effect = mock_results

        # Execute
        result = await ensemble_service.detect_with_ensemble(
            ensemble_id=max_ensemble["id"],
            data=sample_data[:1],  # Single sample for simplicity
        )

        # Verify
        assert result is not None
        # Expected: max(0.3, 0.7) = 0.7
        assert abs(result.anomaly_scores[0].value - 0.7) < 0.001

    @pytest.mark.asyncio
    async def test_detect_with_ensemble_confidence_intervals(
        self,
        ensemble_service,
        sample_ensemble,
        sample_data,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test ensemble detection with confidence intervals."""
        # Setup
        ensemble_id = sample_ensemble["id"]
        mock_ensemble_repository.find_by_id.return_value = sample_ensemble

        # Mock individual detector results with confidence intervals
        mock_results = []
        for i, detector in enumerate(sample_ensemble["detectors"]):
            base_score = 0.1 + i * 0.2
            scores = [
                AnomalyScore(
                    base_score,
                    confidence_lower=base_score - 0.05,
                    confidence_upper=base_score + 0.05,
                )
            ]
            result = DetectionResult(
                detector_id=detector.id,
                anomaly_scores=scores,
                anomalies=[],
                threshold=0.5,
            )
            mock_results.append(result)

        mock_detection_service.detect_anomalies.side_effect = mock_results

        # Execute
        result = await ensemble_service.detect_with_ensemble(
            ensemble_id=ensemble_id,
            data=sample_data[:1],  # Single sample for simplicity
            include_confidence=True,
        )

        # Verify
        assert result is not None
        assert result.has_confidence_intervals

        # Verify confidence intervals are properly aggregated
        aggregated_score = result.anomaly_scores[0]
        assert aggregated_score.confidence_lower is not None
        assert aggregated_score.confidence_upper is not None
        assert (
            aggregated_score.confidence_lower
            <= aggregated_score.value
            <= aggregated_score.confidence_upper
        )

    @pytest.mark.asyncio
    async def test_optimize_ensemble_weights(
        self,
        ensemble_service,
        sample_ensemble,
        sample_data,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test ensemble weight optimization."""
        # Setup
        ensemble_id = sample_ensemble["id"]
        mock_ensemble_repository.find_by_id.return_value = sample_ensemble

        # Mock validation data and ground truth
        validation_data = sample_data[:50]
        ground_truth = np.random.randint(0, 2, 50)

        # Mock individual detector results
        mock_results = []
        for detector in sample_ensemble["detectors"]:
            scores = [AnomalyScore(np.random.random()) for _ in range(50)]
            result = DetectionResult(
                detector_id=detector.id,
                anomaly_scores=scores,
                anomalies=[],
                threshold=0.5,
            )
            mock_results.append(result)

        mock_detection_service.detect_anomalies.side_effect = mock_results

        # Execute
        optimized_weights = await ensemble_service.optimize_ensemble_weights(
            ensemble_id=ensemble_id,
            validation_data=validation_data,
            ground_truth=ground_truth,
        )

        # Verify
        assert optimized_weights is not None
        assert len(optimized_weights) == len(sample_ensemble["detectors"])
        assert abs(sum(optimized_weights) - 1.0) < 0.001  # Should sum to 1
        assert all(weight >= 0 for weight in optimized_weights)  # Should be positive

    @pytest.mark.asyncio
    async def test_evaluate_ensemble_performance(
        self,
        ensemble_service,
        sample_ensemble,
        sample_data,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test ensemble performance evaluation."""
        # Setup
        ensemble_id = sample_ensemble["id"]
        mock_ensemble_repository.find_by_id.return_value = sample_ensemble

        # Mock evaluation data and ground truth
        evaluation_data = sample_data[:50]
        ground_truth = np.random.randint(0, 2, 50)

        # Mock individual detector results
        mock_results = []
        for detector in sample_ensemble["detectors"]:
            scores = [AnomalyScore(np.random.random()) for _ in range(50)]
            result = DetectionResult(
                detector_id=detector.id,
                anomaly_scores=scores,
                anomalies=[],
                threshold=0.5,
            )
            mock_results.append(result)

        mock_detection_service.detect_anomalies.side_effect = mock_results

        # Execute
        performance = await ensemble_service.evaluate_ensemble_performance(
            ensemble_id=ensemble_id,
            evaluation_data=evaluation_data,
            ground_truth=ground_truth,
        )

        # Verify
        assert performance is not None
        assert isinstance(performance, PerformanceMetrics)
        assert "ensemble_metrics" in performance.additional_metrics
        assert "individual_metrics" in performance.additional_metrics

    @pytest.mark.asyncio
    async def test_ensemble_diversity_analysis(
        self,
        ensemble_service,
        sample_ensemble,
        sample_data,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test ensemble diversity analysis."""
        # Setup
        ensemble_id = sample_ensemble["id"]
        mock_ensemble_repository.find_by_id.return_value = sample_ensemble

        # Mock individual detector results
        mock_results = []
        for i, detector in enumerate(sample_ensemble["detectors"]):
            # Create diverse predictions
            scores = [
                AnomalyScore(0.1 + i * 0.1 + np.random.random() * 0.1)
                for _ in range(10)
            ]
            result = DetectionResult(
                detector_id=detector.id,
                anomaly_scores=scores,
                anomalies=[],
                threshold=0.5,
            )
            mock_results.append(result)

        mock_detection_service.detect_anomalies.side_effect = mock_results

        # Execute
        diversity_metrics = await ensemble_service.analyze_ensemble_diversity(
            ensemble_id=ensemble_id,
            data=sample_data[:10],  # Small sample for simplicity
        )

        # Verify
        assert diversity_metrics is not None
        assert "pairwise_correlations" in diversity_metrics
        assert "ensemble_diversity_score" in diversity_metrics
        assert "disagreement_metrics" in diversity_metrics

    @pytest.mark.asyncio
    async def test_ensemble_streaming_detection(
        self,
        ensemble_service,
        sample_ensemble,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test ensemble streaming detection."""
        # Setup
        ensemble_id = sample_ensemble["id"]
        mock_ensemble_repository.find_by_id.return_value = sample_ensemble

        # Create streaming data generator
        async def data_generator():
            for i in range(5):
                yield np.random.randn(1, 5)

        # Mock streaming detector results
        streaming_results = []
        for i in range(5):
            batch_results = []
            for detector in sample_ensemble["detectors"]:
                scores = [AnomalyScore(np.random.random())]
                result = DetectionResult(
                    detector_id=detector.id,
                    anomaly_scores=scores,
                    anomalies=[],
                    threshold=0.5,
                )
                batch_results.append(result)
            streaming_results.append(batch_results)

        mock_detection_service.detect_anomalies.side_effect = [
            result for batch in streaming_results for result in batch
        ]

        # Execute
        results = []
        async for result in ensemble_service.detect_streaming_with_ensemble(
            ensemble_id=ensemble_id,
            data_stream=data_generator(),
        ):
            results.append(result)

        # Verify
        assert len(results) == 5
        for result in results:
            assert isinstance(result, DetectionResult)
            assert result.ensemble_id == ensemble_id

    @pytest.mark.asyncio
    async def test_ensemble_error_handling(
        self,
        ensemble_service,
        sample_ensemble,
        sample_data,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test ensemble error handling."""
        # Setup
        ensemble_id = sample_ensemble["id"]
        mock_ensemble_repository.find_by_id.return_value = sample_ensemble

        # Mock one detector to fail
        mock_detection_service.detect_anomalies.side_effect = [
            DetectionResult(
                detector_id=sample_ensemble["detectors"][0].id,
                anomaly_scores=[AnomalyScore(0.5)],
                anomalies=[],
                threshold=0.5,
            ),
            RuntimeError("Detector failed"),
            DetectionResult(
                detector_id=sample_ensemble["detectors"][2].id,
                anomaly_scores=[AnomalyScore(0.7)],
                anomalies=[],
                threshold=0.5,
            ),
        ]

        # Execute with error handling
        result = await ensemble_service.detect_with_ensemble(
            ensemble_id=ensemble_id,
            data=sample_data[:1],
            handle_errors=True,
        )

        # Verify
        assert result is not None
        assert result.failed_detectors == [sample_ensemble["detectors"][1].id]
        assert len(result.anomaly_scores) == 1
        # Should aggregate from 2 successful detectors: (0.5 + 0.7) / 2 = 0.6
        assert abs(result.anomaly_scores[0].value - 0.6) < 0.001

    @pytest.mark.asyncio
    async def test_ensemble_concurrent_detection(
        self,
        ensemble_service,
        sample_ensemble,
        sample_data,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test concurrent ensemble detection."""
        # Setup
        ensemble_id = sample_ensemble["id"]
        mock_ensemble_repository.find_by_id.return_value = sample_ensemble

        # Mock concurrent detector results
        mock_results = [
            DetectionResult(
                detector_id=detector.id,
                anomaly_scores=[AnomalyScore(0.5)],
                anomalies=[],
                threshold=0.5,
            )
            for detector in sample_ensemble["detectors"]
        ]

        mock_detection_service.detect_anomalies.side_effect = mock_results

        # Execute concurrent detection
        result = await ensemble_service.detect_with_ensemble(
            ensemble_id=ensemble_id,
            data=sample_data[:1],
            concurrent_execution=True,
        )

        # Verify
        assert result is not None
        assert result.concurrent_execution is True
        assert result.execution_time < 1.0  # Should be faster than sequential

    @pytest.mark.asyncio
    async def test_ensemble_performance_monitoring(
        self,
        ensemble_service,
        sample_ensemble,
        sample_data,
        mock_ensemble_repository,
        mock_detection_service,
    ):
        """Test ensemble performance monitoring."""
        # Setup
        ensemble_id = sample_ensemble["id"]
        mock_ensemble_repository.find_by_id.return_value = sample_ensemble

        # Mock individual detector results
        mock_results = [
            DetectionResult(
                detector_id=detector.id,
                anomaly_scores=[AnomalyScore(0.5)],
                anomalies=[],
                threshold=0.5,
            )
            for detector in sample_ensemble["detectors"]
        ]

        mock_detection_service.detect_anomalies.side_effect = mock_results

        # Execute with performance monitoring
        result = await ensemble_service.detect_with_ensemble(
            ensemble_id=ensemble_id,
            data=sample_data[:1],
            monitor_performance=True,
        )

        # Verify
        assert result is not None
        assert result.performance_metrics is not None
        assert "ensemble_execution_time" in result.performance_metrics
        assert "individual_execution_times" in result.performance_metrics
        assert "aggregation_time" in result.performance_metrics
        assert "memory_usage" in result.performance_metrics
