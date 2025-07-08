"""
Test suite for DetectionService with mocked infrastructure.
Tests command/query orchestration, side effects, error handling, and async paths.
"""
import pytest
from unittest.mock import AsyncMock, Mock, patch
import asyncio
from uuid import uuid4, UUID
import numpy as np
import pandas as pd

from pynomaly.application.services.detection_service import DetectionService
from pynomaly.domain.entities import Dataset, DetectionResult, Anomaly
from pynomaly.domain.exceptions import DetectorNotFittedError
from pynomaly.domain.value_objects import AnomalyScore, ConfidenceInterval


class TestDetectionService:
    """Test suite for DetectionService with full infrastructure mocking."""

    @pytest.fixture
    def mock_repositories(self):
        """Mock infrastructure repositories."""
        detector_repo = Mock()
        result_repo = Mock()
        anomaly_scorer = Mock()
        threshold_calculator = Mock()
        
        return detector_repo, result_repo, anomaly_scorer, threshold_calculator

    @pytest.fixture
    def detection_service(self, mock_repositories):
        """Create DetectionService with mocked dependencies."""
        detector_repo, result_repo, anomaly_scorer, threshold_calculator = mock_repositories
        
        return DetectionService(
            detector_repository=detector_repo,
            result_repository=result_repo,
            anomaly_scorer=anomaly_scorer,
            threshold_calculator=threshold_calculator
        )

    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for testing."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 100],  # 100 is anomaly
            'feature2': [2, 4, 6, 8, 10, 200]  # 200 is anomaly
        })
        dataset = Dataset(id=uuid4(), name="test_dataset", data=data)
        return dataset

    @pytest.fixture
    def mock_detector(self):
        """Mock detector with required methods."""
        detector = Mock()
        detector.id = uuid4()
        detector.name = "test_detector"
        detector.is_fitted = True
        detector.contamination_rate = 0.1
        
        # Mock detection methods
        detector.detect.return_value = Mock(spec=DetectionResult)
        detector.score.return_value = [
            AnomalyScore(value=0.1, confidence=0.9),
            AnomalyScore(value=0.2, confidence=0.8),
            AnomalyScore(value=0.9, confidence=0.95)  # High anomaly score
        ]
        
        return detector

    @pytest.mark.asyncio
    async def test_detect_with_multiple_detectors_parallel_execution(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test that multiple detectors run in parallel and orchestrate correctly."""
        # Setup
        detector_ids = [uuid4(), uuid4(), uuid4()]
        detection_service.detector_repository.find_by_id.return_value = mock_detector
        
        # Mock async repository behavior
        detection_service.detector_repository.find_by_id = Mock(return_value=mock_detector)
        detection_service.result_repository.save = Mock()
        
        # Execute
        with patch.object(detection_service, '_detect_async', 
                         return_value=Mock(spec=DetectionResult)) as mock_detect:
            results = await detection_service.detect_with_multiple_detectors(
                detector_ids, mock_dataset, save_results=True
            )
        
        # Assertions
        assert len(results) == 3
        assert all(detector_id in results for detector_id in detector_ids)
        assert mock_detect.call_count == 3  # Called for each detector
        assert detection_service.result_repository.save.call_count == 3  # Results saved

    @pytest.mark.asyncio
    async def test_detect_with_multiple_detectors_async_repository(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test async repository pattern handling."""
        detector_ids = [uuid4()]
        
        # Mock async repository methods
        detection_service.detector_repository.find_by_id = AsyncMock(return_value=mock_detector)
        detection_service.result_repository.save = AsyncMock()
        
        with patch.object(detection_service, '_detect_async',
                         return_value=Mock(spec=DetectionResult)) as mock_detect:
            results = await detection_service.detect_with_multiple_detectors(
                detector_ids, mock_dataset, save_results=True
            )
        
        # Verify async methods were called
        detection_service.detector_repository.find_by_id.assert_called_once()
        detection_service.result_repository.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_with_unfitted_detector_error_handling(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test error handling for unfitted detectors."""
        detector_ids = [uuid4()]
        mock_detector.is_fitted = False  # Unfitted detector
        
        detection_service.detector_repository.find_by_id = Mock(return_value=mock_detector)
        
        with pytest.raises(DetectorNotFittedError):
            await detection_service.detect_with_multiple_detectors(
                detector_ids, mock_dataset
            )

    @pytest.mark.asyncio
    async def test_detect_with_missing_detector_error_handling(
        self, detection_service, mock_dataset
    ):
        """Test error handling for missing detectors."""
        detector_ids = [uuid4()]
        detection_service.detector_repository.find_by_id = Mock(return_value=None)
        
        with pytest.raises(ValueError, match="Detector .* not found"):
            await detection_service.detect_with_multiple_detectors(
                detector_ids, mock_dataset
            )

    @pytest.mark.asyncio
    async def test_detect_with_custom_threshold_percentile(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test custom threshold calculation using percentile method."""
        detector_id = uuid4()
        detection_service.detector_repository.find_by_id = Mock(return_value=mock_detector)
        
        # Mock threshold calculator
        expected_threshold = 0.8
        detection_service.threshold_calculator.calculate_by_percentile.return_value = expected_threshold
        
        result = await detection_service.detect_with_custom_threshold(
            detector_id, mock_dataset, 
            threshold_method="percentile", 
            threshold_params={"percentile": 95}
        )
        
        # Verify threshold calculation was called with correct params
        detection_service.threshold_calculator.calculate_by_percentile.assert_called_once_with(
            [0.1, 0.2, 0.9], 95
        )
        
        assert result.threshold == expected_threshold
        assert result.metadata["threshold_method"] == "percentile"

    @pytest.mark.asyncio
    async def test_detect_with_custom_threshold_iqr(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test custom threshold calculation using IQR method."""
        detector_id = uuid4()
        detection_service.detector_repository.find_by_id = Mock(return_value=mock_detector)
        
        expected_threshold = 0.75
        detection_service.threshold_calculator.calculate_by_iqr.return_value = expected_threshold
        
        result = await detection_service.detect_with_custom_threshold(
            detector_id, mock_dataset, 
            threshold_method="iqr", 
            threshold_params={"multiplier": 1.5}
        )
        
        detection_service.threshold_calculator.calculate_by_iqr.assert_called_once_with(
            [0.1, 0.2, 0.9], 1.5
        )
        
        assert result.threshold == expected_threshold

    @pytest.mark.asyncio
    async def test_detect_with_custom_threshold_dynamic(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test custom threshold calculation using dynamic method."""
        detector_id = uuid4()
        detection_service.detector_repository.find_by_id = Mock(return_value=mock_detector)
        
        expected_threshold = 0.6
        detection_service.threshold_calculator.calculate_dynamic_threshold.return_value = (expected_threshold, {})
        
        result = await detection_service.detect_with_custom_threshold(
            detector_id, mock_dataset, 
            threshold_method="dynamic", 
            threshold_params={"method": "knee"}
        )
        
        detection_service.threshold_calculator.calculate_dynamic_threshold.assert_called_once_with(
            [0.1, 0.2, 0.9], method="knee"
        )
        
        assert result.threshold == expected_threshold

    @pytest.mark.asyncio
    async def test_recompute_with_confidence_intervals(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test confidence interval recomputation orchestration."""
        result_id = uuid4()
        
        # Mock original result
        original_result = Mock(spec=DetectionResult)
        original_result.scores = [
            AnomalyScore(value=0.1, confidence=0.9),
            AnomalyScore(value=0.8, confidence=0.95)
        ]
        
        detection_service.result_repository.find_by_id = Mock(return_value=original_result)
        detection_service.result_repository.save = Mock()
        
        # Mock confidence interval addition
        enhanced_scores = [
            Mock(is_confident=True, confidence_lower=0.05, confidence_upper=0.15),
            Mock(is_confident=True, confidence_lower=0.75, confidence_upper=0.85)
        ]
        detection_service.anomaly_scorer.add_confidence_intervals.return_value = enhanced_scores
        
        # Execute
        result = await detection_service.recompute_with_confidence(
            result_id, confidence_level=0.95, method="bootstrap"
        )
        
        # Verify orchestration
        detection_service.anomaly_scorer.add_confidence_intervals.assert_called_once_with(
            original_result.scores, confidence_level=0.95, method="bootstrap"
        )
        
        # Verify side effects
        detection_service.result_repository.save.assert_called_once()
        assert result.scores == enhanced_scores
        assert len(result.confidence_intervals) == 2

    @pytest.mark.asyncio
    async def test_get_detection_history_with_filtering(
        self, detection_service
    ):
        """Test detection history retrieval with filtering."""
        detector_id = uuid4()
        
        # Mock historical results
        mock_results = [
            Mock(spec=DetectionResult, timestamp=pd.Timestamp("2023-01-01")),
            Mock(spec=DetectionResult, timestamp=pd.Timestamp("2023-01-02")),
            Mock(spec=DetectionResult, timestamp=pd.Timestamp("2023-01-03"))
        ]
        
        detection_service.result_repository.find_by_detector = Mock(return_value=mock_results)
        
        results = await detection_service.get_detection_history(
            detector_id=detector_id, limit=2
        )
        
        # Verify filtering and ordering
        detection_service.result_repository.find_by_detector.assert_called_once_with(detector_id)
        assert len(results) == 2  # Limited to 2 results
        # Results should be ordered by timestamp (newest first)
        assert results[0].timestamp >= results[1].timestamp

    @pytest.mark.asyncio
    async def test_get_detection_history_async_repository(
        self, detection_service
    ):
        """Test detection history with async repository pattern."""
        detector_id = uuid4()
        
        mock_results = [Mock(spec=DetectionResult, timestamp=pd.Timestamp("2023-01-01"))]
        detection_service.result_repository.find_by_detector = AsyncMock(return_value=mock_results)
        
        results = await detection_service.get_detection_history(detector_id=detector_id)
        
        detection_service.result_repository.find_by_detector.assert_called_once_with(detector_id)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_compare_detectors_orchestration(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test detector comparison orchestration and metrics calculation."""
        detector_ids = [uuid4(), uuid4()]
        
        # Mock dataset with labels for comparison
        mock_dataset.has_target = True
        mock_dataset.target = pd.Series([0, 0, 1])  # Binary labels
        
        detection_service.detector_repository.find_by_id = Mock(return_value=mock_detector)
        
        # Mock detection results
        mock_result = Mock(spec=DetectionResult)
        mock_result.n_anomalies = 1
        mock_result.anomaly_rate = 0.33
        mock_result.threshold = 0.8
        mock_result.labels = [0, 0, 1]
        mock_result.scores = [
            AnomalyScore(value=0.1, confidence=0.9),
            AnomalyScore(value=0.2, confidence=0.8),
            AnomalyScore(value=0.9, confidence=0.95)
        ]
        
        with patch.object(detection_service, 'detect_with_multiple_detectors',
                         return_value={detector_ids[0]: mock_result, detector_ids[1]: mock_result}):
            comparison = await detection_service.compare_detectors(
                detector_ids, mock_dataset, metrics=["precision", "recall", "f1"]
            )
        
        # Verify orchestration
        assert "detectors" in comparison
        assert "summary" in comparison
        assert len(comparison["detectors"]) == 2
        
        # Verify metrics calculation
        for detector_result in comparison["detectors"].values():
            assert "precision" in detector_result
            assert "recall" in detector_result
            assert "f1" in detector_result

    @pytest.mark.asyncio
    async def test_compare_detectors_without_labels_error(
        self, detection_service, mock_dataset
    ):
        """Test error handling when comparing detectors without labels."""
        detector_ids = [uuid4()]
        mock_dataset.has_target = False
        
        with pytest.raises(ValueError, match="Dataset must have labels for comparison"):
            await detection_service.compare_detectors(detector_ids, mock_dataset)

    @pytest.mark.asyncio
    async def test_async_task_cancellation_handling(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test handling of async task cancellation."""
        detector_ids = [uuid4()]
        detection_service.detector_repository.find_by_id = Mock(return_value=mock_detector)
        
        # Mock a cancelled task
        async def cancelled_detect(*args, **kwargs):
            await asyncio.sleep(0.1)
            raise asyncio.CancelledError("Task was cancelled")
        
        with patch.object(detection_service, '_detect_async', side_effect=cancelled_detect):
            with pytest.raises(asyncio.CancelledError):
                await detection_service.detect_with_multiple_detectors(
                    detector_ids, mock_dataset
                )

    @pytest.mark.asyncio
    async def test_side_effect_verification_model_saving(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test that side effects (saving results) are triggered correctly."""
        detector_ids = [uuid4()]
        detection_service.detector_repository.find_by_id = Mock(return_value=mock_detector)
        detection_service.result_repository.save = Mock()
        
        mock_result = Mock(spec=DetectionResult)
        with patch.object(detection_service, '_detect_async', return_value=mock_result):
            await detection_service.detect_with_multiple_detectors(
                detector_ids, mock_dataset, save_results=True
            )
        
        # Verify side effect was triggered
        detection_service.result_repository.save.assert_called_once_with(mock_result)

    @pytest.mark.asyncio
    async def test_side_effect_verification_no_saving(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test that side effects are not triggered when save_results=False."""
        detector_ids = [uuid4()]
        detection_service.detector_repository.find_by_id = Mock(return_value=mock_detector)
        detection_service.result_repository.save = Mock()
        
        mock_result = Mock(spec=DetectionResult)
        with patch.object(detection_service, '_detect_async', return_value=mock_result):
            await detection_service.detect_with_multiple_detectors(
                detector_ids, mock_dataset, save_results=False
            )
        
        # Verify side effect was NOT triggered
        detection_service.result_repository.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_training_before_detection_requirement(
        self, detection_service, mock_dataset
    ):
        """Test that detection requires training (fitted detector) before detection."""
        detector_ids = [uuid4()]
        
        # Mock unfitted detector
        unfitted_detector = Mock()
        unfitted_detector.id = uuid4()
        unfitted_detector.name = "unfitted_detector"
        unfitted_detector.is_fitted = False
        
        detection_service.detector_repository.find_by_id = Mock(return_value=unfitted_detector)
        
        # Should raise error because training is required before detection
        with pytest.raises(DetectorNotFittedError) as exc_info:
            await detection_service.detect_with_multiple_detectors(
                detector_ids, mock_dataset
            )
        
        assert "detect" in str(exc_info.value)
        assert unfitted_detector.name in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_concurrent_detection_task_orchestration(
        self, detection_service, mock_dataset, mock_detector
    ):
        """Test that multiple detection tasks run concurrently (not sequentially)."""
        detector_ids = [uuid4(), uuid4(), uuid4()]
        detection_service.detector_repository.find_by_id = Mock(return_value=mock_detector)
        
        # Track call timing to verify parallel execution
        call_times = []
        
        async def timed_detect_async(*args, **kwargs):
            import time
            call_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate work
            return Mock(spec=DetectionResult)
        
        with patch.object(detection_service, '_detect_async', side_effect=timed_detect_async):
            start_time = time.time()
            await detection_service.detect_with_multiple_detectors(
                detector_ids, mock_dataset, save_results=False
            )
            total_time = time.time() - start_time
        
        # If tasks ran in parallel, total time should be close to single task time
        # If sequential, it would be ~0.3 seconds (3 * 0.1)
        assert total_time < 0.25  # Should be closer to 0.1 than 0.3
        assert len(call_times) == 3  # All tasks were called
        
        # Verify tasks started close together (parallel execution)
        time_diffs = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
        assert all(diff < 0.05 for diff in time_diffs)  # Tasks started within 50ms of each other
