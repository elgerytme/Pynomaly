"""Tests for application use cases."""

from __future__ import annotations

from uuid import uuid4

import pytest

from pynomaly.application.use_cases import (
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    DetectAnomaliesUseCase,
    TrainDetectorRequest,
    TrainDetectorResponse,
    TrainDetectorUseCase,
)
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.infrastructure.config import Container


class TestTrainDetectorUseCase:
    """Test TrainDetectorUseCase."""
    
    @pytest.mark.asyncio
    async def test_train_detector(
        self,
        container: Container,
        sample_detector: Detector,
        sample_dataset: Dataset
    ):
        """Test training a detector."""
        # Save entities
        detector_repo = container.detector_repository()
        dataset_repo = container.dataset_repository()
        
        detector_repo.save(sample_detector)
        dataset_repo.save(sample_dataset)
        
        # Create use case
        use_case = container.train_detector_use_case()
        
        # Create request
        request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            dataset=sample_dataset,
            validate_data=True,
            save_model=True
        )
        
        # Execute
        response = await use_case.execute(request)
        
        # Verify response
        assert isinstance(response, TrainDetectorResponse)
        assert response.detector_id == sample_detector.id
        assert response.success is True
        assert response.training_time_ms > 0
        assert response.dataset_summary["n_samples"] == sample_dataset.n_samples
        assert response.dataset_summary["n_features"] == sample_dataset.n_features
        
        # Verify detector is trained
        trained_detector = detector_repo.find_by_id(sample_detector.id)
        assert trained_detector.is_fitted
    
    @pytest.mark.asyncio
    async def test_train_detector_not_found(
        self,
        container: Container,
        sample_dataset: Dataset
    ):
        """Test training with non-existent detector."""
        use_case = container.train_detector_use_case()
        
        request = TrainDetectorRequest(
            detector_id=uuid4(),  # Non-existent
            dataset=sample_dataset,
            validate_data=True,
            save_model=True
        )
        
        with pytest.raises(ValueError, match="Detector not found"):
            await use_case.execute(request)


class TestDetectAnomaliesUseCase:
    """Test DetectAnomaliesUseCase."""
    
    @pytest.mark.asyncio
    async def test_detect_anomalies(
        self,
        container: Container,
        trained_detector: Detector,
        sample_dataset: Dataset
    ):
        """Test detecting anomalies."""
        # Save entities
        detector_repo = container.detector_repository()
        dataset_repo = container.dataset_repository()
        
        detector_repo.save(trained_detector)
        dataset_repo.save(sample_dataset)
        
        # Create use case
        use_case = container.detect_anomalies_use_case()
        
        # Create request
        request = DetectAnomaliesRequest(
            detector_id=trained_detector.id,
            dataset=sample_dataset,
            validate_features=True,
            save_results=True
        )
        
        # Execute
        response = await use_case.execute(request)
        
        # Verify response
        assert isinstance(response, DetectAnomaliesResponse)
        assert response.success is True
        assert response.result.detector_id == trained_detector.id
        assert response.result.dataset_id == sample_dataset.id
        assert response.result.n_samples == sample_dataset.n_samples
        assert response.result.n_anomalies >= 0
        assert 0 <= response.result.anomaly_rate <= 1
        assert response.result.execution_time_ms > 0
        
        # Verify results are saved
        result_repo = container.result_repository()
        saved_results = result_repo.find_by_detector(trained_detector.id)
        assert len(saved_results) > 0
    
    @pytest.mark.asyncio
    async def test_detect_with_untrained_detector(
        self,
        container: Container,
        sample_detector: Detector,
        sample_dataset: Dataset
    ):
        """Test detecting with untrained detector."""
        # Save entities
        detector_repo = container.detector_repository()
        dataset_repo = container.dataset_repository()
        
        detector_repo.save(sample_detector)  # Not trained
        dataset_repo.save(sample_dataset)
        
        # Create use case
        use_case = container.detect_anomalies_use_case()
        
        # Create request
        request = DetectAnomaliesRequest(
            detector_id=sample_detector.id,
            dataset=sample_dataset,
            validate_features=True,
            save_results=True
        )
        
        # Should raise error
        with pytest.raises(ValueError, match="not trained"):
            await use_case.execute(request)