"""Tests for infrastructure repository implementations."""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from pynomaly.domain.entities import (
    Anomaly, Dataset, Detector, DetectionResult
)
from pynomaly.domain.value_objects import (
    AnomalyScore, ContaminationRate
)
from pynomaly.infrastructure.repositories import (
    InMemoryDatasetRepository,
    InMemoryDetectorRepository,
    InMemoryResultRepository
)


class MockDetector(Detector):
    """Mock detector for testing."""
    
    def fit(self, dataset: Dataset) -> None:
        """Mock fit method."""
        self.is_fitted = True
        self.trained_at = datetime.utcnow()
    
    def detect(self, dataset: Dataset) -> DetectionResult:
        """Mock detect method."""
        return DetectionResult(
            detector_id=self.id,
            dataset_id=dataset.id,
            anomalies=[],
            scores=[AnomalyScore(0.5)],
            labels=[0],
            threshold=0.5,
            execution_time_ms=100.0
        )
    
    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Mock score method."""
        return [AnomalyScore(0.5)]


class TestInMemoryDetectorRepository:
    """Test InMemoryDetectorRepository."""
    
    def test_init(self):
        """Test repository initialization."""
        repo = InMemoryDetectorRepository()
        
        assert len(repo.find_all()) == 0
    
    def test_save_and_find_by_id(self):
        """Test saving and finding detector by ID."""
        repo = InMemoryDetectorRepository()
        detector = MockDetector(
            name="test_detector",
            algorithm_name="TestAlgorithm"
        )
        
        # Save detector
        repo.save(detector)
        
        # Find by ID
        found = repo.find_by_id(detector.id)
        assert found is not None
        assert found.id == detector.id
        assert found.name == detector.name
        assert found.algorithm_name == detector.algorithm_name
    
    def test_find_by_name(self):
        """Test finding detector by name."""
        repo = InMemoryDetectorRepository()
        detector = MockDetector(
            name="unique_detector",
            algorithm_name="TestAlgorithm"
        )
        
        repo.save(detector)
        
        found = repo.find_by_name("unique_detector")
        assert found is not None
        assert found.name == "unique_detector"
        
        # Test non-existent name
        not_found = repo.find_by_name("non_existent")
        assert not_found is None
    
    def test_find_all(self):
        """Test finding all detectors."""
        repo = InMemoryDetectorRepository()
        
        detectors = [
            MockDetector(name=f"detector_{i}", algorithm_name="TestAlgorithm")
            for i in range(3)
        ]
        
        for detector in detectors:
            repo.save(detector)
        
        all_detectors = repo.find_all()
        assert len(all_detectors) == 3
        
        names = {d.name for d in all_detectors}
        expected_names = {f"detector_{i}" for i in range(3)}
        assert names == expected_names
    
    def test_delete(self):
        """Test deleting detector."""
        repo = InMemoryDetectorRepository()
        detector = MockDetector(
            name="to_delete",
            algorithm_name="TestAlgorithm"
        )
        
        repo.save(detector)
        assert repo.find_by_id(detector.id) is not None
        
        # Delete detector
        success = repo.delete(detector.id)
        assert success is True
        
        # Verify deletion
        assert repo.find_by_id(detector.id) is None
        assert repo.find_by_name("to_delete") is None
        
        # Try to delete non-existent
        success = repo.delete(uuid.uuid4())
        assert success is False
    
    def test_save_model_artifact(self):
        """Test saving model artifacts."""
        repo = InMemoryDetectorRepository()
        detector = MockDetector(
            name="with_artifact",
            algorithm_name="TestAlgorithm"
        )
        
        repo.save(detector)
        
        # Save artifact
        artifact_data = b"mock_model_data"
        repo.save_model_artifact(detector.id, artifact_data)
        
        # Retrieve artifact
        retrieved = repo.load_model_artifact(detector.id)
        assert retrieved == artifact_data
        
        # Test non-existent artifact
        non_existent = repo.load_model_artifact(uuid.uuid4())
        assert non_existent is None


class TestInMemoryDatasetRepository:
    """Test InMemoryDatasetRepository."""
    
    def test_init(self):
        """Test repository initialization."""
        repo = InMemoryDatasetRepository()
        
        assert len(repo.find_all()) == 0
    
    def test_save_and_find_by_id(self):
        """Test saving and finding dataset by ID."""
        repo = InMemoryDatasetRepository()
        
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        dataset = Dataset(name="test_dataset", data=data)
        
        # Save dataset
        repo.save(dataset)
        
        # Find by ID
        found = repo.find_by_id(dataset.id)
        assert found is not None
        assert found.id == dataset.id
        assert found.name == dataset.name
        pd.testing.assert_frame_equal(found.data, dataset.data)
    
    def test_find_by_name(self):
        """Test finding dataset by name."""
        repo = InMemoryDatasetRepository()
        
        data = pd.DataFrame({'x': [1, 2, 3]})
        dataset = Dataset(name="unique_dataset", data=data)
        
        repo.save(dataset)
        
        found = repo.find_by_name("unique_dataset")
        assert found is not None
        assert found.name == "unique_dataset"
        
        # Test non-existent name
        not_found = repo.find_by_name("non_existent")
        assert not_found is None
    
    def test_delete(self):
        """Test deleting dataset."""
        repo = InMemoryDatasetRepository()
        
        data = pd.DataFrame({'x': [1, 2, 3]})
        dataset = Dataset(name="to_delete", data=data)
        
        repo.save(dataset)
        assert repo.find_by_id(dataset.id) is not None
        
        # Delete dataset
        success = repo.delete(dataset.id)
        assert success is True
        
        # Verify deletion
        assert repo.find_by_id(dataset.id) is None
        assert repo.find_by_name("to_delete") is None


class TestInMemoryResultRepository:
    """Test InMemoryResultRepository."""
    
    def test_init(self):
        """Test repository initialization."""
        repo = InMemoryResultRepository()
        
        assert len(repo.find_all()) == 0
    
    def test_save_and_find_by_id(self):
        """Test saving and finding detection result by ID."""
        repo = InMemoryResultRepository()
        
        # Create test data
        detector_id = uuid.uuid4()
        dataset_id = uuid.uuid4()
        
        # Create anomaly to match label=1
        anomaly = Anomaly(
            score=AnomalyScore(0.7),
            data_point={"feature1": 1.5},
            detector_name="test_detector"
        )
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=[AnomalyScore(0.7)],
            labels=[1],
            threshold=0.5,
            execution_time_ms=150.0
        )
        
        # Save result
        repo.save(result)
        
        # Find by ID
        found = repo.find_by_id(result.id)
        assert found is not None
        assert found.id == result.id
        assert found.detector_id == detector_id
        assert found.dataset_id == dataset_id
        assert found.threshold == 0.5
    
    def test_find_by_detector_id(self):
        """Test finding results by detector ID."""
        repo = InMemoryResultRepository()
        
        detector_id = uuid.uuid4()
        dataset_id1 = uuid.uuid4()
        dataset_id2 = uuid.uuid4()
        
        # Create multiple results for same detector
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 2.0},
            detector_name="test_detector"
        )
        
        results = [
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id1,
                anomalies=[],  # No anomalies for label=0
                scores=[AnomalyScore(0.6)],
                labels=[0],
                threshold=0.5,
                execution_time_ms=100.0
            ),
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id2,
                anomalies=[anomaly],  # One anomaly for label=1
                scores=[AnomalyScore(0.8)],
                labels=[1],
                threshold=0.5,
                execution_time_ms=120.0
            )
        ]
        
        for result in results:
            repo.save(result)
        
        # Find by detector ID
        found_results = repo.find_by_detector(detector_id)
        assert len(found_results) == 2
        
        # Verify all results belong to the detector
        for result in found_results:
            assert result.detector_id == detector_id
    
    def test_find_by_dataset_id(self):
        """Test finding results by dataset ID."""
        repo = InMemoryResultRepository()
        
        detector_id1 = uuid.uuid4()
        detector_id2 = uuid.uuid4()
        dataset_id = uuid.uuid4()
        
        # Create multiple results for same dataset
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 2.0},
            detector_name="test_detector2"
        )
        
        results = [
            DetectionResult(
                detector_id=detector_id1,
                dataset_id=dataset_id,
                anomalies=[],  # No anomalies for label=0
                scores=[AnomalyScore(0.6)],
                labels=[0],
                threshold=0.5,
                execution_time_ms=100.0
            ),
            DetectionResult(
                detector_id=detector_id2,
                dataset_id=dataset_id,
                anomalies=[anomaly],  # One anomaly for label=1
                scores=[AnomalyScore(0.8)],
                labels=[1],
                threshold=0.5,
                execution_time_ms=120.0
            )
        ]
        
        for result in results:
            repo.save(result)
        
        # Find by dataset ID
        found_results = repo.find_by_dataset(dataset_id)
        assert len(found_results) == 2
        
        # Verify all results belong to the dataset
        for result in found_results:
            assert result.dataset_id == dataset_id
    
    def test_delete(self):
        """Test deleting detection result."""
        repo = InMemoryResultRepository()
        
        result = DetectionResult(
            detector_id=uuid.uuid4(),
            dataset_id=uuid.uuid4(),
            anomalies=[],  # No anomalies for label=0
            scores=[AnomalyScore(0.5)],
            labels=[0],
            threshold=0.5,
            execution_time_ms=100.0
        )
        
        repo.save(result)
        assert repo.find_by_id(result.id) is not None
        
        # Delete result
        success = repo.delete(result.id)
        assert success is True
        
        # Verify deletion
        assert repo.find_by_id(result.id) is None