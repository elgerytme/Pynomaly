"""Tests for database repository implementations."""

import pytest
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from uuid import uuid4

from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.value_objects.algorithm_config import AlgorithmConfig
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.detection_metadata import DetectionMetadata
from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
from pynomaly.infrastructure.persistence.database_repositories import (
    DatabaseDetectorRepository,
    DatabaseDatasetRepository,
    DatabaseDetectionResultRepository,
)
from pynomaly.infrastructure.persistence.database import DatabaseManager
from pynomaly.infrastructure.persistence.database_config import DatabaseSettings


@pytest.fixture
def test_database_settings():
    """Test database settings using SQLite in memory."""
    return DatabaseSettings(
        database_type="sqlite",
        sqlite_path=":memory:",
        create_tables=True,
        migration_enabled=False,
    )


@pytest.fixture
def database_manager(test_database_settings):
    """Database manager for testing."""
    manager = DatabaseManager(f"sqlite:///{test_database_settings.sqlite_path}")
    manager.create_tables()
    yield manager
    manager.close()


@pytest.fixture
def detector_repository(database_manager):
    """Detector repository for testing."""
    return DatabaseDetectorRepository(database_manager.session_factory)


@pytest.fixture
def dataset_repository(database_manager):
    """Dataset repository for testing."""
    return DatabaseDatasetRepository(database_manager.session_factory)


@pytest.fixture
def detection_result_repository(database_manager):
    """Detection result repository for testing."""
    return DatabaseDetectionResultRepository(database_manager.session_factory)


@pytest.fixture
def sample_detector():
    """Sample detector for testing."""
    return Detector(
        id=uuid4(),
        name="Test Detector",
        algorithm_name="isolation_forest",
        contamination_rate=ContaminationRate(0.1),
        parameters={"contamination": 0.1, "random_state": 42},
        created_at=datetime.now(timezone.utc),
        is_fitted=False,
    )


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature3': [10, 20, 30, 40, 50]
    })
    
    return Dataset(
        id=uuid4(),
        name="Test Dataset",
        data=data,
        description="Test dataset for anomaly detection",
        created_at=datetime.now(timezone.utc),
        metadata={"source": "test", "rows": 5}
    )


@pytest.fixture
def sample_detection_result():
    """Sample detection result for testing."""
    return DetectionResult(
        id="test-result-1",
        detector_id="test-detector-1",
        dataset_id="test-dataset-1",
        anomaly_scores=[AnomalyScore(value=0.8, confidence=0.9)],
        metadata=DetectionMetadata(
            execution_time=1.5,
            parameters={"threshold": 0.5},
            timestamp=datetime.now(timezone.utc)
        ),
        created_at=datetime.now(timezone.utc),
    )


class TestDatabaseDetectorRepository:
    """Tests for DatabaseDetectorRepository."""
    
    def test_save_detector(self, detector_repository, sample_detector):
        """Test saving a detector."""
        saved_detector = detector_repository.save(sample_detector)
        
        assert saved_detector.id == sample_detector.id
        assert saved_detector.name == sample_detector.name
        assert saved_detector.algorithm_name == sample_detector.algorithm_name
    
    def test_find_by_id(self, detector_repository, sample_detector):
        """Test finding detector by ID."""
        detector_repository.save(sample_detector)
        
        found_detector = detector_repository.find_by_id(sample_detector.id)
        
        assert found_detector is not None
        assert found_detector.id == sample_detector.id
        assert found_detector.name == sample_detector.name
    
    def test_find_by_id_not_found(self, detector_repository):
        """Test finding detector by ID when not found."""
        found_detector = detector_repository.find_by_id("non-existent")
        
        assert found_detector is None
    
    def test_find_by_name(self, detector_repository, sample_detector):
        """Test finding detector by name."""
        detector_repository.save(sample_detector)
        
        found_detector = detector_repository.find_by_name(sample_detector.name)
        
        assert found_detector is not None
        assert found_detector.name == sample_detector.name
    
    def test_find_by_algorithm(self, detector_repository, sample_detector):
        """Test finding detectors by algorithm."""
        detector_repository.save(sample_detector)
        
        detectors = detector_repository.find_by_algorithm(sample_detector.algorithm_name)
        
        assert len(detectors) == 1
        assert detectors[0].algorithm_name == sample_detector.algorithm_name
    
    def test_find_all(self, detector_repository, sample_detector):
        """Test finding all detectors."""
        detector_repository.save(sample_detector)
        
        detectors = detector_repository.find_all()
        
        assert len(detectors) == 1
        assert detectors[0].id == sample_detector.id
    
    def test_delete(self, detector_repository, sample_detector):
        """Test deleting a detector."""
        detector_repository.save(sample_detector)
        
        success = detector_repository.delete(sample_detector.id)
        
        assert success is True
        assert detector_repository.find_by_id(sample_detector.id) is None
    
    def test_exists(self, detector_repository, sample_detector):
        """Test checking if detector exists."""
        assert detector_repository.exists(sample_detector.id) is False
        
        detector_repository.save(sample_detector)
        
        assert detector_repository.exists(sample_detector.id) is True
    
    def test_count(self, detector_repository, sample_detector):
        """Test counting detectors."""
        assert detector_repository.count() == 0
        
        detector_repository.save(sample_detector)
        
        assert detector_repository.count() == 1


class TestDatabaseDatasetRepository:
    """Tests for DatabaseDatasetRepository."""
    
    def test_save_dataset(self, dataset_repository, sample_dataset):
        """Test saving a dataset."""
        saved_dataset = dataset_repository.save(sample_dataset)
        
        assert saved_dataset.id == sample_dataset.id
        assert saved_dataset.name == sample_dataset.name
        assert saved_dataset.description == sample_dataset.description
    
    def test_find_by_id(self, dataset_repository, sample_dataset):
        """Test finding dataset by ID."""
        dataset_repository.save(sample_dataset)
        
        found_dataset = dataset_repository.find_by_id(sample_dataset.id)
        
        assert found_dataset is not None
        assert found_dataset.id == sample_dataset.id
        assert found_dataset.name == sample_dataset.name
    
    def test_find_by_name(self, dataset_repository, sample_dataset):
        """Test finding dataset by name."""
        dataset_repository.save(sample_dataset)
        
        found_dataset = dataset_repository.find_by_name(sample_dataset.name)
        
        assert found_dataset is not None
        assert found_dataset.name == sample_dataset.name
    
    def test_find_by_metadata(self, dataset_repository, sample_dataset):
        """Test finding datasets by metadata."""
        dataset_repository.save(sample_dataset)
        
        datasets = dataset_repository.find_by_metadata({"source": "test"})
        
        assert len(datasets) == 1
        assert datasets[0].id == sample_dataset.id
    
    def test_save_and_load_data(self, dataset_repository, sample_dataset):
        """Test saving and loading dataset data."""
        # Create sample data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'label': [0, 0, 1, 0, 1]
        })
        
        # Save dataset with data
        dataset_repository.save(sample_dataset)
        dataset_repository.save_data(sample_dataset.id, data)
        
        # Load data
        loaded_data = dataset_repository.load_data(sample_dataset.id)
        
        assert loaded_data is not None
        assert len(loaded_data) == 5
        assert list(loaded_data.columns) == ['feature1', 'feature2', 'label']
        pd.testing.assert_frame_equal(data, loaded_data)


class TestDatabaseDetectionResultRepository:
    """Tests for DatabaseDetectionResultRepository."""
    
    def test_save_detection_result(self, detection_result_repository, sample_detection_result):
        """Test saving a detection result."""
        saved_result = detection_result_repository.save(sample_detection_result)
        
        assert saved_result.id == sample_detection_result.id
        assert saved_result.detector_id == sample_detection_result.detector_id
        assert saved_result.dataset_id == sample_detection_result.dataset_id
    
    def test_find_by_id(self, detection_result_repository, sample_detection_result):
        """Test finding detection result by ID."""
        detection_result_repository.save(sample_detection_result)
        
        found_result = detection_result_repository.find_by_id(sample_detection_result.id)
        
        assert found_result is not None
        assert found_result.id == sample_detection_result.id
        assert found_result.detector_id == sample_detection_result.detector_id
    
    def test_find_by_detector_id(self, detection_result_repository, sample_detection_result):
        """Test finding detection results by detector ID."""
        detection_result_repository.save(sample_detection_result)
        
        results = detection_result_repository.find_by_detector_id(sample_detection_result.detector_id)
        
        assert len(results) == 1
        assert results[0].detector_id == sample_detection_result.detector_id
    
    def test_find_by_dataset_id(self, detection_result_repository, sample_detection_result):
        """Test finding detection results by dataset ID."""
        detection_result_repository.save(sample_detection_result)
        
        results = detection_result_repository.find_by_dataset_id(sample_detection_result.dataset_id)
        
        assert len(results) == 1
        assert results[0].dataset_id == sample_detection_result.dataset_id
    
    def test_find_recent(self, detection_result_repository, sample_detection_result):
        """Test finding recent detection results."""
        detection_result_repository.save(sample_detection_result)
        
        results = detection_result_repository.find_recent(limit=10)
        
        assert len(results) == 1
        assert results[0].id == sample_detection_result.id
    
    def test_get_summary_stats(self, detection_result_repository, sample_detection_result):
        """Test getting summary statistics."""
        detection_result_repository.save(sample_detection_result)
        
        stats = detection_result_repository.get_summary_stats(sample_detection_result.id)
        
        assert stats is not None
        assert "total_samples" in stats
        assert "anomaly_count" in stats
        assert "anomaly_rate" in stats


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database repositories."""
    
    def test_end_to_end_workflow(self, detector_repository, dataset_repository, 
                                detection_result_repository, sample_detector, 
                                sample_dataset, sample_detection_result):
        """Test complete workflow with all repositories."""
        # Save detector
        saved_detector = detector_repository.save(sample_detector)
        assert saved_detector.id == sample_detector.id
        
        # Save dataset
        saved_dataset = dataset_repository.save(sample_dataset)
        assert saved_dataset.id == sample_dataset.id
        
        # Save detection result
        saved_result = detection_result_repository.save(sample_detection_result)
        assert saved_result.id == sample_detection_result.id
        
        # Verify relationships
        found_detector = detector_repository.find_by_id(sample_detector.id)
        found_dataset = dataset_repository.find_by_id(sample_dataset.id)
        found_result = detection_result_repository.find_by_id(sample_detection_result.id)
        
        assert found_detector is not None
        assert found_dataset is not None
        assert found_result is not None
        assert found_result.detector_id == found_detector.id
        assert found_result.dataset_id == found_dataset.id
