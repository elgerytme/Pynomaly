"""
Basic infrastructure repository tests.
Tests core repository functionality with minimal dependencies.
"""

from uuid import uuid4

import pytest

from monorepo.domain.entities import Dataset, Detector
from monorepo.infrastructure.repositories.in_memory_repositories import (
    InMemoryDatasetRepository,
    InMemoryDetectorRepository,
    InMemoryResultRepository,
)


class TestInMemoryDetectorRepository:
    """Test suite for in-memory detector repository."""

    @pytest.fixture
    def detector_repository(self):
        """Create detector repository for testing."""
        return InMemoryDetectorRepository()

    @pytest.fixture
    def sample_detector(self):
        """Create sample detector for testing."""
        return Detector(
            name="test-detector",
            algorithm_name="IsolationForest",
            parameters={"n_estimators": 100},
        )

    def test_detector_repository_save_and_find(
        self, detector_repository, sample_detector
    ):
        """Test basic save and find operations."""
        # Test save
        detector_repository.save(sample_detector)
        assert detector_repository.count() == 1
        assert detector_repository.exists(sample_detector.id)

        # Test find by ID
        found_detector = detector_repository.find_by_id(sample_detector.id)
        assert found_detector is not None
        assert found_detector.id == sample_detector.id
        assert found_detector.name == sample_detector.name

        # Test find by name
        found_by_name = detector_repository.find_by_name(sample_detector.name)
        assert found_by_name is not None
        assert found_by_name.id == sample_detector.id

    def test_detector_repository_find_all(self, detector_repository):
        """Test find all functionality."""
        # Create multiple detectors
        detectors = []
        for i in range(3):
            detector = Detector(
                name=f"detector-{i}",
                algorithm_name="IsolationForest",
                parameters={"param": i},
            )
            detectors.append(detector)
            detector_repository.save(detector)

        # Test find all
        all_detectors = detector_repository.find_all()
        assert len(all_detectors) == 3

        # Verify all detectors are present
        found_names = {d.name for d in all_detectors}
        expected_names = {f"detector-{i}" for i in range(3)}
        assert found_names == expected_names

    def test_detector_repository_delete(self, detector_repository, sample_detector):
        """Test delete functionality."""
        # Save detector
        detector_repository.save(sample_detector)
        assert detector_repository.exists(sample_detector.id)

        # Delete detector
        deleted = detector_repository.delete(sample_detector.id)
        assert deleted is True
        assert detector_repository.count() == 0
        assert not detector_repository.exists(sample_detector.id)

        # Test deleting non-existent detector
        deleted_again = detector_repository.delete(sample_detector.id)
        assert deleted_again is False

    def test_detector_repository_find_by_algorithm(self, detector_repository):
        """Test finding detectors by algorithm."""
        # Create detectors with different algorithms
        algorithms = ["IsolationForest", "LOF", "OneClassSVM", "IsolationForest"]

        for i, algorithm in enumerate(algorithms):
            detector = Detector(
                name=f"detector-{i}",
                algorithm_name=algorithm,
            )
            detector_repository.save(detector)

        # Test find by algorithm
        isolation_forest_detectors = detector_repository.find_by_algorithm(
            "IsolationForest"
        )
        assert len(isolation_forest_detectors) == 2

        lof_detectors = detector_repository.find_by_algorithm("LOF")
        assert len(lof_detectors) == 1

        # Test non-existent algorithm
        non_existent = detector_repository.find_by_algorithm("NonExistent")
        assert len(non_existent) == 0

    def test_detector_repository_find_fitted(self, detector_repository):
        """Test finding fitted detectors."""
        # Create detectors with different fitted status
        for i in range(3):
            detector = Detector(
                name=f"detector-{i}",
                algorithm_name="IsolationForest",
            )
            detector.is_fitted = i % 2 == 0  # Alternate fitted status
            detector_repository.save(detector)

        # Test find fitted
        fitted_detectors = detector_repository.find_fitted()
        assert len(fitted_detectors) == 2  # Indices 0 and 2
        assert all(d.is_fitted for d in fitted_detectors)

    def test_detector_repository_model_artifacts(
        self, detector_repository, sample_detector
    ):
        """Test model artifact storage."""
        detector_repository.save(sample_detector)

        # Test saving model artifact
        model_data = b"serialized_model_data"
        detector_repository.save_model_artifact(sample_detector.id, model_data)

        # Test loading model artifact
        loaded_data = detector_repository.load_model_artifact(sample_detector.id)
        assert loaded_data == model_data

        # Test loading non-existent artifact
        non_existent_id = uuid4()
        assert detector_repository.load_model_artifact(non_existent_id) is None


class TestInMemoryDatasetRepository:
    """Test suite for in-memory dataset repository."""

    @pytest.fixture
    def dataset_repository(self):
        """Create dataset repository for testing."""
        return InMemoryDatasetRepository()

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return Dataset(
            name="test-dataset",
            description="Test dataset",
            metadata={"rows": 1000, "columns": 5},
        )

    def test_dataset_repository_basic_operations(
        self, dataset_repository, sample_dataset
    ):
        """Test basic CRUD operations."""
        # Test save
        dataset_repository.save(sample_dataset)
        assert dataset_repository.count() == 1
        assert dataset_repository.exists(sample_dataset.id)

        # Test find by ID
        found_dataset = dataset_repository.find_by_id(sample_dataset.id)
        assert found_dataset is not None
        assert found_dataset.id == sample_dataset.id
        assert found_dataset.name == sample_dataset.name

        # Test find by name
        found_by_name = dataset_repository.find_by_name(sample_dataset.name)
        assert found_by_name is not None
        assert found_by_name.id == sample_dataset.id

        # Test delete
        deleted = dataset_repository.delete(sample_dataset.id)
        assert deleted is True
        assert dataset_repository.count() == 0
        assert not dataset_repository.exists(sample_dataset.id)

    def test_dataset_repository_multiple_datasets(self, dataset_repository):
        """Test operations with multiple datasets."""
        # Create multiple datasets
        datasets = []
        for i in range(3):
            dataset = Dataset(
                name=f"dataset-{i}",
                description=f"Dataset {i}",
                metadata={"index": i},
            )
            datasets.append(dataset)
            dataset_repository.save(dataset)

        # Test find all
        all_datasets = dataset_repository.find_all()
        assert len(all_datasets) == 3

        # Verify all datasets are present
        found_names = {d.name for d in all_datasets}
        expected_names = {f"dataset-{i}" for i in range(3)}
        assert found_names == expected_names


class TestInMemoryResultRepository:
    """Test suite for in-memory result repository."""

    @pytest.fixture
    def result_repository(self):
        """Create result repository for testing."""
        return InMemoryResultRepository()

    def test_result_repository_basic_operations(self, result_repository):
        """Test basic operations."""
        # Just test that the repository can be created and basic methods exist
        assert result_repository.count() == 0

        # Test with non-existent ID
        non_existent_id = uuid4()
        assert result_repository.find_by_id(non_existent_id) is None
        assert not result_repository.exists(non_existent_id)
        assert result_repository.delete(non_existent_id) is False


class TestRepositoryIntegration:
    """Test integration between repositories."""

    def test_repository_isolation(self):
        """Test that repositories don't interfere with each other."""
        detector_repo = InMemoryDetectorRepository()
        dataset_repo = InMemoryDatasetRepository()
        result_repo = InMemoryResultRepository()

        # Create and save entities
        detector = Detector(name="test-detector", algorithm_name="IsolationForest")
        dataset = Dataset(name="test-dataset", description="Test")

        detector_repo.save(detector)
        dataset_repo.save(dataset)

        # Verify isolation
        assert detector_repo.count() == 1
        assert dataset_repo.count() == 1
        assert result_repo.count() == 0

        # Verify correct retrieval
        assert detector_repo.find_by_name("test-detector") is not None
        assert dataset_repo.find_by_name("test-dataset") is not None
        assert detector_repo.find_by_name("test-dataset") is None
        assert dataset_repo.find_by_name("test-detector") is None
