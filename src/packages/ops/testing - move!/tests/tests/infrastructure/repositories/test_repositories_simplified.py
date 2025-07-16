"""
Simplified infrastructure repository tests.
Tests the core in-memory repository implementations.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.infrastructure.repositories.in_memory_repositories import (
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
            id=uuid4(),
            name="test-detector",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100},
            created_at=datetime.now(UTC),
            is_fitted=False,
        )

    def test_detector_repository_basic_operations(
        self, detector_repository, sample_detector
    ):
        """Test basic CRUD operations for detector repository."""
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

        # Test find all
        all_detectors = detector_repository.find_all()
        assert len(all_detectors) == 1
        assert all_detectors[0].id == sample_detector.id

        # Test delete
        deleted = detector_repository.delete(sample_detector.id)
        assert deleted is True
        assert detector_repository.count() == 0
        assert not detector_repository.exists(sample_detector.id)

    def test_detector_repository_advanced_queries(self, detector_repository):
        """Test advanced query capabilities."""
        # Create multiple detectors with different algorithms
        detectors = []
        algorithms = ["IsolationForest", "LOF", "OneClassSVM", "IsolationForest", "LOF"]

        for i, algorithm in enumerate(algorithms):
            detector = Detector(
                id=uuid4(),
                name=f"detector-{i}",
                algorithm_name=algorithm,
                hyperparameters={},
                created_at=datetime.now(UTC),
                is_fitted=(i % 2 == 0),  # Alternate fitted status
            )
            detectors.append(detector)
            detector_repository.save(detector)

        # Test find by algorithm
        isolation_forest_detectors = detector_repository.find_by_algorithm(
            "IsolationForest"
        )
        assert len(isolation_forest_detectors) == 2
        assert all(
            d.algorithm_name == "IsolationForest" for d in isolation_forest_detectors
        )

        lof_detectors = detector_repository.find_by_algorithm("LOF")
        assert len(lof_detectors) == 2
        assert all(d.algorithm_name == "LOF" for d in lof_detectors)

        # Test find fitted
        fitted_detectors = detector_repository.find_fitted()
        assert len(fitted_detectors) == 3  # Even indices (0, 2, 4)
        assert all(d.is_fitted for d in fitted_detectors)

        # Test non-existent algorithm
        non_existent = detector_repository.find_by_algorithm("NonExistent")
        assert len(non_existent) == 0

    def test_detector_repository_model_artifacts(
        self, detector_repository, sample_detector
    ):
        """Test model artifact storage and retrieval."""
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

        # Test artifact is deleted when detector is deleted
        detector_repository.delete(sample_detector.id)
        assert detector_repository.load_model_artifact(sample_detector.id) is None

    def test_detector_repository_thread_safety(self, detector_repository):
        """Test repository thread safety with concurrent operations."""

        def create_detectors(thread_id, count):
            """Create detectors in a thread."""
            for i in range(count):
                detector = Detector(
                    id=uuid4(),
                    name=f"thread-{thread_id}-detector-{i}",
                    algorithm_name="IsolationForest",
                    hyperparameters={},
                    created_at=datetime.now(UTC),
                )
                detector_repository.save(detector)

        # Create multiple threads
        threads = []
        detectors_per_thread = 5
        thread_count = 3

        for thread_id in range(thread_count):
            thread = threading.Thread(
                target=create_detectors, args=(thread_id, detectors_per_thread)
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all detectors were created
        expected_count = thread_count * detectors_per_thread
        assert detector_repository.count() == expected_count

        # Verify no duplicates in names
        all_detectors = detector_repository.find_all()
        detector_names = [d.name for d in all_detectors]
        assert len(set(detector_names)) == expected_count


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
            id=uuid4(),
            name="test-dataset",
            description="Test dataset",
            metadata={"rows": 1000, "columns": 5},
            created_at=datetime.now(UTC),
        )

    def test_dataset_repository_basic_operations(
        self, dataset_repository, sample_dataset
    ):
        """Test basic CRUD operations for dataset repository."""
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

        # Test find all
        all_datasets = dataset_repository.find_all()
        assert len(all_datasets) == 1
        assert all_datasets[0].id == sample_dataset.id

        # Test delete
        deleted = dataset_repository.delete(sample_dataset.id)
        assert deleted is True
        assert dataset_repository.count() == 0
        assert not dataset_repository.exists(sample_dataset.id)

    def test_dataset_repository_data_storage(self, dataset_repository, sample_dataset):
        """Test dataset data storage and retrieval."""
        dataset_repository.save(sample_dataset)

        # Test saving dataset data
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
                "target": [0, 0, 1, 0, 1],
            }
        )
        dataset_repository.save_data(sample_dataset.id, data)

        # Test loading dataset data
        loaded_data = dataset_repository.load_data(sample_dataset.id)
        assert loaded_data is not None
        pd.testing.assert_frame_equal(loaded_data, data)

        # Test loading non-existent data
        non_existent_id = uuid4()
        assert dataset_repository.load_data(non_existent_id) is None

        # Test data is deleted when dataset is deleted
        dataset_repository.delete(sample_dataset.id)
        assert dataset_repository.load_data(sample_dataset.id) is None

    def test_dataset_repository_metadata_operations(
        self, dataset_repository, sample_dataset
    ):
        """Test dataset metadata operations."""
        dataset_repository.save(sample_dataset)

        # Test updating metadata
        new_metadata = {"rows": 2000, "columns": 10, "size_mb": 5.5}
        dataset_repository.update_metadata(sample_dataset.id, new_metadata)

        updated_dataset = dataset_repository.find_by_id(sample_dataset.id)
        assert updated_dataset.metadata == new_metadata

        # Test partial metadata update
        partial_update = {"processed": True, "version": "1.1"}
        dataset_repository.update_metadata(sample_dataset.id, partial_update)

        updated_again = dataset_repository.find_by_id(sample_dataset.id)
        expected_metadata = {**new_metadata, **partial_update}
        assert updated_again.metadata == expected_metadata


class TestInMemoryResultRepository:
    """Test suite for in-memory result repository."""

    @pytest.fixture
    def result_repository(self):
        """Create result repository for testing."""
        return InMemoryResultRepository()

    @pytest.fixture
    def sample_detection_result(self):
        """Create sample detection result for testing."""
        return DetectionResult(
            id=uuid4(),
            detector_id=uuid4(),
            dataset_id=uuid4(),
            anomaly_scores=[0.1, 0.9, 0.3, 0.8, 0.2],
            predictions=[False, True, False, True, False],
            threshold=0.5,
            created_at=datetime.now(UTC),
        )

    def test_result_repository_basic_operations(
        self, result_repository, sample_detection_result
    ):
        """Test basic CRUD operations for result repository."""
        # Test save
        result_repository.save(sample_detection_result)
        assert result_repository.count() == 1
        assert result_repository.exists(sample_detection_result.id)

        # Test find by ID
        found_result = result_repository.find_by_id(sample_detection_result.id)
        assert found_result is not None
        assert found_result.id == sample_detection_result.id
        assert found_result.detector_id == sample_detection_result.detector_id

        # Test find all
        all_results = result_repository.find_all()
        assert len(all_results) == 1
        assert all_results[0].id == sample_detection_result.id

        # Test delete
        deleted = result_repository.delete(sample_detection_result.id)
        assert deleted is True
        assert result_repository.count() == 0
        assert not result_repository.exists(sample_detection_result.id)

    def test_result_repository_queries(self, result_repository):
        """Test query capabilities for detection results."""
        # Create multiple detection results
        detector_ids = [uuid4() for _ in range(3)]
        dataset_ids = [uuid4() for _ in range(2)]

        results = []
        for i in range(10):
            result = DetectionResult(
                id=uuid4(),
                detector_id=detector_ids[i % 3],
                dataset_id=dataset_ids[i % 2],
                anomaly_scores=[0.1 * j for j in range(5)],
                predictions=[j % 2 == 0 for j in range(5)],
                threshold=0.5,
                created_at=datetime.now(UTC),
            )
            results.append(result)
            result_repository.save(result)

        # Test find by detector
        detector_0_results = result_repository.find_by_detector(detector_ids[0])
        expected_count = len([r for r in results if r.detector_id == detector_ids[0]])
        assert len(detector_0_results) == expected_count

        # Test find by dataset
        dataset_0_results = result_repository.find_by_dataset(dataset_ids[0])
        expected_count = len([r for r in results if r.dataset_id == dataset_ids[0]])
        assert len(dataset_0_results) == expected_count

        # Test find recent
        recent_results = result_repository.find_recent(limit=5)
        assert len(recent_results) == 5
        # Results should be sorted by creation time (most recent first)
        for i in range(len(recent_results) - 1):
            assert recent_results[i].created_at >= recent_results[i + 1].created_at


class TestRepositoryPerformance:
    """Test suite for repository performance characteristics."""

    def test_detector_repository_large_scale(self):
        """Test detector repository performance with large number of entities."""
        repository = InMemoryDetectorRepository()

        start_time = time.time()

        # Create large number of detectors
        large_count = 500
        detector_ids = []

        for i in range(large_count):
            detector = Detector(
                id=uuid4(),
                name=f"detector-{i:04d}",
                algorithm_name=["IsolationForest", "LOF", "OneClassSVM"][i % 3],
                hyperparameters={"param": i},
                created_at=datetime.now(UTC),
                is_fitted=(i % 10 == 0),
            )
            detector_ids.append(detector.id)
            repository.save(detector)

        creation_time = time.time() - start_time
        assert creation_time < 5.0  # Should complete within 5 seconds

        # Test bulk operations performance
        start_time = time.time()

        # Test find all
        all_detectors = repository.find_all()
        assert len(all_detectors) == large_count

        # Test find by algorithm
        isolation_forest_detectors = repository.find_by_algorithm("IsolationForest")
        expected_if_count = len([i for i in range(large_count) if i % 3 == 0])
        assert len(isolation_forest_detectors) == expected_if_count

        # Test find fitted
        fitted_detectors = repository.find_fitted()
        expected_fitted_count = len([i for i in range(large_count) if i % 10 == 0])
        assert len(fitted_detectors) == expected_fitted_count

        query_time = time.time() - start_time
        assert query_time < 2.0  # Should complete within 2 seconds

    def test_dataset_repository_memory_efficiency(self):
        """Test dataset repository memory efficiency."""
        repository = InMemoryDatasetRepository()

        # Create datasets with varying data sizes
        dataset_count = 50

        for i in range(dataset_count):
            dataset = Dataset(
                id=uuid4(),
                name=f"memory-test-dataset-{i}",
                description=f"Memory test dataset {i}",
                metadata={"size": i * 100},
                created_at=datetime.now(UTC),
            )
            repository.save(dataset)

            # Add data of varying sizes
            rows = 50 + i * 5
            data = pd.DataFrame(
                {"feature1": np.random.randn(rows), "feature2": np.random.randn(rows)}
            )
            repository.save_data(dataset.id, data)

        # Verify all data is accessible
        for i in range(0, dataset_count, 5):  # Sample every 5th dataset
            dataset_name = f"memory-test-dataset-{i}"
            dataset = repository.find_by_name(dataset_name)
            assert dataset is not None

            data = repository.load_data(dataset.id)
            assert data is not None
            expected_rows = 50 + i * 5
            assert len(data) == expected_rows


class TestRepositoryErrorHandling:
    """Test suite for repository error handling scenarios."""

    def test_detector_repository_error_handling(self):
        """Test detector repository error handling scenarios."""
        repository = InMemoryDetectorRepository()

        # Test operations on non-existent entities
        non_existent_id = uuid4()

        assert repository.find_by_id(non_existent_id) is None
        assert repository.delete(non_existent_id) is False
        assert not repository.exists(non_existent_id)
        assert repository.load_model_artifact(non_existent_id) is None

        # Test operations with invalid data
        assert repository.find_by_name("") is None
        assert repository.find_by_algorithm("") == []

    def test_dataset_repository_data_consistency(self):
        """Test dataset repository data consistency."""
        repository = InMemoryDatasetRepository()

        dataset = Dataset(
            id=uuid4(),
            name="consistency-test",
            description="Test consistency",
            metadata={"test": True},
            created_at=datetime.now(UTC),
        )
        repository.save(dataset)

        # Test data consistency after multiple updates
        for i in range(5):
            updated_metadata = {"iteration": i, "timestamp": time.time()}
            repository.update_metadata(dataset.id, updated_metadata)

            # Verify consistency
            retrieved = repository.find_by_id(dataset.id)
            assert retrieved.metadata["iteration"] == i
            assert "timestamp" in retrieved.metadata

    def test_result_repository_concurrent_access(self):
        """Test concurrent access to result repository."""
        repository = InMemoryResultRepository()

        def create_results(worker_id, count):
            """Create results in a worker."""
            for i in range(count):
                result = DetectionResult(
                    id=uuid4(),
                    detector_id=uuid4(),
                    dataset_id=uuid4(),
                    anomaly_scores=[0.1, 0.9],
                    predictions=[False, True],
                    threshold=0.5,
                    created_at=datetime.now(UTC),
                )
                repository.save(result)

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for worker_id in range(3):
                future = executor.submit(create_results, worker_id, 5)
                futures.append(future)

            for future in futures:
                future.result()

        # Verify all results were saved
        assert repository.count() == 15  # 3 workers * 5 results each


class TestRepositoryIntegration:
    """Integration tests for repository systems."""

    def test_cross_repository_operations(self):
        """Test operations across multiple repositories."""
        detector_repo = InMemoryDetectorRepository()
        dataset_repo = InMemoryDatasetRepository()
        result_repo = InMemoryResultRepository()

        # Create dataset
        dataset = Dataset(
            id=uuid4(),
            name="integration-dataset",
            description="Integration test dataset",
            metadata={"rows": 1000},
            created_at=datetime.now(UTC),
        )
        dataset_repo.save(dataset)

        # Create detector
        detector = Detector(
            id=uuid4(),
            name="integration-detector",
            algorithm_name="IsolationForest",
            hyperparameters={},
            created_at=datetime.now(UTC),
            is_fitted=True,
        )
        detector_repo.save(detector)

        # Create detection result linking dataset and detector
        result = DetectionResult(
            id=uuid4(),
            detector_id=detector.id,
            dataset_id=dataset.id,
            anomaly_scores=[0.1, 0.9, 0.3],
            predictions=[False, True, False],
            threshold=0.5,
            created_at=datetime.now(UTC),
        )
        result_repo.save(result)

        # Verify relationships
        found_result = result_repo.find_by_id(result.id)
        assert found_result.detector_id == detector.id
        assert found_result.dataset_id == dataset.id

        # Test cascade operations
        detector_results = result_repo.find_by_detector(detector.id)
        assert len(detector_results) == 1
        assert detector_results[0].id == result.id

        dataset_results = result_repo.find_by_dataset(dataset.id)
        assert len(dataset_results) == 1
        assert dataset_results[0].id == result.id
