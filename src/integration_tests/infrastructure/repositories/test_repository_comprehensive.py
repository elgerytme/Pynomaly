"""
Comprehensive infrastructure repository tests.
Tests all repository implementations with extensive scenarios including
threading, transactions, error recovery, and performance validation.
"""

import asyncio
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


class TestInMemoryRepositories:
    """Test suite for in-memory repository implementations."""

    @pytest.fixture
    def detector_repository(self):
        """Create detector repository for testing."""
        return InMemoryDetectorRepository()

    @pytest.fixture
    def dataset_repository(self):
        """Create dataset repository for testing."""
        return InMemoryDatasetRepository()

    @pytest.fixture
    def detection_result_repository(self):
        """Create detection result repository for testing."""
        return InMemoryResultRepository()

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

    @pytest.fixture
    def sample_detection_result(self, sample_detector, sample_dataset):
        """Create sample detection result for testing."""
        return DetectionResult(
            id=uuid4(),
            detector_id=sample_detector.id,
            dataset_id=sample_dataset.id,
            anomaly_scores=[0.1, 0.9, 0.3, 0.8, 0.2],
            predictions=[False, True, False, True, False],
            threshold=0.5,
            created_at=datetime.now(UTC),
        )

    # Detector Repository Tests

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

    def test_detector_repository_name_conflicts(self, detector_repository):
        """Test handling of name conflicts."""
        # Create first detector
        detector1 = Detector(
            id=uuid4(),
            name="same-name",
            algorithm_name="IsolationForest",
            hyperparameters={},
            created_at=datetime.now(UTC),
        )
        detector_repository.save(detector1)

        # Create second detector with same name (should overwrite in name index)
        detector2 = Detector(
            id=uuid4(),
            name="same-name",
            algorithm_name="LOF",
            hyperparameters={},
            created_at=datetime.now(UTC),
        )
        detector_repository.save(detector2)

        # Should find the latest detector by name
        found_by_name = detector_repository.find_by_name("same-name")
        assert found_by_name.id == detector2.id
        assert found_by_name.algorithm_name == "LOF"

        # Both detectors should still exist by ID
        assert detector_repository.find_by_id(detector1.id) is not None
        assert detector_repository.find_by_id(detector2.id) is not None
        assert detector_repository.count() == 2

    # Dataset Repository Tests

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

    def test_dataset_repository_tags_and_filtering(self, dataset_repository):
        """Test dataset tags and filtering capabilities."""
        # Create datasets with different tags
        datasets_with_tags = [
            (
                Dataset(
                    id=uuid4(),
                    name=f"dataset-{i}",
                    description=f"Dataset {i}",
                    metadata={"tag_type": tag_type},
                    created_at=datetime.now(UTC),
                ),
                tags,
            )
            for i, (tag_type, tags) in enumerate(
                [
                    ("financial", ["finance", "fraud"]),
                    ("network", ["security", "intrusion"]),
                    ("iot", ["sensors", "anomaly"]),
                    ("mixed", ["finance", "security"]),
                    ("single", ["single_tag"]),
                ]
            )
        ]

        # Save all datasets
        for dataset, tags in datasets_with_tags:
            dataset_repository.save(dataset)
            dataset_repository.add_tags(dataset.id, tags)

        # Test find by single tag
        finance_datasets = dataset_repository.find_by_tags(["finance"])
        assert len(finance_datasets) == 2  # financial and mixed

        security_datasets = dataset_repository.find_by_tags(["security"])
        assert len(security_datasets) == 2  # network and mixed

        # Test find by multiple tags (intersection)
        finance_security = dataset_repository.find_by_tags(["finance", "security"])
        assert len(finance_security) == 1  # only mixed dataset

        # Test find by non-existent tag
        non_existent = dataset_repository.find_by_tags(["non_existent"])
        assert len(non_existent) == 0

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

    # Detection Result Repository Tests

    def test_detection_result_repository_basic_operations(
        self, detection_result_repository, sample_detection_result
    ):
        """Test basic CRUD operations for detection result repository."""
        # Test save
        detection_result_repository.save(sample_detection_result)
        assert detection_result_repository.count() == 1
        assert detection_result_repository.exists(sample_detection_result.id)

        # Test find by ID
        found_result = detection_result_repository.find_by_id(
            sample_detection_result.id
        )
        assert found_result is not None
        assert found_result.id == sample_detection_result.id
        assert found_result.detector_id == sample_detection_result.detector_id

        # Test find all
        all_results = detection_result_repository.find_all()
        assert len(all_results) == 1
        assert all_results[0].id == sample_detection_result.id

        # Test delete
        deleted = detection_result_repository.delete(sample_detection_result.id)
        assert deleted is True
        assert detection_result_repository.count() == 0
        assert not detection_result_repository.exists(sample_detection_result.id)

    def test_detection_result_repository_queries(self, detection_result_repository):
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
            detection_result_repository.save(result)

        # Test find by detector
        detector_0_results = detection_result_repository.find_by_detector(
            detector_ids[0]
        )
        expected_count = len([r for r in results if r.detector_id == detector_ids[0]])
        assert len(detector_0_results) == expected_count

        # Test find by dataset
        dataset_0_results = detection_result_repository.find_by_dataset(dataset_ids[0])
        expected_count = len([r for r in results if r.dataset_id == dataset_ids[0]])
        assert len(dataset_0_results) == expected_count

        # Test find recent
        recent_results = detection_result_repository.find_recent(limit=5)
        assert len(recent_results) == 5
        # Results should be sorted by creation time (most recent first)
        for i in range(len(recent_results) - 1):
            assert recent_results[i].created_at >= recent_results[i + 1].created_at

    # Threading and Concurrency Tests

    def test_repository_thread_safety(self, detector_repository):
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
        detectors_per_thread = 10
        thread_count = 5

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

    def test_repository_concurrent_read_write(self, dataset_repository):
        """Test concurrent read and write operations."""
        # Pre-populate with some datasets
        datasets = []
        for i in range(20):
            dataset = Dataset(
                id=uuid4(),
                name=f"dataset-{i}",
                description=f"Dataset {i}",
                metadata={"index": i},
                created_at=datetime.now(UTC),
            )
            datasets.append(dataset)
            dataset_repository.save(dataset)

        results = []

        def reader_worker():
            """Worker that performs read operations."""
            for _ in range(50):
                all_datasets = dataset_repository.find_all()
                results.append(len(all_datasets))
                time.sleep(0.001)  # Small delay

        def writer_worker():
            """Worker that performs write operations."""
            for i in range(10):
                dataset = Dataset(
                    id=uuid4(),
                    name=f"new-dataset-{i}",
                    description=f"New dataset {i}",
                    metadata={"new": True},
                    created_at=datetime.now(UTC),
                )
                dataset_repository.save(dataset)
                time.sleep(0.002)  # Small delay

        # Start concurrent readers and writers
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            # Start readers
            for _ in range(2):
                futures.append(executor.submit(reader_worker))

            # Start writers
            for _ in range(2):
                futures.append(executor.submit(writer_worker))

            # Wait for completion
            for future in futures:
                future.result()

        # Verify final state
        final_count = dataset_repository.count()
        assert final_count == 40  # 20 initial + 20 new (10 per writer)

        # All read operations should have returned reasonable counts
        assert all(count >= 20 for count in results)
        assert all(count <= 40 for count in results)

    # Performance and Stress Tests

    def test_repository_large_scale_operations(self, detector_repository):
        """Test repository performance with large number of entities."""
        start_time = time.time()

        # Create large number of detectors
        large_count = 1000
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
            detector_repository.save(detector)

        creation_time = time.time() - start_time
        assert creation_time < 10.0  # Should complete within 10 seconds

        # Test bulk operations performance
        start_time = time.time()

        # Test find all
        all_detectors = detector_repository.find_all()
        assert len(all_detectors) == large_count

        # Test find by algorithm
        isolation_forest_detectors = detector_repository.find_by_algorithm(
            "IsolationForest"
        )
        expected_if_count = len([i for i in range(large_count) if i % 3 == 0])
        assert len(isolation_forest_detectors) == expected_if_count

        # Test find fitted
        fitted_detectors = detector_repository.find_fitted()
        expected_fitted_count = len([i for i in range(large_count) if i % 10 == 0])
        assert len(fitted_detectors) == expected_fitted_count

        # Test random access
        for _ in range(100):
            random_id = detector_ids[np.random.randint(0, large_count)]
            found = detector_repository.find_by_id(random_id)
            assert found is not None

        query_time = time.time() - start_time
        assert query_time < 5.0  # Should complete within 5 seconds

    def test_repository_memory_efficiency(self, dataset_repository):
        """Test repository memory efficiency with large datasets."""
        # Create datasets with varying data sizes
        dataset_count = 100

        for i in range(dataset_count):
            dataset = Dataset(
                id=uuid4(),
                name=f"memory-test-dataset-{i}",
                description=f"Memory test dataset {i}",
                metadata={"size": i * 100},
                created_at=datetime.now(UTC),
            )
            dataset_repository.save(dataset)

            # Add data of varying sizes
            rows = 100 + i * 10
            data = pd.DataFrame(
                {
                    "feature1": np.random.randn(rows),
                    "feature2": np.random.randn(rows),
                    "feature3": np.random.randn(rows),
                }
            )
            dataset_repository.save_data(dataset.id, data)

        # Verify all data is accessible
        for i in range(0, dataset_count, 10):  # Sample every 10th dataset
            dataset_name = f"memory-test-dataset-{i}"
            dataset = dataset_repository.find_by_name(dataset_name)
            assert dataset is not None

            data = dataset_repository.load_data(dataset.id)
            assert data is not None
            expected_rows = 100 + i * 10
            assert len(data) == expected_rows

    # Error Handling and Edge Cases

    def test_repository_error_handling(self, detector_repository):
        """Test repository error handling scenarios."""
        # Test operations on non-existent entities
        non_existent_id = uuid4()

        assert detector_repository.find_by_id(non_existent_id) is None
        assert detector_repository.delete(non_existent_id) is False
        assert not detector_repository.exists(non_existent_id)
        assert detector_repository.load_model_artifact(non_existent_id) is None

        # Test operations with invalid data
        assert detector_repository.find_by_name("") is None
        assert detector_repository.find_by_name(None) is None
        assert detector_repository.find_by_algorithm("") == []
        assert detector_repository.find_by_algorithm(None) == []

    def test_repository_data_consistency(self, dataset_repository):
        """Test repository data consistency under various conditions."""
        dataset = Dataset(
            id=uuid4(),
            name="consistency-test",
            description="Test consistency",
            metadata={"test": True},
            created_at=datetime.now(UTC),
        )
        dataset_repository.save(dataset)

        # Test data consistency after multiple updates
        for i in range(10):
            updated_metadata = {"iteration": i, "timestamp": time.time()}
            dataset_repository.update_metadata(dataset.id, updated_metadata)

            # Verify consistency
            retrieved = dataset_repository.find_by_id(dataset.id)
            assert retrieved.metadata["iteration"] == i
            assert "timestamp" in retrieved.metadata

        # Test data consistency with tags
        tags_sets = [["tag1", "tag2"], ["tag2", "tag3"], ["tag1", "tag3", "tag4"]]

        for tags in tags_sets:
            dataset_repository.add_tags(dataset.id, tags)
            found_datasets = dataset_repository.find_by_tags(tags)
            assert any(d.id == dataset.id for d in found_datasets)


class TestFileRepositories:
    """Test suite for file-based repository implementations."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for file repositories."""
        return tmp_path

    @pytest.fixture
    def file_detector_repository(self, temp_dir):
        """Create file-based detector repository."""
        return FileDetectorRepository(storage_path=str(temp_dir / "detectors"))

    @pytest.fixture
    def file_dataset_repository(self, temp_dir):
        """Create file-based dataset repository."""
        return FileDatasetRepository(storage_path=str(temp_dir / "datasets"))

    def test_file_repository_persistence(self, file_detector_repository):
        """Test file repository persistence across instances."""
        # Create and save a detector
        detector = Detector(
            id=uuid4(),
            name="persistent-detector",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100},
            created_at=datetime.now(UTC),
        )
        file_detector_repository.save(detector)

        # Create new repository instance with same storage path
        storage_path = file_detector_repository._storage_path
        new_repository = FileDetectorRepository(storage_path=storage_path)

        # Verify detector is still accessible
        found_detector = new_repository.find_by_id(detector.id)
        assert found_detector is not None
        assert found_detector.id == detector.id
        assert found_detector.name == detector.name

    def test_file_repository_concurrent_access(self, temp_dir):
        """Test concurrent access to file repositories."""
        storage_path = str(temp_dir / "concurrent_test")

        def worker(worker_id, detector_count):
            """Worker function for concurrent testing."""
            repository = FileDetectorRepository(storage_path=storage_path)

            for i in range(detector_count):
                detector = Detector(
                    id=uuid4(),
                    name=f"worker-{worker_id}-detector-{i}",
                    algorithm_name="IsolationForest",
                    hyperparameters={},
                    created_at=datetime.now(UTC),
                )
                repository.save(detector)

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for worker_id in range(3):
                future = executor.submit(worker, worker_id, 5)
                futures.append(future)

            for future in futures:
                future.result()

        # Verify all detectors were saved
        final_repository = FileDetectorRepository(storage_path=storage_path)
        all_detectors = final_repository.find_all()
        assert len(all_detectors) == 15  # 3 workers * 5 detectors each

        # Verify unique names
        detector_names = [d.name for d in all_detectors]
        assert len(set(detector_names)) == 15


class TestRepositoryFactory:
    """Test suite for repository factory."""

    def test_repository_factory_creation(self):
        """Test repository factory creates correct instances."""
        factory = RepositoryFactory()

        # Test in-memory repositories
        detector_repo = factory.create_detector_repository("memory")
        assert isinstance(detector_repo, InMemoryDetectorRepository)

        dataset_repo = factory.create_dataset_repository("memory")
        assert isinstance(dataset_repo, InMemoryDatasetRepository)

        result_repo = factory.create_detection_result_repository("memory")
        assert isinstance(result_repo, InMemoryResultRepository)

    def test_repository_factory_configuration(self, tmp_path):
        """Test repository factory with different configurations."""
        factory = RepositoryFactory()

        # Test file repositories
        file_detector_repo = factory.create_detector_repository(
            "file", config={"storage_path": str(tmp_path / "detectors")}
        )
        assert isinstance(file_detector_repo, FileDetectorRepository)

        file_dataset_repo = factory.create_dataset_repository(
            "file", config={"storage_path": str(tmp_path / "datasets")}
        )
        assert isinstance(file_dataset_repo, FileDatasetRepository)

    def test_repository_factory_invalid_type(self):
        """Test repository factory with invalid repository type."""
        factory = RepositoryFactory()

        with pytest.raises(ValueError):
            factory.create_detector_repository("invalid_type")

        with pytest.raises(ValueError):
            factory.create_dataset_repository("invalid_type")


class TestAsyncRepositoryWrapper:
    """Test suite for async repository wrappers."""

    @pytest.fixture
    def sync_repository(self):
        """Create synchronous repository for wrapping."""
        return InMemoryDetectorRepository()

    @pytest.fixture
    def async_repository(self, sync_repository):
        """Create async repository wrapper."""
        return AsyncRepositoryWrapper(sync_repository)

    def test_async_wrapper_basic_operations(self, async_repository):
        """Test basic async operations."""

        async def test_operations():
            # Create detector
            detector = Detector(
                id=uuid4(),
                name="async-detector",
                algorithm_name="IsolationForest",
                hyperparameters={},
                created_at=datetime.now(UTC),
            )

            # Test async save
            await async_repository.save(detector)

            # Test async find by ID
            found = await async_repository.find_by_id(detector.id)
            assert found is not None
            assert found.id == detector.id

            # Test async find all
            all_detectors = await async_repository.find_all()
            assert len(all_detectors) == 1

            # Test async delete
            deleted = await async_repository.delete(detector.id)
            assert deleted is True

            # Test async count
            count = await async_repository.count()
            assert count == 0

        # Run async test
        asyncio.run(test_operations())

    def test_async_wrapper_concurrent_operations(self, async_repository):
        """Test concurrent async operations."""

        async def create_detector(index):
            """Create a detector asynchronously."""
            detector = Detector(
                id=uuid4(),
                name=f"concurrent-detector-{index}",
                algorithm_name="IsolationForest",
                hyperparameters={},
                created_at=datetime.now(UTC),
            )
            await async_repository.save(detector)
            return detector.id

        async def test_concurrent():
            # Create multiple detectors concurrently
            tasks = [create_detector(i) for i in range(10)]
            detector_ids = await asyncio.gather(*tasks)

            # Verify all detectors were created
            count = await async_repository.count()
            assert count == 10

            # Verify all detectors can be found
            find_tasks = [
                async_repository.find_by_id(detector_id) for detector_id in detector_ids
            ]
            found_detectors = await asyncio.gather(*find_tasks)

            assert all(detector is not None for detector in found_detectors)
            assert len(set(d.id for d in found_detectors)) == 10

        # Run concurrent test
        asyncio.run(test_concurrent())


class TestRepositoryIntegration:
    """Integration tests for repository systems."""

    @pytest.fixture
    def repository_system(self):
        """Create complete repository system."""
        factory = RepositoryFactory()
        return {
            "detector": factory.create_detector_repository("memory"),
            "dataset": factory.create_dataset_repository("memory"),
            "result": factory.create_detection_result_repository("memory"),
        }

    def test_cross_repository_operations(self, repository_system):
        """Test operations across multiple repositories."""
        # Create dataset
        dataset = Dataset(
            id=uuid4(),
            name="integration-dataset",
            description="Integration test dataset",
            metadata={"rows": 1000},
            created_at=datetime.now(UTC),
        )
        repository_system["dataset"].save(dataset)

        # Create detector
        detector = Detector(
            id=uuid4(),
            name="integration-detector",
            algorithm_name="IsolationForest",
            hyperparameters={},
            created_at=datetime.now(UTC),
            is_fitted=True,
        )
        repository_system["detector"].save(detector)

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
        repository_system["result"].save(result)

        # Verify relationships
        found_result = repository_system["result"].find_by_id(result.id)
        assert found_result.detector_id == detector.id
        assert found_result.dataset_id == dataset.id

        # Test cascade operations
        detector_results = repository_system["result"].find_by_detector(detector.id)
        assert len(detector_results) == 1
        assert detector_results[0].id == result.id

        dataset_results = repository_system["result"].find_by_dataset(dataset.id)
        assert len(dataset_results) == 1
        assert dataset_results[0].id == result.id

    def test_repository_transaction_simulation(self, repository_system):
        """Test transaction-like behavior across repositories."""
        # Simulate a complex operation involving multiple repositories
        operations_log = []

        try:
            # Step 1: Create dataset
            dataset = Dataset(
                id=uuid4(),
                name="transaction-dataset",
                description="Transaction test",
                metadata={"test": True},
                created_at=datetime.now(UTC),
            )
            repository_system["dataset"].save(dataset)
            operations_log.append(("dataset", "create", dataset.id))

            # Step 2: Create detector
            detector = Detector(
                id=uuid4(),
                name="transaction-detector",
                algorithm_name="IsolationForest",
                hyperparameters={},
                created_at=datetime.now(UTC),
            )
            repository_system["detector"].save(detector)
            operations_log.append(("detector", "create", detector.id))

            # Step 3: Simulate failure and rollback
            if True:  # Simulate failure condition
                # Rollback operations
                for operation_type, action, entity_id in reversed(operations_log):
                    if operation_type == "dataset" and action == "create":
                        repository_system["dataset"].delete(entity_id)
                    elif operation_type == "detector" and action == "create":
                        repository_system["detector"].delete(entity_id)

                operations_log.clear()

            # Verify rollback
            assert repository_system["dataset"].count() == 0
            assert repository_system["detector"].count() == 0

        except Exception as e:
            # Cleanup on exception
            for operation_type, action, entity_id in reversed(operations_log):
                try:
                    if operation_type == "dataset":
                        repository_system["dataset"].delete(entity_id)
                    elif operation_type == "detector":
                        repository_system["detector"].delete(entity_id)
                except:
                    pass  # Ignore cleanup errors
            raise e
