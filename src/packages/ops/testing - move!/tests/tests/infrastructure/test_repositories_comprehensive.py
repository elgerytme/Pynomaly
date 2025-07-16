"""Comprehensive tests for infrastructure repositories - Phase 2 Coverage."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock, patch

import pytest

from monorepo.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from monorepo.domain.exceptions import EntityNotFoundError, RepositoryError
from monorepo.domain.value_objects import AnomalyScore, ContaminationRate
from monorepo.infrastructure.repositories import (
    DatabaseDatasetRepository,
    DatabaseDetectorRepository,
    DatabaseResultRepository,
    InMemoryDatasetRepository,
    InMemoryDetectorRepository,
    InMemoryResultRepository,
    RepositoryFactory,
)


@pytest.fixture
def sample_detector():
    """Create a sample detector for testing."""
    return Detector(
        name="test_detector",
        algorithm="isolation_forest",
        contamination=ContaminationRate(0.1),
        hyperparameters={"n_estimators": 100, "random_state": 42},
    )


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    import numpy as np

    features = np.random.RandomState(42).normal(0, 1, (100, 5))
    targets = np.random.RandomState(42).choice([0, 1], size=100, p=[0.9, 0.1])
    return Dataset(name="test_dataset", features=features, targets=targets)


@pytest.fixture
def sample_detection_result(sample_detector, sample_dataset):
    """Create a sample detection result for testing."""
    import numpy as np

    scores = np.random.RandomState(42).random(len(sample_dataset.features))
    anomalies = [
        Anomaly(score=AnomalyScore(0.95), index=0),
        Anomaly(score=AnomalyScore(0.87), index=10),
    ]
    return DetectionResult(
        detector=sample_detector,
        dataset=sample_dataset,
        anomalies=anomalies,
        scores=scores,
    )


@pytest.fixture
def multiple_detectors():
    """Create multiple detectors for testing."""
    detectors = []
    algorithms = ["isolation_forest", "local_outlier_factor", "one_class_svm"]
    contamination_rates = [0.05, 0.1, 0.15]

    for i, (algo, cont_rate) in enumerate(
        zip(algorithms, contamination_rates, strict=False)
    ):
        detector = Detector(
            name=f"detector_{i}",
            algorithm=algo,
            contamination=ContaminationRate(cont_rate),
            hyperparameters={"random_state": 42 + i},
        )
        detectors.append(detector)

    return detectors


class TestInMemoryDetectorRepository:
    """Comprehensive tests for InMemoryDetectorRepository."""

    @pytest.mark.asyncio
    async def test_detector_crud_operations(self, sample_detector):
        """Test basic CRUD operations for detectors."""
        repo = InMemoryDetectorRepository()

        # Test save
        saved_detector = await repo.save(sample_detector)
        assert saved_detector.id == sample_detector.id
        assert saved_detector.name == sample_detector.name

        # Test find by id
        found_detector = await repo.find_by_id(sample_detector.id)
        assert found_detector is not None
        assert found_detector.id == sample_detector.id
        assert found_detector.name == sample_detector.name

        # Test find all
        all_detectors = await repo.find_all()
        assert len(all_detectors) == 1
        assert all_detectors[0].id == sample_detector.id

        # Test update
        sample_detector.name = "updated_detector"
        updated_detector = await repo.update(sample_detector)
        assert updated_detector.name == "updated_detector"

        # Test delete
        await repo.delete(sample_detector.id)
        deleted_detector = await repo.find_by_id(sample_detector.id)
        assert deleted_detector is None

        # Test find all after delete
        all_detectors = await repo.find_all()
        assert len(all_detectors) == 0

    @pytest.mark.asyncio
    async def test_detector_find_by_name(self, multiple_detectors):
        """Test finding detectors by name."""
        repo = InMemoryDetectorRepository()

        # Save multiple detectors
        for detector in multiple_detectors:
            await repo.save(detector)

        # Test find by name
        found_detector = await repo.find_by_name("detector_1")
        assert found_detector is not None
        assert found_detector.name == "detector_1"
        assert found_detector.algorithm == "local_outlier_factor"

        # Test find non-existent name
        not_found = await repo.find_by_name("non_existent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_detector_find_by_algorithm(self, multiple_detectors):
        """Test finding detectors by algorithm."""
        repo = InMemoryDetectorRepository()

        # Save multiple detectors
        for detector in multiple_detectors:
            await repo.save(detector)

        # Test find by algorithm
        iso_forest_detectors = await repo.find_by_algorithm("isolation_forest")
        assert len(iso_forest_detectors) == 1
        assert iso_forest_detectors[0].algorithm == "isolation_forest"

        # Test find non-existent algorithm
        unknown_algo = await repo.find_by_algorithm("unknown_algorithm")
        assert len(unknown_algo) == 0

    @pytest.mark.asyncio
    async def test_detector_exists(self, sample_detector):
        """Test detector existence checking."""
        repo = InMemoryDetectorRepository()

        # Test non-existent detector
        exists_before = await repo.exists(sample_detector.id)
        assert exists_before is False

        # Save detector
        await repo.save(sample_detector)

        # Test existing detector
        exists_after = await repo.exists(sample_detector.id)
        assert exists_after is True

    @pytest.mark.asyncio
    async def test_detector_count(self, multiple_detectors):
        """Test detector counting."""
        repo = InMemoryDetectorRepository()

        # Test empty count
        count_empty = await repo.count()
        assert count_empty == 0

        # Save detectors one by one and check count
        for i, detector in enumerate(multiple_detectors):
            await repo.save(detector)
            count_current = await repo.count()
            assert count_current == i + 1

        # Test final count
        final_count = await repo.count()
        assert final_count == len(multiple_detectors)

    @pytest.mark.asyncio
    async def test_detector_pagination(self, multiple_detectors):
        """Test detector pagination."""
        repo = InMemoryDetectorRepository()

        # Save multiple detectors
        for detector in multiple_detectors:
            await repo.save(detector)

        # Test pagination
        page_1 = await repo.find_all(limit=2, offset=0)
        assert len(page_1) == 2

        page_2 = await repo.find_all(limit=2, offset=2)
        assert len(page_2) == 1  # Only 3 total detectors

        # Test limit only
        limited = await repo.find_all(limit=1)
        assert len(limited) == 1

    @pytest.mark.asyncio
    async def test_detector_error_handling(self):
        """Test error handling in detector repository."""
        repo = InMemoryDetectorRepository()

        # Test find non-existent detector
        with pytest.raises(EntityNotFoundError):
            await repo.find_by_id("non_existent_id")

        # Test update non-existent detector
        fake_detector = Detector(
            name="fake", algorithm="fake_algo", contamination=ContaminationRate(0.1)
        )
        fake_detector.id = "fake_id"

        with pytest.raises(EntityNotFoundError):
            await repo.update(fake_detector)

        # Test delete non-existent detector
        with pytest.raises(EntityNotFoundError):
            await repo.delete("non_existent_id")


class TestInMemoryDatasetRepository:
    """Comprehensive tests for InMemoryDatasetRepository."""

    @pytest.mark.asyncio
    async def test_dataset_crud_operations(self, sample_dataset):
        """Test basic CRUD operations for datasets."""
        repo = InMemoryDatasetRepository()

        # Test save
        saved_dataset = await repo.save(sample_dataset)
        assert saved_dataset.id == sample_dataset.id
        assert saved_dataset.name == sample_dataset.name

        # Test find by id
        found_dataset = await repo.find_by_id(sample_dataset.id)
        assert found_dataset is not None
        assert found_dataset.id == sample_dataset.id
        assert found_dataset.name == sample_dataset.name

        # Test update
        sample_dataset.name = "updated_dataset"
        updated_dataset = await repo.update(sample_dataset)
        assert updated_dataset.name == "updated_dataset"

        # Test delete
        await repo.delete(sample_dataset.id)
        deleted_dataset = await repo.find_by_id(sample_dataset.id)
        assert deleted_dataset is None

    @pytest.mark.asyncio
    async def test_dataset_find_by_size_range(self):
        """Test finding datasets by size range."""
        repo = InMemoryDatasetRepository()

        # Create datasets of different sizes
        import numpy as np

        small_dataset = Dataset(name="small", features=np.random.random((50, 3)))
        medium_dataset = Dataset(name="medium", features=np.random.random((200, 5)))
        large_dataset = Dataset(name="large", features=np.random.random((1000, 10)))

        await repo.save(small_dataset)
        await repo.save(medium_dataset)
        await repo.save(large_dataset)

        # Test size range queries
        small_datasets = await repo.find_by_size_range(min_samples=0, max_samples=100)
        assert len(small_datasets) == 1
        assert small_datasets[0].name == "small"

        medium_datasets = await repo.find_by_size_range(
            min_samples=100, max_samples=500
        )
        assert len(medium_datasets) == 1
        assert medium_datasets[0].name == "medium"

        large_datasets = await repo.find_by_size_range(
            min_samples=500, max_samples=2000
        )
        assert len(large_datasets) == 1
        assert large_datasets[0].name == "large"

    @pytest.mark.asyncio
    async def test_dataset_find_by_feature_count(self):
        """Test finding datasets by feature count."""
        repo = InMemoryDatasetRepository()

        # Create datasets with different feature counts
        import numpy as np

        datasets = [
            Dataset(name="low_dim", features=np.random.random((100, 2))),
            Dataset(name="med_dim", features=np.random.random((100, 5))),
            Dataset(name="high_dim", features=np.random.random((100, 20))),
        ]

        for dataset in datasets:
            await repo.save(dataset)

        # Test feature count queries
        low_dim = await repo.find_by_feature_count(min_features=1, max_features=3)
        assert len(low_dim) == 1
        assert low_dim[0].name == "low_dim"

        med_dim = await repo.find_by_feature_count(min_features=4, max_features=10)
        assert len(med_dim) == 1
        assert med_dim[0].name == "med_dim"

        high_dim = await repo.find_by_feature_count(min_features=15, max_features=25)
        assert len(high_dim) == 1
        assert high_dim[0].name == "high_dim"

    @pytest.mark.asyncio
    async def test_dataset_statistics(self, sample_dataset):
        """Test dataset statistics calculation."""
        repo = InMemoryDatasetRepository()
        await repo.save(sample_dataset)

        # Get repository statistics
        stats = await repo.get_statistics()

        assert stats["total_datasets"] == 1
        assert stats["total_samples"] == sample_dataset.n_samples
        assert stats["total_features"] == sample_dataset.n_features
        assert "avg_samples_per_dataset" in stats
        assert "avg_features_per_dataset" in stats


class TestInMemoryResultRepository:
    """Comprehensive tests for InMemoryResultRepository."""

    @pytest.mark.asyncio
    async def test_result_crud_operations(self, sample_detection_result):
        """Test basic CRUD operations for detection results."""
        repo = InMemoryResultRepository()

        # Test save
        saved_result = await repo.save(sample_detection_result)
        assert saved_result.id == sample_detection_result.id

        # Test find by id
        found_result = await repo.find_by_id(sample_detection_result.id)
        assert found_result is not None
        assert found_result.id == sample_detection_result.id

        # Test find all
        all_results = await repo.find_all()
        assert len(all_results) == 1

        # Test delete
        await repo.delete(sample_detection_result.id)
        deleted_result = await repo.find_by_id(sample_detection_result.id)
        assert deleted_result is None

    @pytest.mark.asyncio
    async def test_result_find_by_detector(self, sample_detection_result):
        """Test finding results by detector."""
        repo = InMemoryResultRepository()
        await repo.save(sample_detection_result)

        # Test find by detector id
        detector_results = await repo.find_by_detector_id(
            sample_detection_result.detector.id
        )
        assert len(detector_results) == 1
        assert detector_results[0].detector.id == sample_detection_result.detector.id

        # Test find by non-existent detector
        empty_results = await repo.find_by_detector_id("non_existent")
        assert len(empty_results) == 0

    @pytest.mark.asyncio
    async def test_result_find_by_dataset(self, sample_detection_result):
        """Test finding results by dataset."""
        repo = InMemoryResultRepository()
        await repo.save(sample_detection_result)

        # Test find by dataset id
        dataset_results = await repo.find_by_dataset_id(
            sample_detection_result.dataset.id
        )
        assert len(dataset_results) == 1
        assert dataset_results[0].dataset.id == sample_detection_result.dataset.id

        # Test find by non-existent dataset
        empty_results = await repo.find_by_dataset_id("non_existent")
        assert len(empty_results) == 0

    @pytest.mark.asyncio
    async def test_result_find_by_anomaly_count(self, sample_detection_result):
        """Test finding results by anomaly count range."""
        repo = InMemoryResultRepository()
        await repo.save(sample_detection_result)

        # Test find by anomaly count range
        results = await repo.find_by_anomaly_count_range(
            min_anomalies=1, max_anomalies=5
        )
        assert len(results) == 1

        # Test find outside range
        no_results = await repo.find_by_anomaly_count_range(
            min_anomalies=10, max_anomalies=20
        )
        assert len(no_results) == 0

    @pytest.mark.asyncio
    async def test_result_performance_metrics(self, sample_detection_result):
        """Test result performance metrics calculation."""
        repo = InMemoryResultRepository()
        await repo.save(sample_detection_result)

        # Get performance metrics
        metrics = await repo.get_performance_metrics()

        assert "total_results" in metrics
        assert "avg_anomalies_per_result" in metrics
        assert "avg_detection_time" in metrics
        assert metrics["total_results"] == 1


class TestDatabaseRepositories:
    """Comprehensive tests for database repositories."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        session.refresh = Mock()
        session.query = Mock()
        session.delete = Mock()
        session.close = Mock()
        return session

    @pytest.mark.asyncio
    async def test_database_detector_repository(self, sample_detector, mock_db_session):
        """Test database detector repository operations."""
        with patch(
            "monorepo.infrastructure.persistence.database.get_session"
        ) as mock_get_session:
            mock_get_session.return_value = mock_db_session

            # Mock query results
            mock_query = Mock()
            mock_query.filter.return_value.first.return_value = sample_detector
            mock_query.all.return_value = [sample_detector]
            mock_db_session.query.return_value = mock_query

            repo = DatabaseDetectorRepository()

            # Test save
            await repo.save(sample_detector)
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called()

            # Test find by id
            found_detector = await repo.find_by_id(sample_detector.id)
            assert found_detector == sample_detector

            # Test find all
            all_detectors = await repo.find_all()
            assert len(all_detectors) == 1

    @pytest.mark.asyncio
    async def test_database_connection_error_handling(self, sample_detector):
        """Test database connection error handling."""
        with patch(
            "monorepo.infrastructure.persistence.database.get_session"
        ) as mock_get_session:
            # Mock connection error
            mock_get_session.side_effect = Exception("Database connection failed")

            repo = DatabaseDetectorRepository()

            with pytest.raises(RepositoryError, match="Database connection failed"):
                await repo.save(sample_detector)

    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self, sample_detector):
        """Test database transaction rollback on error."""
        mock_session = Mock()
        mock_session.add = Mock()
        mock_session.commit = Mock(side_effect=Exception("Commit failed"))
        mock_session.rollback = Mock()
        mock_session.close = Mock()

        with patch(
            "monorepo.infrastructure.persistence.database.get_session"
        ) as mock_get_session:
            mock_get_session.return_value = mock_session

            repo = DatabaseDetectorRepository()

            with pytest.raises(RepositoryError):
                await repo.save(sample_detector)

            # Verify rollback was called
            mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_query_optimization(self, multiple_detectors):
        """Test database query optimization features."""
        mock_session = Mock()
        mock_query = Mock()

        # Mock query builder methods
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = multiple_detectors[:2]

        mock_session.query.return_value = mock_query

        with patch(
            "monorepo.infrastructure.persistence.database.get_session"
        ) as mock_get_session:
            mock_get_session.return_value = mock_session

            repo = DatabaseDetectorRepository()

            # Test optimized query with pagination
            results = await repo.find_all(limit=2, offset=0, order_by="name")

            # Verify query optimization methods were called
            mock_query.order_by.assert_called()
            mock_query.limit.assert_called_with(2)
            mock_query.offset.assert_called_with(0)

            assert len(results) == 2


class TestRepositoryFactory:
    """Test repository factory functionality."""

    def test_factory_in_memory_creation(self):
        """Test factory creation of in-memory repositories."""
        # Test detector repository
        detector_repo = RepositoryFactory.create_detector_repository("memory")
        assert isinstance(detector_repo, InMemoryDetectorRepository)

        # Test dataset repository
        dataset_repo = RepositoryFactory.create_dataset_repository("memory")
        assert isinstance(dataset_repo, InMemoryDatasetRepository)

        # Test result repository
        result_repo = RepositoryFactory.create_result_repository("memory")
        assert isinstance(result_repo, InMemoryResultRepository)

    def test_factory_database_creation(self):
        """Test factory creation of database repositories."""
        with patch("monorepo.infrastructure.persistence.database.get_session"):
            # Test detector repository
            detector_repo = RepositoryFactory.create_detector_repository("database")
            assert isinstance(detector_repo, DatabaseDetectorRepository)

            # Test dataset repository
            dataset_repo = RepositoryFactory.create_dataset_repository("database")
            assert isinstance(dataset_repo, DatabaseDatasetRepository)

            # Test result repository
            result_repo = RepositoryFactory.create_result_repository("database")
            assert isinstance(result_repo, DatabaseResultRepository)

    def test_factory_configuration(self):
        """Test factory configuration options."""
        # Test with configuration
        config = {
            "database_url": "sqlite:///test.db",
            "connection_pool_size": 10,
            "echo": True,
        }

        with patch("monorepo.infrastructure.persistence.database.get_session"):
            repo = RepositoryFactory.create_detector_repository("database", **config)
            assert isinstance(repo, DatabaseDetectorRepository)

    def test_factory_unsupported_type(self):
        """Test factory with unsupported repository type."""
        with pytest.raises(ValueError, match="Unsupported repository type"):
            RepositoryFactory.create_detector_repository("unsupported")

    def test_factory_auto_detection(self):
        """Test automatic repository type detection."""
        # Test auto detection based on configuration
        auto_memory = RepositoryFactory.create_detector_repository(
            "auto", prefer_memory=True
        )
        assert isinstance(auto_memory, InMemoryDetectorRepository)

        with patch("monorepo.infrastructure.persistence.database.get_session"):
            auto_db = RepositoryFactory.create_detector_repository(
                "auto", prefer_memory=False
            )
            assert isinstance(auto_db, DatabaseDetectorRepository)


class TestRepositoryPerformance:
    """Test repository performance characteristics."""

    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, multiple_detectors):
        """Test bulk operations performance."""
        repo = InMemoryDetectorRepository()

        # Test bulk save
        import time

        start_time = time.time()

        saved_detectors = await repo.bulk_save(multiple_detectors)

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify bulk operation
        assert len(saved_detectors) == len(multiple_detectors)
        assert execution_time < 1.0  # Should be fast for in-memory

        # Test bulk delete
        detector_ids = [d.id for d in saved_detectors]
        start_time = time.time()

        await repo.bulk_delete(detector_ids)

        end_time = time.time()
        delete_time = end_time - start_time

        # Verify deletion
        remaining = await repo.find_all()
        assert len(remaining) == 0
        assert delete_time < 1.0

    @pytest.mark.asyncio
    async def test_concurrent_access(self, sample_detector):
        """Test concurrent repository access."""
        repo = InMemoryDetectorRepository()

        # Create multiple concurrent operations
        async def save_detector(detector_id: str):
            detector = Detector(
                name=f"detector_{detector_id}",
                algorithm="isolation_forest",
                contamination=ContaminationRate(0.1),
            )
            return await repo.save(detector)

        # Execute concurrent saves
        tasks = [save_detector(str(i)) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all saves succeeded
        assert len(results) == 10
        assert all(result is not None for result in results)

        # Verify repository state
        all_detectors = await repo.find_all()
        assert len(all_detectors) == 10

    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self):
        """Test memory usage optimization in repositories."""
        repo = InMemoryDetectorRepository()

        # Create many detectors to test memory management
        import numpy as np

        large_detectors = []
        for i in range(100):
            detector = Detector(
                name=f"large_detector_{i}",
                algorithm="isolation_forest",
                contamination=ContaminationRate(0.1),
                hyperparameters={
                    "n_estimators": 100,
                    "random_state": i,
                    "large_data": np.random.random(1000).tolist(),  # Large data
                },
            )
            large_detectors.append(detector)

        # Save all detectors
        for detector in large_detectors:
            await repo.save(detector)

        # Test memory optimization methods
        memory_stats = await repo.get_memory_usage_stats()
        assert "total_memory_mb" in memory_stats
        assert "detector_count" in memory_stats

        # Test cleanup
        await repo.cleanup_expired_entries(max_age_hours=0)  # Clean all
        remaining = await repo.find_all()
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_query_performance_optimization(self, multiple_detectors):
        """Test query performance optimization."""
        repo = InMemoryDetectorRepository()

        # Save many detectors
        extended_detectors = multiple_detectors * 100  # 300 detectors
        for detector in extended_detectors:
            await repo.save(detector)

        # Test index-based queries
        import time

        # Test by ID (should be fast)
        start_time = time.time()
        found = await repo.find_by_id(extended_detectors[0].id)
        id_query_time = time.time() - start_time

        assert found is not None
        assert id_query_time < 0.01  # Should be very fast

        # Test by algorithm (should use index)
        start_time = time.time()
        algo_results = await repo.find_by_algorithm("isolation_forest")
        algo_query_time = time.time() - start_time

        assert len(algo_results) > 0
        assert algo_query_time < 0.1  # Should be reasonably fast


class TestRepositoryIntegration:
    """Test repository integration and cross-repository operations."""

    @pytest.mark.asyncio
    async def test_cross_repository_consistency(
        self, sample_detector, sample_dataset, sample_detection_result
    ):
        """Test consistency across multiple repositories."""
        detector_repo = InMemoryDetectorRepository()
        dataset_repo = InMemoryDatasetRepository()
        result_repo = InMemoryResultRepository()

        # Save entities in order
        saved_detector = await detector_repo.save(sample_detector)
        saved_dataset = await dataset_repo.save(sample_dataset)

        # Update result with saved entities
        sample_detection_result.detector = saved_detector
        sample_detection_result.dataset = saved_dataset
        saved_result = await result_repo.save(sample_detection_result)

        # Verify relationships
        assert saved_result.detector.id == saved_detector.id
        assert saved_result.dataset.id == saved_dataset.id

        # Test cascading operations
        detector_results = await result_repo.find_by_detector_id(saved_detector.id)
        assert len(detector_results) == 1
        assert detector_results[0].id == saved_result.id

    @pytest.mark.asyncio
    async def test_repository_transaction_coordination(
        self, sample_detector, sample_dataset
    ):
        """Test transaction coordination across repositories."""
        detector_repo = InMemoryDetectorRepository()
        dataset_repo = InMemoryDatasetRepository()

        # Simulate transaction across repositories
        async def save_both_entities():
            try:
                saved_detector = await detector_repo.save(sample_detector)
                saved_dataset = await dataset_repo.save(sample_dataset)
                return saved_detector, saved_dataset
            except Exception as e:
                # Rollback both if either fails
                await detector_repo.delete(sample_detector.id)
                await dataset_repo.delete(sample_dataset.id)
                raise e

        # Execute transaction
        detector, dataset = await save_both_entities()

        # Verify both were saved
        assert await detector_repo.exists(detector.id)
        assert await dataset_repo.exists(dataset.id)

    @pytest.mark.asyncio
    async def test_repository_data_migration(self, multiple_detectors):
        """Test data migration between repository types."""
        # Setup source and target repositories
        source_repo = InMemoryDetectorRepository()
        target_repo = (
            InMemoryDetectorRepository()
        )  # In practice, this could be DatabaseDetectorRepository

        # Save data to source
        for detector in multiple_detectors:
            await source_repo.save(detector)

        # Migrate data
        async def migrate_detectors():
            all_detectors = await source_repo.find_all()
            migrated_detectors = []

            for detector in all_detectors:
                migrated = await target_repo.save(detector)
                migrated_detectors.append(migrated)

            return migrated_detectors

        migrated = await migrate_detectors()

        # Verify migration
        assert len(migrated) == len(multiple_detectors)

        target_count = await target_repo.count()
        assert target_count == len(multiple_detectors)

        # Verify data integrity
        for original in multiple_detectors:
            found = await target_repo.find_by_id(original.id)
            assert found is not None
            assert found.name == original.name
            assert found.algorithm == original.algorithm


class TestRepositoryErrorRecovery:
    """Test repository error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_connection_recovery(self, sample_detector):
        """Test automatic connection recovery."""
        mock_session = Mock()

        # Simulate connection failure then recovery
        connection_attempts = [0]

        def mock_get_session():
            connection_attempts[0] += 1
            if connection_attempts[0] == 1:
                raise Exception("Connection failed")
            else:
                return mock_session

        with patch(
            "monorepo.infrastructure.persistence.database.get_session", mock_get_session
        ):
            repo = DatabaseDetectorRepository(retry_attempts=2, retry_delay=0.1)

            # Should recover after first failure
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.refresh = Mock()

            await repo.save(sample_detector)

            # Verify retry occurred
            assert connection_attempts[0] == 2
            mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_consistency_recovery(self, sample_detector):
        """Test data consistency recovery after failures."""
        repo = InMemoryDetectorRepository()

        # Simulate partial failure scenario
        original_save = repo.save

        async def failing_save(detector):
            if detector.name == "fail_detector":
                raise Exception("Simulated failure")
            return await original_save(detector)

        repo.save = failing_save

        # Test recovery mechanism
        detectors = [
            sample_detector,
            Detector(
                name="fail_detector",
                algorithm="test",
                contamination=ContaminationRate(0.1),
            ),
            Detector(
                name="success_detector",
                algorithm="test",
                contamination=ContaminationRate(0.1),
            ),
        ]

        successful_saves = []
        failed_saves = []

        for detector in detectors:
            try:
                saved = await repo.save(detector)
                successful_saves.append(saved)
            except Exception:
                failed_saves.append(detector)

        # Verify partial success
        assert len(successful_saves) == 2  # sample_detector and success_detector
        assert len(failed_saves) == 1  # fail_detector

        # Verify repository state consistency
        count = await repo.count()
        assert count == len(successful_saves)

    @pytest.mark.asyncio
    async def test_concurrent_access_conflict_resolution(self, sample_detector):
        """Test conflict resolution for concurrent access."""
        repo = InMemoryDetectorRepository()

        # Save initial detector
        await repo.save(sample_detector)

        # Simulate concurrent updates
        async def update_detector(new_name: str, delay: float = 0):
            await asyncio.sleep(delay)
            detector_copy = await repo.find_by_id(sample_detector.id)
            detector_copy.name = new_name
            return await repo.update(detector_copy)

        # Execute concurrent updates
        tasks = [
            update_detector("name_1", 0.01),
            update_detector("name_2", 0.02),
            update_detector("name_3", 0.03),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify at least one update succeeded
        successful_updates = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_updates) >= 1

        # Verify final state is consistent
        final_detector = await repo.find_by_id(sample_detector.id)
        assert final_detector.name in ["name_1", "name_2", "name_3"]
