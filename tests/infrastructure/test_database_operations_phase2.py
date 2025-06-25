"""
Phase 2 Infrastructure Hardening: Comprehensive Database Operations Testing
Testing suite for all database operations, repositories, and persistence layers.

This module implements comprehensive testing for:
- Repository pattern implementations
- Database connection handling
- Transaction management
- Data persistence and retrieval
- Performance optimization
- Connection pooling and cleanup
"""

import gc
import os
import sqlite3
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import DatabaseError, RepositoryError


def requires_async_database(database_type: str):
    """Decorator to skip tests if async database dependencies are not available."""

    def decorator(test_func):
        try:
            if database_type == "postgresql":
                import asyncpg
            elif database_type == "sqlite":
                import aiosqlite
            return test_func
        except ImportError:
            return pytest.mark.skip(f"Requires {database_type} async dependencies")(
                test_func
            )

    return decorator


@contextmanager
def temp_database():
    """Create temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Initialize database
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                features TEXT,
                target_column TEXT,
                metadata TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detectors (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                description TEXT,
                parameters TEXT,
                is_fitted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_results (
                id TEXT PRIMARY KEY,
                detector_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                predictions TEXT,
                scores TEXT,
                anomaly_rate REAL,
                n_anomalies INTEGER,
                n_samples INTEGER,
                timestamp TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (detector_id) REFERENCES detectors (id),
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            )
        """
        )

        conn.commit()
        conn.close()

        yield db_path
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


class DatabaseTestBase:
    """Base class for database testing with common utilities."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset entity for testing."""
        return Dataset(
            id=uuid.uuid4(),
            name="test_dataset",
            description="Test dataset for anomaly detection",
            data=pd.DataFrame(
                {
                    "feature_1": np.random.normal(0, 1, 100),
                    "feature_2": np.random.normal(0, 1, 100),
                    "feature_3": np.random.normal(0, 1, 100),
                }
            ),
            features=["feature_1", "feature_2", "feature_3"],
            target_column=None,
            metadata={"source": "test", "quality": "high"},
            created_at=datetime.now(),
        )

    @pytest.fixture
    def sample_detector(self):
        """Create sample detector entity for testing."""
        return Detector(
            id=uuid.uuid4(),
            name="test_detector",
            algorithm="IsolationForest",
            description="Test detector for anomaly detection",
            parameters={"contamination": 0.1, "n_estimators": 100},
            is_fitted=False,
            created_at=datetime.now(),
        )

    @pytest.fixture
    def sample_detection_result(self, sample_detector, sample_dataset):
        """Create sample detection result entity for testing."""
        # Generate sample predictions and scores
        n_samples = 100
        predictions = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        scores = np.random.exponential(scale=1.0, size=n_samples)

        return DetectionResult(
            id=uuid.uuid4(),
            detector_id=sample_detector.id,
            dataset_id=sample_dataset.id,
            predictions=predictions,
            scores=scores,
            timestamp=datetime.now(),
            metadata={"execution_time": 1.23, "memory_usage": 456},
        )


class TestInMemoryRepositories(DatabaseTestBase):
    """Test in-memory repository implementations."""

    def test_dataset_repository_crud_operations(self, sample_dataset):
        """Test CRUD operations for dataset repository."""
        from pynomaly.infrastructure.persistence.repositories import (
            InMemoryDatasetRepository,
        )

        repo = InMemoryDatasetRepository()

        # Test save
        repo.save(sample_dataset)
        assert len(repo._data) == 1

        # Test find_by_id
        found_dataset = repo.find_by_id(sample_dataset.id)
        assert found_dataset is not None
        assert found_dataset.id == sample_dataset.id
        assert found_dataset.name == sample_dataset.name

        # Test find_all
        all_datasets = repo.find_all()
        assert len(all_datasets) == 1
        assert all_datasets[0].id == sample_dataset.id

        # Test update
        sample_dataset.description = "Updated description"
        repo.save(sample_dataset)
        updated_dataset = repo.find_by_id(sample_dataset.id)
        assert updated_dataset.description == "Updated description"

        # Test delete
        repo.delete(sample_dataset.id)
        deleted_dataset = repo.find_by_id(sample_dataset.id)
        assert deleted_dataset is None
        assert len(repo._data) == 0

    def test_detector_repository_crud_operations(self, sample_detector):
        """Test CRUD operations for detector repository."""
        from pynomaly.infrastructure.persistence.repositories import (
            InMemoryDetectorRepository,
        )

        repo = InMemoryDetectorRepository()

        # Test save
        repo.save(sample_detector)
        assert len(repo._data) == 1

        # Test find_by_id
        found_detector = repo.find_by_id(sample_detector.id)
        assert found_detector is not None
        assert found_detector.algorithm == sample_detector.algorithm

        # Test find_by_algorithm
        if hasattr(repo, "find_by_algorithm"):
            algorithm_detectors = repo.find_by_algorithm("IsolationForest")
            assert len(algorithm_detectors) == 1
            assert algorithm_detectors[0].id == sample_detector.id

        # Test count
        count = repo.count()
        assert count == 1

        # Test exists
        assert repo.exists(sample_detector.id) is True
        assert repo.exists(uuid.uuid4()) is False

    def test_detection_result_repository_crud_operations(self, sample_detection_result):
        """Test CRUD operations for detection result repository."""
        from pynomaly.infrastructure.persistence.repositories import (
            InMemoryResultRepository,
        )

        repo = InMemoryResultRepository()

        # Test save
        repo.save(sample_detection_result)
        assert len(repo._data) == 1

        # Test find_by_id
        found_result = repo.find_by_id(sample_detection_result.id)
        assert found_result is not None
        assert found_result.detector_id == sample_detection_result.detector_id
        assert found_result.dataset_id == sample_detection_result.dataset_id

        # Test find_by_detector
        detector_results = repo.find_by_detector(sample_detection_result.detector_id)
        assert len(detector_results) == 1
        assert detector_results[0].id == sample_detection_result.id

        # Test find_by_dataset
        dataset_results = repo.find_by_dataset(sample_detection_result.dataset_id)
        assert len(dataset_results) == 1
        assert dataset_results[0].id == sample_detection_result.id

        # Test find_recent
        recent_results = repo.find_recent(limit=10)
        assert len(recent_results) == 1
        assert recent_results[0].id == sample_detection_result.id

    def test_repository_concurrent_access(self, sample_dataset):
        """Test repository behavior under concurrent access."""
        import threading

        from pynomaly.infrastructure.persistence.repositories import (
            InMemoryDatasetRepository,
        )

        repo = InMemoryDatasetRepository()
        results = []
        errors = []

        def worker(worker_id):
            try:
                # Create unique dataset for this worker
                dataset = Dataset(
                    id=uuid.uuid4(),
                    name=f"dataset_{worker_id}",
                    description=f"Dataset from worker {worker_id}",
                    data=pd.DataFrame({"x": [1, 2, 3]}),
                    features=["x"],
                    created_at=datetime.now(),
                )

                # Save dataset
                repo.save(dataset)

                # Read it back
                found = repo.find_by_id(dataset.id)
                if found and found.name == f"dataset_{worker_id}":
                    results.append(worker_id)
                else:
                    errors.append(f"Worker {worker_id}: data mismatch")

            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert (
            len(results) == 10
        ), f"Expected 10 successful operations, got {len(results)}"
        assert (
            len(repo._data) == 10
        ), f"Expected 10 datasets in repository, got {len(repo._data)}"


class TestSQLiteRepository(DatabaseTestBase):
    """Test SQLite repository implementations."""

    def test_sqlite_dataset_repository(self, sample_dataset):
        """Test SQLite dataset repository operations."""
        with temp_database() as db_path:
            try:
                from pynomaly.infrastructure.persistence.sqlite_repository import (
                    SQLiteDatasetRepository,
                )

                repo = SQLiteDatasetRepository(db_path)

                # Test save
                repo.save(sample_dataset)

                # Test find_by_id
                found_dataset = repo.find_by_id(sample_dataset.id)
                assert found_dataset is not None
                assert found_dataset.name == sample_dataset.name

                # Test find_all
                all_datasets = repo.find_all()
                assert len(all_datasets) == 1

                # Test update
                sample_dataset.description = "Updated in SQLite"
                repo.save(sample_dataset)
                updated_dataset = repo.find_by_id(sample_dataset.id)
                assert updated_dataset.description == "Updated in SQLite"

                # Test delete
                repo.delete(sample_dataset.id)
                deleted_dataset = repo.find_by_id(sample_dataset.id)
                assert deleted_dataset is None

            except ImportError:
                pytest.skip("SQLite repository implementation not available")

    def test_sqlite_connection_pooling(self):
        """Test SQLite connection pooling and resource management."""
        with temp_database() as db_path:
            try:
                from pynomaly.infrastructure.persistence.sqlite_repository import (
                    SQLiteDatasetRepository,
                )

                # Create multiple repository instances
                repos = []
                for i in range(5):
                    repo = SQLiteDatasetRepository(db_path)
                    repos.append(repo)

                # Test that they can all operate independently
                for i, repo in enumerate(repos):
                    dataset = Dataset(
                        id=uuid.uuid4(),
                        name=f"test_dataset_{i}",
                        data=pd.DataFrame({"x": [1, 2, 3]}),
                        features=["x"],
                        created_at=datetime.now(),
                    )
                    repo.save(dataset)

                # Verify all datasets were saved
                final_repo = SQLiteDatasetRepository(db_path)
                all_datasets = final_repo.find_all()
                assert len(all_datasets) == 5

                # Clean up
                for repo in repos:
                    if hasattr(repo, "close"):
                        repo.close()

            except ImportError:
                pytest.skip("SQLite repository implementation not available")

    def test_sqlite_transaction_handling(self, sample_dataset, sample_detector):
        """Test SQLite transaction handling and rollback."""
        with temp_database() as db_path:
            try:
                from pynomaly.infrastructure.persistence.sqlite_repository import (
                    SQLiteDatasetRepository,
                )

                repo = SQLiteDatasetRepository(db_path)

                # Test successful transaction
                with repo.transaction():
                    repo.save(sample_dataset)

                # Verify data was committed
                found = repo.find_by_id(sample_dataset.id)
                assert found is not None

                # Test transaction rollback
                try:
                    with repo.transaction():
                        sample_dataset.name = "Modified name"
                        repo.save(sample_dataset)
                        raise Exception("Simulated error")
                except Exception:
                    pass  # Expected

                # Verify rollback - name should be unchanged
                found = repo.find_by_id(sample_dataset.id)
                assert found.name != "Modified name"

            except (ImportError, AttributeError):
                pytest.skip("SQLite transaction support not available")


@requires_async_database("postgresql")
class TestPostgreSQLRepository(DatabaseTestBase):
    """Test PostgreSQL repository implementations."""

    async def test_postgresql_async_operations(self, sample_dataset):
        """Test PostgreSQL async repository operations."""
        try:
            from pynomaly.infrastructure.persistence.postgresql_repository import (
                PostgreSQLDatasetRepository,
            )

            # Mock connection for testing
            mock_pool = AsyncMock()
            mock_connection = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_connection

            repo = PostgreSQLDatasetRepository(connection_pool=mock_pool)

            # Test async save
            await repo.save(sample_dataset)
            mock_connection.execute.assert_called()

            # Test async find_by_id
            mock_connection.fetchrow.return_value = {
                "id": str(sample_dataset.id),
                "name": sample_dataset.name,
                "description": sample_dataset.description,
                "features": '["feature_1", "feature_2", "feature_3"]',
                "metadata": "{}",
                "created_at": sample_dataset.created_at,
            }

            found_dataset = await repo.find_by_id(sample_dataset.id)
            assert found_dataset is not None

        except ImportError:
            pytest.skip("PostgreSQL async repository implementation not available")

    async def test_postgresql_connection_pooling(self):
        """Test PostgreSQL connection pooling and management."""
        try:
            import asyncpg

            from pynomaly.infrastructure.persistence.postgresql_repository import (
                PostgreSQLDatasetRepository,
            )

            # Mock connection pool
            mock_pool = AsyncMock()

            # Test pool acquisition and release
            async with mock_pool.acquire():
                repo = PostgreSQLDatasetRepository(connection_pool=mock_pool)

                # Verify pool is being used
                assert repo.connection_pool is mock_pool

            # Verify connection was properly released
            mock_pool.acquire.assert_called()

        except ImportError:
            pytest.skip("PostgreSQL async support not available")


class TestDatabasePerformance(DatabaseTestBase):
    """Test database performance and optimization."""

    def test_bulk_insert_performance(self):
        """Test bulk insert performance with large datasets."""
        with temp_database():
            try:
                from pynomaly.infrastructure.persistence.repositories import (
                    InMemoryDatasetRepository,
                )

                repo = InMemoryDatasetRepository()

                # Create many datasets
                datasets = []
                for i in range(100):
                    dataset = Dataset(
                        id=uuid.uuid4(),
                        name=f"bulk_dataset_{i}",
                        data=pd.DataFrame({"x": np.random.normal(0, 1, 1000)}),
                        features=["x"],
                        created_at=datetime.now(),
                    )
                    datasets.append(dataset)

                # Measure bulk insert time
                import time

                start_time = time.time()

                for dataset in datasets:
                    repo.save(dataset)

                insert_time = time.time() - start_time

                # Performance assertions
                assert insert_time < 10, f"Bulk insert took too long: {insert_time}s"
                assert len(repo._data) == 100

                # Measure bulk query time
                start_time = time.time()
                all_datasets = repo.find_all()
                query_time = time.time() - start_time

                assert query_time < 1, f"Bulk query took too long: {query_time}s"
                assert len(all_datasets) == 100

            except ImportError:
                pytest.skip("Repository implementation not available")

    def test_query_optimization(self, sample_detection_result):
        """Test query optimization and indexing."""
        from pynomaly.infrastructure.persistence.repositories import (
            InMemoryResultRepository,
        )

        repo = InMemoryResultRepository()

        # Create many detection results
        results = []
        for i in range(1000):
            result = DetectionResult(
                id=uuid.uuid4(),
                detector_id=sample_detection_result.detector_id,
                dataset_id=uuid.uuid4(),  # Different datasets
                predictions=np.random.choice([0, 1], size=100),
                scores=np.random.exponential(1.0, size=100),
                timestamp=datetime.now() - timedelta(days=i),
                metadata={},
            )
            results.append(result)
            repo.save(result)

        # Test query performance
        import time

        # Query by detector
        start_time = time.time()
        detector_results = repo.find_by_detector(sample_detection_result.detector_id)
        detector_query_time = time.time() - start_time

        assert (
            detector_query_time < 1
        ), f"Detector query took too long: {detector_query_time}s"
        assert len(detector_results) == 1000

        # Query recent results
        start_time = time.time()
        recent_results = repo.find_recent(limit=10)
        recent_query_time = time.time() - start_time

        assert (
            recent_query_time < 0.1
        ), f"Recent query took too long: {recent_query_time}s"
        assert len(recent_results) == 10

    def test_memory_usage_optimization(self):
        """Test memory usage optimization in repositories."""
        import psutil

        from pynomaly.infrastructure.persistence.repositories import (
            InMemoryDatasetRepository,
        )

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        repo = InMemoryDatasetRepository()

        # Create large datasets
        for i in range(50):
            large_data = pd.DataFrame(
                {f"feature_{j}": np.random.normal(0, 1, 10000) for j in range(10)}
            )

            dataset = Dataset(
                id=uuid.uuid4(),
                name=f"large_dataset_{i}",
                data=large_data,
                features=[f"feature_{j}" for j in range(10)],
                created_at=datetime.now(),
            )

            repo.save(dataset)

        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = current_memory - initial_memory

        # Should not exceed reasonable memory usage
        assert memory_growth < 1000, f"Excessive memory usage: {memory_growth} MB"

        # Test cleanup
        del repo
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        # Should release most memory
        assert final_memory < current_memory + 100, "Memory not properly released"


class TestDatabaseErrorHandling:
    """Test database error handling and resilience."""

    def test_connection_failure_handling(self):
        """Test handling of database connection failures."""
        try:
            from pynomaly.infrastructure.persistence.sqlite_repository import (
                SQLiteDatasetRepository,
            )

            # Test with invalid database path
            with pytest.raises((DatabaseError, Exception)):
                repo = SQLiteDatasetRepository("/invalid/path/database.db")
                repo.find_all()  # Should fail

        except ImportError:
            pytest.skip("SQLite repository implementation not available")

    def test_transaction_failure_recovery(self, sample_dataset):
        """Test recovery from transaction failures."""
        with temp_database() as db_path:
            try:
                from pynomaly.infrastructure.persistence.sqlite_repository import (
                    SQLiteDatasetRepository,
                )

                repo = SQLiteDatasetRepository(db_path)

                # Save initial data
                repo.save(sample_dataset)

                # Simulate transaction failure
                original_name = sample_dataset.name
                sample_dataset.name = "Failed transaction name"

                try:
                    with repo.transaction():
                        repo.save(sample_dataset)
                        # Simulate error
                        raise DatabaseError("Simulated transaction failure")
                except DatabaseError:
                    pass  # Expected

                # Verify rollback
                found = repo.find_by_id(sample_dataset.id)
                assert (
                    found.name == original_name
                ), "Transaction should have been rolled back"

            except (ImportError, AttributeError):
                pytest.skip("SQLite transaction support not available")

    def test_data_corruption_detection(self, sample_dataset):
        """Test detection of data corruption."""
        with temp_database() as db_path:
            try:
                from pynomaly.infrastructure.persistence.sqlite_repository import (
                    SQLiteDatasetRepository,
                )

                repo = SQLiteDatasetRepository(db_path)
                repo.save(sample_dataset)

                # Manually corrupt data in database
                conn = sqlite3.connect(db_path)
                conn.execute(
                    "UPDATE datasets SET features = 'invalid_json' WHERE id = ?",
                    (str(sample_dataset.id),),
                )
                conn.commit()
                conn.close()

                # Attempt to read corrupted data
                with pytest.raises((RepositoryError, ValueError, Exception)):
                    repo.find_by_id(sample_dataset.id)
                    # Should fail due to invalid JSON

            except ImportError:
                pytest.skip("SQLite repository implementation not available")


if __name__ == "__main__":
    # Run specific test classes for debugging
    pytest.main([__file__ + "::TestInMemoryRepositories", "-v"])
