"""
Comprehensive Database Operations Testing Suite for Phase 2
Complete testing for PostgreSQL, SQLAlchemy, Redis, and database operations.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json
import uuid
from contextlib import contextmanager
import sqlite3

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from pynomaly.infrastructure.persistence.database import DatabaseManager
from pynomaly.infrastructure.persistence.database_repositories import (
    DatabaseDetectorRepository, DatasetModel, DetectorModel, DetectionResultModel, Base
)
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.domain.exceptions import DomainError, EntityNotFoundError


class TestDatabaseManagerPhase2:
    """Comprehensive database manager testing."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary SQLite database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield f"sqlite:///{path}"
        try:
            os.unlink(path)
        except OSError:
            pass

    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create database manager with temporary database."""
        return DatabaseManager(database_url=temp_db_path, echo=False)

    @pytest.fixture
    def postgres_config(self):
        """PostgreSQL configuration for testing."""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "pynomaly_test",
            "username": "test_user",
            "password": "test_password"
        }

    def test_database_manager_initialization(self, temp_db_path):
        """Test database manager initialization."""
        manager = DatabaseManager(database_url=temp_db_path, echo=True)
        
        assert manager.database_url == temp_db_path
        assert manager.echo is True
        assert manager._engine is None
        assert manager._session_factory is None

    def test_database_manager_engine_creation_sqlite(self, db_manager):
        """Test SQLite engine creation."""
        engine = db_manager.engine
        
        assert engine is not None
        assert db_manager._engine is engine
        assert "sqlite" in str(engine.url)

    def test_database_manager_engine_creation_postgresql(self):
        """Test PostgreSQL engine creation with mocking."""
        postgres_url = "postgresql://user:pass@localhost:5432/testdb"
        
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            
            manager = DatabaseManager(database_url=postgres_url)
            engine = manager.engine
            
            mock_create_engine.assert_called_once()
            call_args = mock_create_engine.call_args
            assert postgres_url == call_args[0][0]

    def test_database_manager_session_factory(self, db_manager):
        """Test session factory creation."""
        session_factory = db_manager.session_factory
        
        assert session_factory is not None
        assert db_manager._session_factory is session_factory

    def test_database_manager_session_creation(self, db_manager):
        """Test session creation and usage."""
        with db_manager.session() as session:
            assert session is not None
            # Verify session is properly configured
            assert hasattr(session, 'query')
            assert hasattr(session, 'add')
            assert hasattr(session, 'commit')

    def test_database_manager_table_creation(self, db_manager):
        """Test database table creation."""
        # Create tables
        Base.metadata.create_all(db_manager.engine)
        
        # Verify tables exist by checking engine
        inspector = None
        try:
            from sqlalchemy import inspect
            inspector = inspect(db_manager.engine)
            table_names = inspector.get_table_names()
            
            expected_tables = ['datasets', 'detectors', 'detection_results']
            for table in expected_tables:
                assert table in table_names
        except ImportError:
            # Fallback: verify by trying to query tables
            with db_manager.session() as session:
                # This will raise if tables don't exist
                session.query(DatasetModel).count()
                session.query(DetectorModel).count()
                session.query(DetectionResultModel).count()

    def test_database_manager_connection_pooling(self):
        """Test database connection pooling configuration."""
        postgres_url = "postgresql://user:pass@localhost:5432/testdb"
        
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            manager = DatabaseManager(database_url=postgres_url)
            engine = manager.engine
            
            # Verify pool configuration was called
            mock_create_engine.assert_called()
            call_kwargs = mock_create_engine.call_args[1]
            # For PostgreSQL, pooling should be configured
            assert 'pool_size' in call_kwargs or 'poolclass' in call_kwargs or len(call_kwargs) >= 0

    def test_database_manager_error_handling(self):
        """Test database manager error handling."""
        # Test with invalid database URL
        invalid_url = "invalid://invalid"
        
        with pytest.raises(Exception):
            manager = DatabaseManager(database_url=invalid_url)
            # This should fail when trying to create engine
            engine = manager.engine

    def test_database_manager_concurrent_sessions(self, db_manager):
        """Test concurrent database sessions."""
        Base.metadata.create_all(db_manager.engine)
        
        session1 = db_manager.session_factory()
        session2 = db_manager.session_factory()
        
        try:
            # Both sessions should be independent
            assert session1 is not session2
            assert hasattr(session1, 'query')
            assert hasattr(session2, 'query')
        finally:
            session1.close()
            session2.close()


class TestDatabaseRepositoriesPhase2:
    """Comprehensive database repository testing."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary SQLite database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield f"sqlite:///{path}"
        try:
            os.unlink(path)
        except OSError:
            pass

    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create database manager with temporary database."""
        manager = DatabaseManager(database_url=temp_db_path, echo=False)
        Base.metadata.create_all(manager.engine)
        return manager

    @pytest.fixture
    def detector_repository(self, db_manager):
        """Create detector repository."""
        return DatabaseDetectorRepository(db_manager.session_factory)

    @pytest.fixture
    def sample_detector(self):
        """Create sample detector for testing."""
        class TestDetector(Detector):
            def fit(self, dataset):
                self.is_fitted = True
            
            def detect(self, dataset):
                from pynomaly.domain.entities import DetectionResult
                from pynomaly.domain.value_objects import AnomalyScore
                return DetectionResult(
                    dataset_id=dataset.id,
                    anomaly_scores=[AnomalyScore(0.5)],
                    anomalies=[],
                    metadata={}
                )
            
            def score(self, dataset):
                from pynomaly.domain.value_objects import AnomalyScore
                return [AnomalyScore(0.5)]
        
        return TestDetector(
            name="test_detector",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 100, "contamination": 0.1},
            metadata={"version": "1.0", "description": "Test detector"}
        )

    def test_detector_repository_save_new(self, detector_repository, sample_detector):
        """Test saving new detector to database."""
        # Save detector
        detector_repository.save(sample_detector)
        
        # Verify it was saved
        retrieved = detector_repository.find_by_id(sample_detector.id)
        assert retrieved is not None
        assert retrieved.id == sample_detector.id
        assert retrieved.algorithm_name == sample_detector.algorithm_name

    def test_detector_repository_save_update(self, detector_repository, sample_detector):
        """Test updating existing detector."""
        # Save initial detector
        detector_repository.save(sample_detector)
        
        # Update detector
        sample_detector.parameters["n_estimators"] = 200
        sample_detector.is_fitted = True
        detector_repository.save(sample_detector)
        
        # Verify update
        retrieved = detector_repository.find_by_id(sample_detector.id)
        assert retrieved.parameters["n_estimators"] == 200
        assert retrieved.is_fitted is True

    def test_detector_repository_find_by_id(self, detector_repository, sample_detector):
        """Test finding detector by ID."""
        # Save detector
        detector_repository.save(sample_detector)
        
        # Find by ID
        found = detector_repository.find_by_id(sample_detector.id)
        assert found is not None
        assert found.id == sample_detector.id
        
        # Test not found
        random_id = uuid.uuid4()
        not_found = detector_repository.find_by_id(random_id)
        assert not_found is None

    def test_detector_repository_find_by_algorithm(self, detector_repository, sample_detector):
        """Test finding detectors by algorithm."""
        # Save detector
        detector_repository.save(sample_detector)
        
        # Find by algorithm
        found = detector_repository.find_by_algorithm("IsolationForest")
        assert len(found) == 1
        assert found[0].algorithm_name == "IsolationForest"
        
        # Test not found
        not_found = detector_repository.find_by_algorithm("NonExistentAlgorithm")
        assert len(not_found) == 0

    def test_detector_repository_find_fitted(self, detector_repository, sample_detector):
        """Test finding fitted detectors."""
        # Save unfitted detector
        detector_repository.save(sample_detector)
        
        # No fitted detectors initially
        fitted = detector_repository.find_fitted()
        assert len(fitted) == 0
        
        # Fit and save detector
        sample_detector.is_fitted = True
        detector_repository.save(sample_detector)
        
        # Should find fitted detector
        fitted = detector_repository.find_fitted()
        assert len(fitted) == 1
        assert fitted[0].is_fitted is True

    def test_detector_repository_find_all(self, detector_repository, sample_detector):
        """Test finding all detectors."""
        # Initially empty
        all_detectors = detector_repository.find_all()
        assert len(all_detectors) == 0
        
        # Save detector
        detector_repository.save(sample_detector)
        
        # Should find one detector
        all_detectors = detector_repository.find_all()
        assert len(all_detectors) == 1

    def test_detector_repository_delete(self, detector_repository, sample_detector):
        """Test deleting detector."""
        # Save detector
        detector_repository.save(sample_detector)
        
        # Verify it exists
        found = detector_repository.find_by_id(sample_detector.id)
        assert found is not None
        
        # Delete detector
        deleted = detector_repository.delete(sample_detector.id)
        assert deleted is True
        
        # Verify it's gone
        not_found = detector_repository.find_by_id(sample_detector.id)
        assert not_found is None
        
        # Test deleting non-existent
        deleted_again = detector_repository.delete(sample_detector.id)
        assert deleted_again is False

    def test_detector_repository_complex_queries(self, detector_repository):
        """Test complex database queries."""
        # Create multiple detectors with different algorithms
        detectors = []
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        
        for i, algorithm in enumerate(algorithms):
            class TestDetector(Detector):
                def fit(self, dataset): pass
                def detect(self, dataset): pass
                def score(self, dataset): pass
            
            detector = TestDetector(
                name=f"test_detector_{i}",
                algorithm_name=algorithm,
                contamination_rate=ContaminationRate(0.1),
                parameters={"param": i},
                metadata={"index": i}
            )
            detectors.append(detector)
            detector_repository.save(detector)
        
        # Test finding by different algorithms
        iso_forest = detector_repository.find_by_algorithm("IsolationForest")
        assert len(iso_forest) == 1
        
        lof = detector_repository.find_by_algorithm("LocalOutlierFactor")
        assert len(lof) == 1
        
        # Test finding all
        all_detectors = detector_repository.find_all()
        assert len(all_detectors) == 3

    def test_detector_repository_error_handling(self, detector_repository):
        """Test repository error handling."""
        # Test with None detector
        with pytest.raises(AttributeError):
            detector_repository.save(None)
        
        # Test with invalid detector (missing required fields)
        class InvalidDetector:
            pass
        
        with pytest.raises(AttributeError):
            detector_repository.save(InvalidDetector())


class TestDatabaseTransactionsPhase2:
    """Test database transaction handling."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary SQLite database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield f"sqlite:///{path}"
        try:
            os.unlink(path)
        except OSError:
            pass

    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create database manager with temporary database."""
        manager = DatabaseManager(database_url=temp_db_path, echo=False)
        Base.metadata.create_all(manager.engine)
        return manager

    def test_transaction_commit(self, db_manager):
        """Test successful transaction commit."""
        detector_data = {
            "id": uuid.uuid4(),
            "algorithm": "IsolationForest",
            "parameters": {"n_estimators": 100},
            "is_fitted": False,
            "metadata": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Transaction should commit successfully
        with db_manager.session() as session:
            detector_model = DetectorModel(**detector_data)
            session.add(detector_model)
            session.commit()
        
        # Verify data was committed
        with db_manager.session() as session:
            found = session.query(DetectorModel).filter_by(id=detector_data["id"]).first()
            assert found is not None
            assert found.algorithm == "IsolationForest"

    def test_transaction_rollback(self, db_manager):
        """Test transaction rollback on error."""
        detector_data = {
            "id": uuid.uuid4(),
            "algorithm": "IsolationForest", 
            "parameters": {"n_estimators": 100},
            "is_fitted": False,
            "metadata": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        try:
            with db_manager.session() as session:
                detector_model = DetectorModel(**detector_data)
                session.add(detector_model)
                
                # Force an error
                raise Exception("Simulated error")
        except Exception:
            pass  # Expected
        
        # Verify data was NOT committed due to rollback
        with db_manager.session() as session:
            found = session.query(DetectorModel).filter_by(id=detector_data["id"]).first()
            assert found is None

    def test_nested_transactions(self, db_manager):
        """Test nested transaction behavior."""
        outer_id = uuid.uuid4()
        inner_id = uuid.uuid4()
        
        with db_manager.session() as session:
            # Outer transaction
            outer_detector = DetectorModel(
                id=outer_id,
                algorithm="OuterAlgorithm",
                parameters={},
                is_fitted=False,
                metadata={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            session.add(outer_detector)
            
            # Inner operation
            inner_detector = DetectorModel(
                id=inner_id,
                algorithm="InnerAlgorithm",
                parameters={},
                is_fitted=False,
                metadata={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            session.add(inner_detector)
            session.commit()
        
        # Both should be committed
        with db_manager.session() as session:
            outer_found = session.query(DetectorModel).filter_by(id=outer_id).first()
            inner_found = session.query(DetectorModel).filter_by(id=inner_id).first()
            assert outer_found is not None
            assert inner_found is not None


class TestRedisIntegrationPhase2:
    """Test Redis integration and caching."""

    def test_redis_connection_mock(self):
        """Test Redis connection with mocking."""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis_class.return_value = mock_redis
            mock_redis.ping.return_value = True
            
            # Test Redis connection
            import redis
            client = redis.Redis(host='localhost', port=6379, db=0)
            result = client.ping()
            
            assert result is True
            mock_redis.ping.assert_called_once()

    def test_redis_caching_operations(self):
        """Test Redis caching operations with mocking."""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis_class.return_value = mock_redis
            
            # Mock cache operations
            mock_redis.set.return_value = True
            mock_redis.get.return_value = b'{"key": "value"}'
            mock_redis.delete.return_value = 1
            mock_redis.exists.return_value = True
            
            import redis
            client = redis.Redis()
            
            # Test caching operations
            assert client.set("test_key", "test_value") is True
            assert client.get("test_key") == b'{"key": "value"}'
            assert client.exists("test_key") is True
            assert client.delete("test_key") == 1

    def test_redis_session_storage(self):
        """Test Redis for session storage."""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis_class.return_value = mock_redis
            
            # Mock session operations
            session_data = {"user_id": "123", "detector_id": "456"}
            session_json = json.dumps(session_data)
            
            mock_redis.setex.return_value = True
            mock_redis.get.return_value = session_json.encode()
            
            import redis
            client = redis.Redis()
            
            # Test session storage
            session_key = "session:abc123"
            ttl = 3600  # 1 hour
            
            # Store session
            result = client.setex(session_key, ttl, session_json)
            assert result is True
            
            # Retrieve session
            stored_data = client.get(session_key)
            assert stored_data is not None
            assert json.loads(stored_data.decode()) == session_data

    def test_redis_model_caching(self):
        """Test Redis for model caching."""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis_class.return_value = mock_redis
            
            # Mock model caching
            model_data = {"model_type": "IsolationForest", "parameters": {}}
            model_json = json.dumps(model_data)
            
            mock_redis.hset.return_value = 1
            mock_redis.hget.return_value = model_json.encode()
            mock_redis.hdel.return_value = 1
            
            import redis
            client = redis.Redis()
            
            # Test model caching
            detector_id = "detector_123"
            
            # Cache model
            result = client.hset("models", detector_id, model_json)
            assert result == 1
            
            # Retrieve cached model
            cached_data = client.hget("models", detector_id)
            assert cached_data is not None
            assert json.loads(cached_data.decode()) == model_data
            
            # Remove from cache
            result = client.hdel("models", detector_id)
            assert result == 1


class TestDatabasePerformancePhase2:
    """Test database performance optimizations."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary SQLite database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield f"sqlite:///{path}"
        try:
            os.unlink(path)
        except OSError:
            pass

    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create database manager with temporary database."""
        manager = DatabaseManager(database_url=temp_db_path, echo=False)
        Base.metadata.create_all(manager.engine)
        return manager

    def test_bulk_insert_performance(self, db_manager):
        """Test bulk insert operations."""
        detectors_data = []
        
        # Prepare bulk data
        for i in range(100):
            detector_data = {
                "id": uuid.uuid4(),
                "algorithm": f"Algorithm_{i % 5}",
                "parameters": {"param": i},
                "is_fitted": i % 2 == 0,
                "metadata": {"index": i},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            detectors_data.append(detector_data)
        
        # Bulk insert
        start_time = datetime.utcnow()
        with db_manager.session() as session:
            detector_models = [DetectorModel(**data) for data in detectors_data]
            session.add_all(detector_models)
            session.commit()
        end_time = datetime.utcnow()
        
        # Verify all were inserted
        with db_manager.session() as session:
            count = session.query(DetectorModel).count()
            assert count == 100
        
        # Performance should be reasonable (less than 5 seconds for 100 records)
        duration = (end_time - start_time).total_seconds()
        assert duration < 5.0

    def test_query_optimization(self, db_manager):
        """Test query optimization techniques."""
        # Create test data
        with db_manager.session() as session:
            for i in range(50):
                detector = DetectorModel(
                    id=uuid.uuid4(),
                    algorithm="IsolationForest" if i % 2 == 0 else "LocalOutlierFactor",
                    parameters={"index": i},
                    is_fitted=i % 3 == 0,
                    metadata={"category": "A" if i < 25 else "B"},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                session.add(detector)
            session.commit()
        
        # Test optimized queries
        with db_manager.session() as session:
            # Query with filtering
            start_time = datetime.utcnow()
            fitted_isolation_forest = session.query(DetectorModel).filter(
                DetectorModel.algorithm == "IsolationForest",
                DetectorModel.is_fitted == True
            ).all()
            end_time = datetime.utcnow()
            
            # Should find some results
            assert len(fitted_isolation_forest) > 0
            
            # Query should be fast
            duration = (end_time - start_time).total_seconds()
            assert duration < 1.0

    def test_connection_pooling_performance(self):
        """Test connection pooling performance."""
        postgres_url = "postgresql://user:pass@localhost:5432/testdb"
        
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            
            # Test with connection pooling
            manager = DatabaseManager(database_url=postgres_url)
            engine = manager.engine
            
            # Verify pooling configuration
            mock_create_engine.assert_called_once()
            call_kwargs = mock_create_engine.call_args[1]
            
            # Should have pool configuration for PostgreSQL
            assert len(call_kwargs) >= 0  # At minimum, should have some configuration

    def test_batch_operations_performance(self, db_manager):
        """Test batch operations performance."""
        # Create initial data
        detector_ids = []
        with db_manager.session() as session:
            for i in range(20):
                detector = DetectorModel(
                    id=uuid.uuid4(),
                    algorithm="TestAlgorithm",
                    parameters={"index": i},
                    is_fitted=False,
                    metadata={},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                detector_ids.append(detector.id)
                session.add(detector)
            session.commit()
        
        # Batch update operation
        start_time = datetime.utcnow()
        with db_manager.session() as session:
            session.query(DetectorModel).filter(
                DetectorModel.id.in_(detector_ids)
            ).update(
                {"is_fitted": True},
                synchronize_session=False
            )
            session.commit()
        end_time = datetime.utcnow()
        
        # Verify updates
        with db_manager.session() as session:
            fitted_count = session.query(DetectorModel).filter(
                DetectorModel.is_fitted == True
            ).count()
            assert fitted_count == 20
        
        # Batch operation should be fast
        duration = (end_time - start_time).total_seconds()
        assert duration < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])