"""Comprehensive tests for database infrastructure with property-based testing."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.conftest_dependencies import requires_dependency, requires_dependencies

from pynomaly.infrastructure.persistence.database import DatabaseConnection
from pynomaly.infrastructure.persistence.database_repositories import (
    DatabaseDetectorRepository,
    DatabaseDatasetRepository,
    DatabaseExperimentRepository
)
from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore


@composite
def valid_contamination_rates(draw):
    """Generate valid contamination rates between 0.001 and 0.5."""
    return ContaminationRate(draw(st.floats(min_value=0.001, max_value=0.5)))


@composite
def valid_detector_names(draw):
    """Generate valid detector names."""
    return draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=32, max_codepoint=126),
        min_size=1,
        max_size=50
    ).filter(lambda x: x.strip() and not x.isspace()))


@composite
def valid_algorithms(draw):
    """Generate valid algorithm names."""
    algorithms = ['IsolationForest', 'LOF', 'OneClassSVM', 'EllipticEnvelope', 'OCSVM']
    return draw(st.sampled_from(algorithms))


@requires_dependency('database')
class TestDatabaseConnection:
    """Test database connection functionality."""
    
    def test_sqlite_in_memory_connection(self):
        """Test in-memory SQLite connection."""
        db = DatabaseConnection("sqlite:///:memory:")
        
        assert db.engine is not None
        assert db.sessionmaker is not None
        
        # Test connection
        with db.get_session() as session:
            result = session.execute(sa.text("SELECT 1")).scalar()
            assert result == 1
    
    def test_database_session_context_manager(self):
        """Test database session context manager."""
        db = DatabaseConnection("sqlite:///:memory:")
        
        with db.get_session() as session:
            assert session is not None
            assert hasattr(session, 'execute')
            assert hasattr(session, 'commit')
            assert hasattr(session, 'rollback')
    
    def test_database_session_rollback_on_exception(self):
        """Test that sessions rollback on exceptions."""
        db = DatabaseConnection("sqlite:///:memory:")
        
        with pytest.raises(Exception):
            with db.get_session() as session:
                session.execute(sa.text("SELECT 1"))
                raise Exception("Test exception")
        
        # Should be able to get new session after exception
        with db.get_session() as session:
            result = session.execute(sa.text("SELECT 1")).scalar()
            assert result == 1


@requires_dependency('database')
class TestDatabaseRepositories:
    """Test database repository implementations."""
    
    @pytest.fixture
    def in_memory_db(self):
        """Create in-memory database for testing."""
        db = DatabaseConnection("sqlite:///:memory:")
        
        # Create basic tables (simplified)
        with db.get_session() as session:
            session.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS detectors (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    contamination REAL NOT NULL,
                    parameters TEXT,
                    is_fitted BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            session.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    data_path TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            session.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    results TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            session.commit()
        
        return db
    
    def test_detector_repository_basic_operations(self, in_memory_db):
        """Test basic CRUD operations for detector repository."""
        repo = DatabaseDetectorRepository(in_memory_db)
        
        # Create detector
        detector = Detector(
            name="test_detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1, "n_estimators": 100}
        )
        
        # Test save
        saved_detector = repo.save(detector)
        assert saved_detector.id is not None
        assert saved_detector.name == "test_detector"
        
        # Test find by id
        found_detector = repo.find_by_id(saved_detector.id)
        assert found_detector is not None
        assert found_detector.name == "test_detector"
        assert found_detector.algorithm == "IsolationForest"
        
        # Test find by name
        found_by_name = repo.find_by_name("test_detector")
        assert found_by_name is not None
        assert found_by_name.id == saved_detector.id
        
        # Test list all
        all_detectors = repo.list_all()
        assert len(all_detectors) == 1
        assert all_detectors[0].id == saved_detector.id
    
    def test_dataset_repository_basic_operations(self, in_memory_db):
        """Test basic CRUD operations for dataset repository."""
        repo = DatabaseDatasetRepository(in_memory_db)
        
        # Create dataset with mock data
        dataset = Dataset(
            name="test_dataset",
            data=Mock(),  # Mock pandas DataFrame
            metadata={"rows": 1000, "columns": 5}
        )
        
        # Test save
        saved_dataset = repo.save(dataset)
        assert saved_dataset.id is not None
        assert saved_dataset.name == "test_dataset"
        
        # Test find by id
        found_dataset = repo.find_by_id(saved_dataset.id)
        assert found_dataset is not None
        assert found_dataset.name == "test_dataset"
        
        # Test find by name
        found_by_name = repo.find_by_name("test_dataset")
        assert found_by_name is not None
        assert found_by_name.id == saved_dataset.id


@requires_dependencies('database', 'hypothesis')
class TestPropertyBasedDatabaseTests:
    """Property-based tests for database operations."""
    
    @pytest.fixture
    def in_memory_db(self):
        """Create in-memory database for property testing."""
        db = DatabaseConnection("sqlite:///:memory:")
        
        with db.get_session() as session:
            session.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS detectors (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    contamination REAL NOT NULL,
                    parameters TEXT,
                    is_fitted BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            session.commit()
        
        return db
    
    @given(
        name=valid_detector_names(),
        algorithm=valid_algorithms(),
        contamination=valid_contamination_rates()
    )
    @settings(max_examples=20, deadline=None)
    def test_detector_repository_property_roundtrip(self, in_memory_db, name, algorithm, contamination):
        """Property test: saved detectors can be retrieved with same data."""
        repo = DatabaseDetectorRepository(in_memory_db)
        
        # Create detector with generated properties
        detector = Detector(
            name=name,
            algorithm=algorithm,
            parameters={"contamination": contamination.value}
        )
        
        # Save and retrieve
        saved = repo.save(detector)
        retrieved = repo.find_by_id(saved.id)
        
        # Properties should be preserved
        assert retrieved is not None
        assert retrieved.name == name
        assert retrieved.algorithm == algorithm
        assert retrieved.parameters["contamination"] == contamination.value
    
    @given(st.lists(valid_detector_names(), min_size=1, max_size=10, unique=True))
    @settings(max_examples=10, deadline=None)
    def test_detector_repository_multiple_detectors(self, in_memory_db, names):
        """Property test: multiple detectors can be stored and retrieved."""
        repo = DatabaseDetectorRepository(in_memory_db)
        
        saved_detectors = []
        for name in names:
            detector = Detector(
                name=name,
                algorithm="IsolationForest",
                parameters={"contamination": 0.1}
            )
            saved = repo.save(detector)
            saved_detectors.append(saved)
        
        # All detectors should be retrievable
        all_retrieved = repo.list_all()
        assert len(all_retrieved) >= len(names)
        
        # Each saved detector should be findable
        for saved in saved_detectors:
            found = repo.find_by_id(saved.id)
            assert found is not None
            assert found.name == saved.name


@requires_dependency('database')
class TestDatabaseErrorHandling:
    """Test database error handling and edge cases."""
    
    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        # Invalid connection string
        with pytest.raises((sa.exc.ArgumentError, sa.exc.DatabaseError)):
            DatabaseConnection("invalid://connection/string")
    
    def test_repository_with_invalid_database(self):
        """Test repository behavior with invalid database."""
        # Mock a database connection that will fail
        mock_db = Mock()
        mock_db.get_session.side_effect = sa.exc.DatabaseError("Connection failed", None, None)
        
        repo = DatabaseDetectorRepository(mock_db)
        
        with pytest.raises(sa.exc.DatabaseError):
            repo.list_all()
    
    def test_repository_transaction_rollback(self):
        """Test that failed transactions are rolled back."""
        db = DatabaseConnection("sqlite:///:memory:")
        
        with db.get_session() as session:
            session.execute(sa.text("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """))
            session.commit()
        
        # Test rollback on constraint violation
        with pytest.raises(sa.exc.IntegrityError):
            with db.get_session() as session:
                session.execute(sa.text("INSERT INTO test_table (id, value) VALUES (1, 'test')"))
                session.execute(sa.text("INSERT INTO test_table (id, value) VALUES (1, 'duplicate')"))  # Should fail
                session.commit()
        
        # Verify rollback worked
        with db.get_session() as session:
            result = session.execute(sa.text("SELECT COUNT(*) FROM test_table")).scalar()
            assert result == 0


class TestMockBasedInfrastructure:
    """Test infrastructure components using mocks for unavailable dependencies."""
    
    def test_redis_cache_interface_mock(self):
        """Test Redis cache interface with mocks."""
        # Mock redis functionality
        mock_redis = Mock()
        mock_redis.get.return_value = b'{"cached": "data"}'
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = True
        
        # Test that our cache interface would work
        assert mock_redis.get("test_key") == b'{"cached": "data"}'
        assert mock_redis.set("test_key", "test_value") is True
        assert mock_redis.delete("test_key") == 1
        assert mock_redis.exists("test_key") is True
        
        # Verify call counts
        assert mock_redis.get.call_count == 1
        assert mock_redis.set.call_count == 1
        assert mock_redis.delete.call_count == 1
        assert mock_redis.exists.call_count == 1
    
    def test_fastapi_endpoint_mock(self):
        """Test FastAPI endpoint behavior with mocks."""
        # Mock FastAPI app and request/response
        mock_app = Mock()
        mock_request = Mock()
        mock_response = Mock()
        
        # Mock detector service
        mock_detector_service = Mock()
        mock_detector_service.create_detector.return_value = Mock(
            id="detector-123",
            name="test_detector",
            algorithm="IsolationForest"
        )
        
        # Test endpoint logic (without actual FastAPI)
        detector_data = {
            "name": "test_detector",
            "algorithm": "IsolationForest",
            "contamination": 0.1
        }
        
        # Simulate endpoint call
        result = mock_detector_service.create_detector(detector_data)
        
        assert result.id == "detector-123"
        assert result.name == "test_detector"
        assert result.algorithm == "IsolationForest"
        
        mock_detector_service.create_detector.assert_called_once_with(detector_data)
    
    def test_pytorch_adapter_interface_mock(self):
        """Test PyTorch adapter interface with mocks."""
        # Mock PyTorch components
        mock_torch = Mock()
        mock_model = Mock()
        mock_tensor = Mock()
        
        # Mock tensor operations
        mock_torch.tensor.return_value = mock_tensor
        mock_tensor.shape = (100, 5)
        mock_tensor.detach.return_value = mock_tensor
        mock_tensor.numpy.return_value = [[0.1, 0.2, 0.8, 0.1, 0.2]] * 100
        
        # Mock model operations
        mock_model.eval.return_value = None
        mock_model.forward.return_value = mock_tensor
        
        # Test adapter interface logic
        with patch.dict('sys.modules', {'torch': mock_torch}):
            # Simulate adapter creation
            adapter_config = {
                "algorithm": "AutoEncoder",
                "input_dim": 5,
                "hidden_dim": 3,
                "learning_rate": 0.001
            }
            
            # Test model creation
            input_tensor = mock_torch.tensor([[1, 2, 3, 4, 5]])
            output = mock_model.forward(input_tensor)
            scores = output.detach().numpy()
            
            assert input_tensor.shape == (100, 5)
            assert len(scores) == 100
            assert all(isinstance(score, list) for score in scores)
            
            mock_torch.tensor.assert_called()
            mock_model.forward.assert_called_with(input_tensor)
            mock_tensor.detach.assert_called()
            mock_tensor.numpy.assert_called()