"""Integration test configuration with Testcontainers support."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import responses
import pandas as pd
import numpy as np

# Add src to Python path for imports
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Core dependencies
import sys
sys.path.insert(0, str(project_root / "src"))

# Test containers for services
try:
    from testcontainers.postgres import PostgreSQLContainer
    from testcontainers.redis import RedisContainer
    from testcontainers.elasticsearch import ElasticSearchContainer
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    # Mock containers for when testcontainers is not available
    class MockContainer:
        def __init__(self, *args, **kwargs):
            self.host = "localhost"
            self.port = 5432
            self.database = "test"
            self.username = "test"
            self.password = "test"
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def get_connection_url(self):
            return "sqlite:///:memory:"
            
        def get_exposed_port(self, port):
            return port
    
    PostgreSQLContainer = MockContainer
    RedisContainer = MockContainer
    ElasticSearchContainer = MockContainer

# Try to import pynomaly components
try:
    from pynomaly.domain.entities import Dataset, DetectionResult, Detector
    from pynomaly.domain.value_objects import AnomalyScore
    from pynomaly.infrastructure.config import Container, Settings
    PYNOMALY_AVAILABLE = True
except ImportError:
    PYNOMALY_AVAILABLE = False
    # Mock basic entities for testing
    class MockDataset:
        def __init__(self, name, data, features=None, description=None):
            self.name = name
            self.data = data
            self.features = features or []
            self.description = description
            self.id = "test-dataset-id"
    
    class MockDetector:
        def __init__(self, algorithm_name, parameters=None, metadata=None):
            self.algorithm_name = algorithm_name
            self.parameters = parameters or {}
            self.metadata = metadata or {}
            self.id = "test-detector-id"
            self.is_fitted = False
    
    class MockDetectionResult:
        def __init__(self, detector_id, dataset_id, scores):
            self.detector_id = detector_id
            self.dataset_id = dataset_id
            self.scores = scores
            self.id = "test-result-id"
    
    Dataset = MockDataset
    Detector = MockDetector
    DetectionResult = MockDetectionResult


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgreSQLContainer, None, None]:
    """Provide a PostgreSQL container for integration tests."""
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("Testcontainers not available")
    
    with PostgreSQLContainer("postgres:15") as postgres:
        postgres.driver = "psycopg2"
        yield postgres


@pytest.fixture(scope="session")
def redis_container() -> Generator[RedisContainer, None, None]:
    """Provide a Redis container for integration tests."""
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("Testcontainers not available")
    
    with RedisContainer("redis:7-alpine") as redis:
        yield redis


@pytest.fixture(scope="session")
def elasticsearch_container() -> Generator[ElasticSearchContainer, None, None]:
    """Provide an Elasticsearch container for integration tests."""
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("Testcontainers not available")
    
    with ElasticSearchContainer("elasticsearch:8.8.0") as elasticsearch:
        elasticsearch.with_env("discovery.type", "single-node")
        elasticsearch.with_env("xpack.security.enabled", "false")
        yield elasticsearch


@pytest.fixture
def db_connection_string(postgres_container: PostgreSQLContainer) -> str:
    """Get database connection string from container."""
    return postgres_container.get_connection_url()


@pytest.fixture
def redis_connection_string(redis_container: RedisContainer) -> str:
    """Get Redis connection string from container."""
    return f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"


@pytest.fixture
def elasticsearch_connection_string(elasticsearch_container: ElasticSearchContainer) -> str:
    """Get Elasticsearch connection string from container."""
    return f"http://{elasticsearch_container.get_container_host_ip()}:{elasticsearch_container.get_exposed_port(9200)}"


@pytest.fixture
def test_settings(db_connection_string: str, redis_connection_string: str) -> dict:
    """Create test settings with container connection strings."""
    return {
        "database_url": db_connection_string,
        "redis_url": redis_connection_string,
        "environment": "test",
        "debug": True,
        "auth_enabled": False,
        "cache_enabled": True,
        "monitoring_enabled": False,
    }


@pytest.fixture
def test_container(test_settings: dict):
    """Create test container with mocked dependencies."""
    if not PYNOMALY_AVAILABLE:
        pytest.skip("Pynomaly not available")
    
    container = Container()
    # Override settings
    container.config.override(test_settings)
    return container


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate sample data for testing."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (100, 5))
    
    # Add some anomalies
    data[90:, :] *= 3  # Make last 10 rows outliers
    
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = 0
    df.loc[90:, 'target'] = 1  # Mark outliers
    
    return df


@pytest.fixture
def large_sample_data() -> pd.DataFrame:
    """Generate large sample data for performance testing."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (10000, 10))
    
    # Add some anomalies
    data[9000:, :] *= 2.5  # Make last 1000 rows outliers
    
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = 0
    df.loc[9000:, 'target'] = 1  # Mark outliers
    
    return df


@pytest.fixture
def mock_third_party_service():
    """Mock third-party SaaS calls using responses."""
    with responses.RequestsMock() as rsps:
        # Mock external API calls
        rsps.add(
            responses.GET,
            "https://api.example.com/health",
            json={"status": "healthy"},
            status=200
        )
        
        rsps.add(
            responses.POST,
            "https://api.example.com/process",
            json={"result": "processed"},
            status=200
        )
        
        rsps.add(
            responses.GET,
            "https://api.monitoring.example.com/metrics",
            json={"metrics": {"cpu": 50, "memory": 60}},
            status=200
        )
        
        yield rsps


@pytest.fixture
def async_mock_third_party_service():
    """Mock async third-party SaaS calls using aioresponses."""
    try:
        from aioresponses import aioresponses
        
        with aioresponses() as m:
            # Mock async HTTP calls
            m.get(
                "https://api.example.com/health",
                payload={"status": "healthy"},
                status=200
            )
            
            m.post(
                "https://api.example.com/process",
                payload={"result": "processed"},
                status=200
            )
            
            m.get(
                "https://api.monitoring.example.com/metrics",
                payload={"metrics": {"cpu": 50, "memory": 60}},
                status=200
            )
            
            yield m
    except ImportError:
        # Fallback if aioresponses is not available
        yield None


@pytest.fixture
def db_transaction(db_connection_string: str):
    """Database transaction fixture with savepoint and rollback."""
    try:
        import sqlalchemy
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        engine = create_engine(db_connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Begin transaction
        trans = session.begin()
        savepoint = session.begin_nested()
        
        try:
            yield session
        finally:
            # Always rollback to savepoint
            if savepoint.is_active:
                savepoint.rollback()
            if trans.is_active:
                trans.rollback()
            session.close()
    except ImportError:
        # Fallback if SQLAlchemy is not available
        yield None


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_datasets():
    """Create test datasets for different scenarios."""
    datasets = {}
    
    # Normal dataset
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 5))
    datasets['normal'] = Dataset(
        name="Normal Dataset",
        data=pd.DataFrame(normal_data, columns=[f'feature_{i}' for i in range(5)]),
        features=[f'feature_{i}' for i in range(5)],
        description="Normal distribution data"
    )
    
    # Anomalous dataset
    anomalous_data = np.random.normal(0, 1, (1000, 5))
    anomalous_data[900:, :] *= 5  # Strong anomalies
    datasets['anomalous'] = Dataset(
        name="Anomalous Dataset",
        data=pd.DataFrame(anomalous_data, columns=[f'feature_{i}' for i in range(5)]),
        features=[f'feature_{i}' for i in range(5)],
        description="Data with strong anomalies"
    )
    
    # High-dimensional dataset
    high_dim_data = np.random.normal(0, 1, (500, 20))
    datasets['high_dim'] = Dataset(
        name="High Dimensional Dataset",
        data=pd.DataFrame(high_dim_data, columns=[f'feature_{i}' for i in range(20)]),
        features=[f'feature_{i}' for i in range(20)],
        description="High-dimensional data"
    )
    
    return datasets


@pytest.fixture
def test_detectors():
    """Create test detectors for different algorithms."""
    detectors = {}
    
    # Isolation Forest
    detectors['isolation_forest'] = Detector(
        algorithm_name="IsolationForest",
        parameters={"contamination": 0.1, "random_state": 42},
        metadata={"description": "Isolation Forest detector"}
    )
    
    # Local Outlier Factor
    detectors['lof'] = Detector(
        algorithm_name="LocalOutlierFactor",
        parameters={"contamination": 0.1, "n_neighbors": 20},
        metadata={"description": "Local Outlier Factor detector"}
    )
    
    # One-Class SVM
    detectors['ocsvm'] = Detector(
        algorithm_name="OneClassSVM",
        parameters={"gamma": "auto", "nu": 0.1},
        metadata={"description": "One-Class SVM detector"}
    )
    
    return detectors


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup fixture that runs after each test."""
    yield
    # Cleanup code here if needed
    pass


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration
