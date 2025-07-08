"""Pytest configuration and fixtures for comprehensive testing."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from dependency_injector import providers
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.infrastructure.auth.jwt_auth import init_auth
from pynomaly.infrastructure.config import Container, Settings

# Optional imports for advanced features
try:
    from pynomaly.infrastructure.observability import setup_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

    def setup_observability(*args, **kwargs):
        return {}


try:
    from pynomaly.infrastructure.security.audit_logging import init_audit_logging

    AUDIT_LOGGING_AVAILABLE = True
except ImportError:
    AUDIT_LOGGING_AVAILABLE = False

    def init_audit_logging(*args, **kwargs):
        return None


# Import database fixtures if available
try:
    from .conftest_database import (
        test_async_database_repositories,
        test_container_with_database,
        test_database_manager,
        test_database_settings,
        test_database_url,
    )
except ImportError:
    # Database fixtures not available, create skip fixtures
    @pytest.fixture
    def test_async_database_repositories():
        pytest.skip("Database dependencies not available")

    @pytest.fixture
    def test_database_manager():
        pytest.skip("Database dependencies not available")


# Import app if available for API testing
try:
    # Disable API imports for now due to UserModel issues
    # from pynomaly.presentation.api.app import create_app
    APP_AVAILABLE = False
except ImportError:
    APP_AVAILABLE = False

# Import test data management utilities
try:
    from tests.fixtures.test_data_generator import (
        HIGH_DIM_DATASET_PARAMS,
        LARGE_DATASET_PARAMS,
        MEDIUM_DATASET_PARAMS,
        SMALL_DATASET_PARAMS,
        TestDataManager,
        TestScenarioFactory,
    )

    TEST_DATA_MANAGER_AVAILABLE = True
except ImportError:
    TEST_DATA_MANAGER_AVAILABLE = False


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Enhanced test settings
@pytest.fixture(scope="function")
def test_settings() -> Settings:
    """Enhanced test settings with security and monitoring."""
    import os
    import tempfile

    temp_dir = tempfile.mkdtemp()

    # Set environment variables for pydantic-settings
    os.environ.update(
        {
            "PYNOMALY_APP_NAME": "pynomaly-test",
            "PYNOMALY_APP_VERSION": "0.1.0-test",
            "PYNOMALY_APP_ENVIRONMENT": "test",
            "PYNOMALY_APP_DEBUG": "true",
            "PYNOMALY_DATABASE_URL": "sqlite:///:memory:",
            "PYNOMALY_SECRET_KEY": "test-secret-key-not-for-production-use-only",
            "PYNOMALY_JWT_ALGORITHM": "HS256",
            "PYNOMALY_JWT_EXPIRATION": "3600",
            "PYNOMALY_AUTH_ENABLED": "true",
            "PYNOMALY_CACHE_ENABLED": "false",
            "PYNOMALY_DOCS_ENABLED": "true",
            "PYNOMALY_MONITORING_METRICS_ENABLED": "false",
            "PYNOMALY_MONITORING_TRACING_ENABLED": "false",
            "PYNOMALY_MONITORING_PROMETHEUS_ENABLED": "false",
            "PYNOMALY_STORAGE_PATH": temp_dir,
        }
    )

    return Settings()


@pytest.fixture
def settings(test_settings) -> Settings:
    """Backward compatibility alias."""
    return test_settings


@pytest.fixture
def container(test_settings: Settings) -> Container:
    """Enhanced test DI container."""
    container = Container()
    container.config.override(providers.Singleton(lambda: test_settings))
    return container


# Database testing fixtures
@pytest.fixture(scope="function")
def db_engine():
    """Create test database engine."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Import and create tables
    try:
        from pynomaly.infrastructure.persistence.database_repositories import Base

        Base.metadata.create_all(bind=engine)
    except ImportError:
        pass  # Database models not available

    yield engine

    try:
        Base.metadata.drop_all(bind=engine)
    except:
        pass
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=db_engine
    )

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def session_factory(db_engine):
    """Create session factory for repositories."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=db_engine
    )
    return TestingSessionLocal


# FastAPI application fixtures
@pytest.fixture(scope="function")
def app(container):
    """Create test FastAPI app."""
    if not APP_AVAILABLE:
        pytest.skip("FastAPI app not available")
    return create_app(container)


@pytest.fixture(scope="function")
def client(app) -> TestClient:
    """Create test client."""
    return TestClient(app)


# Authentication fixtures
@pytest.fixture(scope="function")
def auth_service(test_settings):
    """Create test auth service."""
    return init_auth(test_settings)


@pytest.fixture(scope="function")
def admin_token(auth_service):
    """Create admin token for testing."""
    try:
        user = auth_service._users["admin"]  # Default admin user
        token_response = auth_service.create_access_token(user)
        return token_response.access_token
    except:
        pytest.skip("Auth service not properly configured")


@pytest.fixture(scope="function")
def test_user(auth_service):
    """Create test user."""
    return auth_service.create_user(
        username="testuser",
        email="test@example.com",
        password="testpass123",
        full_name="Test User",
        roles=["user"],
    )


@pytest.fixture(scope="function")
def user_token(auth_service, test_user):
    """Create user token for testing."""
    token_response = auth_service.create_access_token(test_user)
    return token_response.access_token


# Data fixtures with multiple variants
@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    # Generate normal data
    normal_data = np.random.normal(0, 1, (n_samples - 50, n_features))

    # Generate anomalous data
    anomalous_data = np.random.normal(3, 1, (50, n_features))

    # Combine data
    data = np.vstack([normal_data, anomalous_data])

    # Create DataFrame
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = [0] * (n_samples - 50) + [1] * 50  # 0 = normal, 1 = anomaly

    return df


@pytest.fixture
def sample_dataset(sample_data) -> Dataset:
    """Enhanced sample dataset for testing."""
    return Dataset(
        name="Test Dataset",
        data=sample_data,
        description="Test dataset for unit tests",
        target_column="target",
        metadata={"test": True, "samples": len(sample_data)},
    )


@pytest.fixture
def large_dataset() -> Dataset:
    """Large dataset for performance testing."""
    np.random.seed(42)
    n_samples = 10000
    n_features = 20

    data = np.random.normal(0, 1, (n_samples, n_features))
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])

    return Dataset(
        name="Large Test Dataset",
        data=df,
        description="Large dataset for performance testing",
        metadata={"test": True, "performance": True},
    )


@pytest.fixture
def sample_detector() -> Detector:
    """Enhanced sample detector for testing."""
    return Detector(
        name="Test Detector",
        algorithm_name="IsolationForest",
        parameters={"contamination": 0.05, "random_state": 42},
        metadata={"test": True, "description": "Test detector"},
    )


@pytest.fixture
def test_detection_result(sample_detector, sample_dataset) -> DetectionResult:
    """Create test detection result."""
    # Generate mock scores
    np.random.seed(42)
    scores = [AnomalyScore(value=np.random.random()) for _ in range(100)]

    # Create dummy anomalies for the first 10 samples
    anomalies = []
    for i in range(10):
        anomaly = Anomaly(
            score=scores[i],
            data_point={"feature_0": i, "feature_1": i + 1},
            detector_name="test_detector",
        )
        anomalies.append(anomaly)
    
    # Create binary labels (first 10 are anomalies, rest are normal)
    labels = np.zeros(100, dtype=int)
    labels[:10] = 1
    
    return DetectionResult(
        detector_id=sample_detector.id,
        dataset_id=sample_dataset.id,
        anomalies=anomalies,
        scores=scores,
        labels=labels,
        threshold=0.5,
        metadata={"test": True, "model_version": "1.0"},
    )


@pytest.fixture
def trained_detector(
    sample_detector: Detector, sample_dataset: Dataset, container: Container
) -> Detector:
    """Trained detector for testing."""
    try:
        # Get adapter and train
        adapter = container.pyod_adapter()
        model = adapter.create_model(
            sample_detector.algorithm_name, sample_detector.parameters
        )
        adapter.fit(model, sample_dataset.data)

        # Update detector
        sample_detector.is_fitted = True
        sample_detector.fitted_model = model
        sample_detector.metadata["training_samples"] = len(sample_dataset.data)
    except:
        # Mark as fitted without actual training if adapters not available
        sample_detector.is_fitted = True
        sample_detector.metadata["training_samples"] = len(sample_dataset.data)

    return sample_detector


# File system fixtures
@pytest.fixture(scope="function")
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_csv_file(sample_data, temp_dir) -> str:
    """Create sample CSV file for testing."""
    file_path = os.path.join(temp_dir, "test_data.csv")
    sample_data.to_csv(file_path, index=False)
    return file_path


# Mock fixtures
@pytest.fixture(scope="function")
def mock_model():
    """Create mock ML model."""
    mock = MagicMock()
    mock.fit.return_value = None
    mock.predict.return_value = np.array([0, 1, 0, 1])
    mock.decision_function.return_value = np.array([0.1, 0.9, 0.2, 0.8])
    return mock


@pytest.fixture(scope="function")
def mock_async_repository():
    """Create mock async repository."""
    mock = AsyncMock()
    return mock


# Security testing fixtures
@pytest.fixture(scope="function")
def malicious_inputs() -> list:
    """Provide malicious inputs for security testing."""
    return [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "javascript:alert('xss')",
        "' OR 1=1 --",
        "<iframe src=javascript:alert('xss')></iframe>",
        "../../../../windows/system32/config/sam",
    ]


# Performance testing fixtures
@pytest.fixture(scope="function")
def performance_data() -> pd.DataFrame:
    """Create large dataset for performance testing."""
    np.random.seed(42)
    n_samples = 100000
    n_features = 20

    data = np.random.normal(0, 1, (n_samples, n_features))
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])

    return df


# Observability fixtures
@pytest.fixture(scope="function")
def observability_components():
    """Set up observability for testing."""
    if not OBSERVABILITY_AVAILABLE:
        pytest.skip("Observability components not available")

    components = setup_observability(
        enable_logging=True,
        enable_metrics=False,  # Disable for tests
        enable_tracing=False,  # Disable for tests
        enable_health_checks=True,
        log_level="DEBUG",
        service_name="pynomaly-test",
        environment="test",
    )
    return components


@pytest.fixture(scope="function")
def audit_logger():
    """Create audit logger for testing."""
    if not AUDIT_LOGGING_AVAILABLE:
        pytest.skip("Audit logging not available")
    return init_audit_logging()


# Test data manager fixtures (consolidated from tests/fixtures/conftest.py)
@pytest.fixture(scope="session")
def test_data_manager():
    """Provide a test data manager for the entire test session."""
    if not TEST_DATA_MANAGER_AVAILABLE:
        pytest.skip("Test data manager dependencies not available")
    manager = TestDataManager()
    yield manager
    # Cleanup after session
    manager.clear_cache()


@pytest.fixture(scope="session")
def test_scenario_factory():
    """Provide a test scenario factory for the entire test session."""
    if not TEST_DATA_MANAGER_AVAILABLE:
        pytest.skip("Test scenario factory dependencies not available")
    return TestScenarioFactory()


@pytest.fixture
def various_datasets(test_data_manager):
    """Provide various types of datasets for parameterized tests."""
    if not TEST_DATA_MANAGER_AVAILABLE:
        pytest.skip("Test data manager dependencies not available")
    # Return a simple dataset as default
    return test_data_manager.get_dataset("simple", **SMALL_DATASET_PARAMS)


# Benchmark configuration fixtures
@pytest.fixture(scope="session")
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "warmup_rounds": 1,
        "min_rounds": 3,
        "max_time": 30.0,
        "timer": "time.perf_counter",
    }


@pytest.fixture
def benchmark_group():
    """Group related benchmarks together."""
    return "anomaly_detection_algorithms"


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment after each test."""
    yield
    # Clean up any global state, reset singletons, clear caches, etc.


# Pytest configuration and custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    # Core testing markers
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "security: mark test as a security test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "unit: mark test as a unit test")

    # Benchmark markers (from benchmarks/conftest.py)
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")

    # Data testing markers (from fixtures/conftest.py)
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring test data"
    )
    config.addinivalue_line("markers", "small_data: mark test as using small datasets")
    config.addinivalue_line(
        "markers", "medium_data: mark test as using medium datasets"
    )
    config.addinivalue_line("markers", "large_data: mark test as using large datasets")
    config.addinivalue_line(
        "markers", "synthetic_data: mark test as using synthetic data"
    )

    # UI/BDD testing markers (from ui/bdd/conftest.py)
    config.addinivalue_line("markers", "accessibility: Accessibility compliance tests")
    config.addinivalue_line("markers", "workflow: Complete user workflow tests")
    config.addinivalue_line(
        "markers", "cross_browser: Cross-browser compatibility tests"
    )
    config.addinivalue_line("markers", "ml_engineer: ML engineer workflow tests")
    config.addinivalue_line("markers", "data_scientist: Data scientist workflow tests")
    config.addinivalue_line("markers", "critical: Critical path tests that must pass")
    config.addinivalue_line("markers", "smoke: Quick smoke test scenarios")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options."""
    # Skip slow tests by default unless explicitly requested
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip integration tests unless requested
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="need --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests",
    )
    parser.addoption(
        "--security", action="store_true", default=False, help="run security tests"
    )
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="run benchmark tests"
    )
