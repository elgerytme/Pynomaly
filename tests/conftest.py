"""Pytest configuration and fixtures for comprehensive testing."""

from __future__ import annotations

import asyncio
import os
import tempfile
import shutil
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import numpy as np
import pytest
from dependency_injector import providers
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.infrastructure.config import Container, Settings
from pynomaly.infrastructure.auth.jwt_auth import init_auth
from pynomaly.infrastructure.observability import setup_observability
from pynomaly.infrastructure.security.audit_logging import init_audit_logging

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
    from pynomaly.presentation.api.app import create_app
    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False


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
    return Settings(
        app={
            "name": "pynomaly-test",
            "version": "0.1.0-test",
            "environment": "test",
            "debug": True
        },
        database_url="sqlite:///:memory:",
        secret_key="test-secret-key-not-for-production-use-only",
        jwt_algorithm="HS256",
        jwt_expiration=3600,
        auth_enabled=True,
        cache_enabled=False,
        docs_enabled=True,
        monitoring={
            "metrics_enabled": False,
            "tracing_enabled": False,
            "prometheus_enabled": False
        },
        debug=True,
        environment="test",
        storage_path="/tmp/pynomaly_test",
        log_level="DEBUG",
    )


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
        roles=["user"]
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
        data=sample_data.drop(columns=["target"]),
        description="Test dataset for unit tests",
        target_column="target",
        features=[f"feature_{i}" for i in range(5)],
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
        features=[f"feature_{i}" for i in range(n_features)],
        metadata={"test": True, "performance": True}
    )


@pytest.fixture
def sample_detector() -> Detector:
    """Enhanced sample detector for testing."""
    return Detector(
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
    
    return DetectionResult(
        detector_id=sample_detector.id,
        dataset_id=sample_dataset.id,
        scores=scores,
        metadata={"test": True, "model_version": "1.0"}
    )


@pytest.fixture
def trained_detector(
    sample_detector: Detector, sample_dataset: Dataset, container: Container
) -> Detector:
    """Trained detector for testing."""
    try:
        # Get adapter and train
        adapter = container.pyod_adapter()
        model = adapter.create_model(sample_detector.algorithm_name, sample_detector.parameters)
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
        "../../../../windows/system32/config/sam"
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
    components = setup_observability(
        enable_logging=True,
        enable_metrics=False,  # Disable for tests
        enable_tracing=False,  # Disable for tests
        enable_health_checks=True,
        log_level="DEBUG",
        service_name="pynomaly-test",
        environment="test"
    )
    return components


@pytest.fixture(scope="function")
def audit_logger():
    """Create audit logger for testing."""
    return init_audit_logging()


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment after each test."""
    yield
    # Clean up any global state, reset singletons, clear caches, etc.


# Pytest configuration and custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )


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
        "--integration", action="store_true", default=False, help="run integration tests"
    )
    parser.addoption(
        "--performance", action="store_true", default=False, help="run performance tests"
    )
    parser.addoption(
        "--security", action="store_true", default=False, help="run security tests"
    )
