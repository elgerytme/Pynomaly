"""Test configuration and fixtures for infrastructure package."""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, MagicMock

from infrastructure.configuration import InfrastructureConfig, set_config
from infrastructure.configuration.environment import Environment


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> InfrastructureConfig:
    """Create test configuration."""
    config = InfrastructureConfig()
    config.environment = "testing"
    config.debug = True
    config.testing = True
    
    # Override database with in-memory SQLite
    config.database.url = "sqlite:///:memory:"
    
    # Override Redis with fake Redis
    config.redis.url = "redis://localhost:6379/15"  # Use different DB for tests
    
    # Override secrets
    config.security.secret_key = "test-secret-key-for-testing-only"
    
    return config


@pytest.fixture
def mock_database():
    """Mock database connection."""
    return Mock()


@pytest.fixture
def mock_redis():
    """Mock Redis connection."""
    return Mock()


@pytest.fixture 
def mock_event_bus():
    """Mock event bus."""
    return Mock()


@pytest.fixture
def mock_cache():
    """Mock cache manager."""
    return Mock()


@pytest.fixture
def mock_logger():
    """Mock logger."""
    return Mock()


@pytest.fixture
def mock_metrics():
    """Mock metrics collector."""
    return Mock()


@pytest.fixture(autouse=True)
def setup_test_environment(test_config: InfrastructureConfig):
    """Setup test environment for each test."""
    # Set test configuration globally
    set_config(test_config)
    
    yield
    
    # Cleanup after test
    # Reset to default configuration
    set_config(InfrastructureConfig())


@pytest.fixture
async def async_mock():
    """Create an async mock."""
    mock = MagicMock()
    mock.__aenter__ = MagicMock(return_value=mock)
    mock.__aexit__ = MagicMock(return_value=None)
    return mock


class MockAsyncContextManager:
    """Mock async context manager for testing."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.fixture
def mock_async_context():
    """Mock async context manager."""
    return MockAsyncContextManager


# Performance testing fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "iterations": 1000,
        "timeout": 30,
        "warmup_iterations": 100
    }


# Integration test fixtures
@pytest.fixture(scope="session")
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "database_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/15",
        "enable_real_services": False,
        "timeout": 60
    }


# Security test fixtures
@pytest.fixture
def security_test_data():
    """Test data for security tests."""
    return {
        "valid_jwt_secret": "secure-secret-key-for-jwt-testing-only-32-chars",
        "weak_passwords": ["123456", "password", "admin", "test"],
        "strong_passwords": ["StrongP@ssw0rd!", "Complex#Pass2023", "Secure*Key&789"],
        "test_user_id": "test-user-123",
        "test_permissions": ["read", "write", "admin"]
    }


# Monitoring test fixtures
@pytest.fixture
def mock_telemetry():
    """Mock telemetry and monitoring."""
    telemetry = Mock()
    telemetry.start_span = Mock(return_value=MockAsyncContextManager())
    telemetry.record_metric = Mock()
    telemetry.log_event = Mock()
    return telemetry


# Caching test fixtures  
@pytest.fixture
def cache_test_data():
    """Test data for caching tests."""
    return {
        "test_keys": ["user:123", "session:abc", "data:xyz"],
        "test_values": [
            {"id": 123, "name": "Test User"},
            {"session_id": "abc", "user_id": 123},
            {"data": "test data", "timestamp": "2023-01-01T00:00:00Z"}
        ],
        "test_ttl": 300  # 5 minutes
    }


# Messaging test fixtures
@pytest.fixture
def message_test_data():
    """Test data for messaging tests."""
    return {
        "test_events": [
            {"event_type": "user.created", "user_id": "123"},
            {"event_type": "order.placed", "order_id": "456"},
            {"event_type": "payment.processed", "payment_id": "789"}
        ],
        "test_queues": ["default", "priority", "notifications"],
        "test_routing_keys": ["user.events", "order.events", "payment.events"]
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup resources after each test."""
    yield
    
    # Cleanup any lingering connections, caches, etc.
    # This runs after each test to ensure clean state
    pass