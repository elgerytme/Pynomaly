"""Global pytest configuration and fixtures."""

from __future__ import annotations

import pytest
import asyncio
from pathlib import Path
from typing import Generator, Any
from unittest.mock import Mock

from test_framework.core.database import Database
from test_framework.models.user import User
from test_framework.services.user_service import UserService


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_database() -> Generator[Database, None, None]:
    """Create test database for the session."""
    db = Database("sqlite:///:memory:")
    db.create_tables()
    yield db
    db.close()


@pytest.fixture
def database(test_database: Database) -> Generator[Database, None, None]:
    """Provide clean database for each test."""
    test_database.begin_transaction()
    yield test_database
    test_database.rollback()


@pytest.fixture
def user_service(database: Database) -> UserService:
    """Create user service with test database."""
    return UserService(database)


@pytest.fixture
def sample_user() -> User:
    """Create a sample user for testing."""
    return User(
        id=1,
        name="Test User",
        email="test@example.com",
        age=25,
        is_active=True
    )


@pytest.fixture
def sample_users() -> list[User]:
    """Create a list of sample users."""
    return [
        User(
            id=i,
            name=f"User {i}",
            email=f"user{i}@example.com",
            age=20 + i,
            is_active=True
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def temp_file(tmp_path: Path) -> Path:
    """Create a temporary file for testing."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("test content")
    return file_path


@pytest.fixture
def mock_email_service() -> Mock:
    """Mock email service for testing."""
    mock = Mock()
    mock.send_email.return_value = True
    mock.validate_email.return_value = True
    return mock


@pytest.fixture
def mock_external_api() -> Mock:
    """Mock external API for testing."""
    mock = Mock()
    mock.fetch_data.return_value = {"status": "success", "data": []}
    mock.post_data.return_value = {"status": "created", "id": 123}
    return mock


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


# Performance testing fixtures
@pytest.fixture
def performance_data() -> dict[str, Any]:
    """Sample data for performance testing."""
    return {
        "items": [{"id": i, "value": f"item_{i}"} for i in range(1000)],
        "large_text": "x" * 10000,
        "nested_data": {
            f"key_{i}": {"data": [j for j in range(100)]}
            for i in range(50)
        }
    }


# Async fixtures
@pytest.fixture
async def async_user_service(database: Database) -> UserService:
    """Create async user service with test database."""
    service = UserService(database)
    await service.initialize()
    return service


# Parametrized fixtures
@pytest.fixture(params=[
    {"name": "John", "age": 25},
    {"name": "Jane", "age": 30},
    {"name": "Bob", "age": 35},
])
def user_data(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parametrized user data for testing."""
    return request.param


# Custom markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )


# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data() -> Generator[None, None, None]:
    """Cleanup test data after each test."""
    yield
    # Cleanup logic here
    # e.g., clear caches, reset global state, etc.