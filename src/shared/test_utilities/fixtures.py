"""Common test fixtures for all packages."""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import Mock

import pytest
import structlog
from fastapi.testclient import TestClient
from httpx import AsyncClient


@pytest.fixture
def mock_logger() -> Mock:
    """Mock logger for testing."""
    return Mock(spec=structlog.BoundLogger)


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_data() -> dict:
    """Sample data for testing."""
    return {
        "id": 1,
        "name": "Test Data",
        "value": 42.0,
        "active": True,
        "metadata": {"source": "test", "version": "1.0.0"},
    }


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing."""
    async with AsyncClient() as client:
        yield client


@pytest.fixture
def sync_client() -> Generator[TestClient, None, None]:
    """Synchronous HTTP client for testing."""
    from fastapi import FastAPI
    
    app = FastAPI()
    
    @app.get("/health")
    def health():
        return {"status": "healthy"}
    
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_session():
    """Mock database session for testing."""
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    session.query = Mock()
    return session


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_client = Mock()
    redis_client.get = Mock(return_value=None)
    redis_client.set = Mock(return_value=True)
    redis_client.delete = Mock(return_value=1)
    redis_client.exists = Mock(return_value=False)
    return redis_client


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    s3_client = Mock()
    s3_client.upload_file = Mock()
    s3_client.download_file = Mock()
    s3_client.list_objects_v2 = Mock(return_value={"Contents": []})
    return s3_client


@pytest.fixture
def mock_external_service():
    """Mock external service for testing."""
    service = Mock()
    service.call = Mock(return_value={"status": "success"})
    service.is_healthy = Mock(return_value=True)
    return service