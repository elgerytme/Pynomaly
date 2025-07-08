"""Mock external clients for testing purposes using pytest-mock."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_http_client(mocker):
    """Mock for external HTTP client."""
    return mocker.patch("httpx.Client")


@pytest.fixture
def mock_db_client(mocker):
    """Mock for external database client."""
    return MagicMock(name="DBClient")


@pytest.fixture
def mock_redis_client(mocker):
    """Mock for external Redis client."""
    return mocker.patch("redis.Redis")


@pytest.fixture
def mock_prometheus_client(mocker):
    """Mock for Prometheus client."""
    return MagicMock(name="PrometheusClient")
