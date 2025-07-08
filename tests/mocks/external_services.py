"""Mock external services for testing purposes."""

from typing import Any
from unittest.mock import MagicMock

import pytest


class MockHttpClient:
    """Mock HTTP client for external API calls."""

    def __init__(self):
        self.responses = {}
        self.call_history = []

    def get(self, url: str, **kwargs) -> MagicMock:
        """Mock GET request."""
        self.call_history.append(("GET", url, kwargs))
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = self.responses.get(url, {})
        return response

    def post(self, url: str, **kwargs) -> MagicMock:
        """Mock POST request."""
        self.call_history.append(("POST", url, kwargs))
        response = MagicMock()
        response.status_code = 201
        response.json.return_value = {"status": "success"}
        return response

    def put(self, url: str, **kwargs) -> MagicMock:
        """Mock PUT request."""
        self.call_history.append(("PUT", url, kwargs))
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"status": "updated"}
        return response

    def delete(self, url: str, **kwargs) -> MagicMock:
        """Mock DELETE request."""
        self.call_history.append(("DELETE", url, kwargs))
        response = MagicMock()
        response.status_code = 204
        return response

    def set_response(self, url: str, response_data: dict[str, Any]) -> None:
        """Set mock response for a specific URL."""
        self.responses[url] = response_data


class MockRedisClient:
    """Mock Redis client for caching operations."""

    def __init__(self):
        self._data = {}
        self._expiry = {}

    def get(self, key: str) -> str | None:
        """Mock get operation."""
        return self._data.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        """Mock set operation."""
        self._data[key] = value
        if ex:
            self._expiry[key] = ex
        return True

    def delete(self, key: str) -> int:
        """Mock delete operation."""
        if key in self._data:
            del self._data[key]
            self._expiry.pop(key, None)
            return 1
        return 0

    def exists(self, key: str) -> bool:
        """Mock exists operation."""
        return key in self._data

    def flushall(self) -> bool:
        """Mock flush all operation."""
        self._data.clear()
        self._expiry.clear()
        return True

    def keys(self, pattern: str = "*") -> list[str]:
        """Mock keys operation."""
        if pattern == "*":
            return list(self._data.keys())
        # Simple pattern matching
        import re

        regex_pattern = pattern.replace("*", ".*")
        return [key for key in self._data.keys() if re.match(regex_pattern, key)]


class MockDatabaseClient:
    """Mock database client for persistence operations."""

    def __init__(self):
        self._tables = {}
        self.transaction_active = False

    def execute(self, query: str, params: dict[str, Any] | None = None) -> MagicMock:
        """Mock query execution."""
        result = MagicMock()
        result.rowcount = 1
        result.fetchall.return_value = []
        result.fetchone.return_value = None
        return result

    def begin_transaction(self) -> None:
        """Mock begin transaction."""
        self.transaction_active = True

    def commit(self) -> None:
        """Mock commit transaction."""
        self.transaction_active = False

    def rollback(self) -> None:
        """Mock rollback transaction."""
        self.transaction_active = False

    def create_table(self, table_name: str, schema: dict[str, str]) -> bool:
        """Mock create table."""
        self._tables[table_name] = {"schema": schema, "data": []}
        return True

    def insert(self, table_name: str, data: dict[str, Any]) -> bool:
        """Mock insert operation."""
        if table_name in self._tables:
            self._tables[table_name]["data"].append(data)
            return True
        return False

    def select(
        self, table_name: str, conditions: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Mock select operation."""
        if table_name not in self._tables:
            return []

        data = self._tables[table_name]["data"]
        if not conditions:
            return data

        # Simple filtering
        filtered_data = []
        for row in data:
            match = True
            for key, value in conditions.items():
                if row.get(key) != value:
                    match = False
                    break
            if match:
                filtered_data.append(row)

        return filtered_data


class MockPrometheusClient:
    """Mock Prometheus client for metrics collection."""

    def __init__(self):
        self.metrics = {}
        self.counters = {}
        self.gauges = {}
        self.histograms = {}

    def inc_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Mock counter increment."""
        key = f"{name}_{labels or {}}"
        self.counters[key] = self.counters.get(key, 0) + value

    def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Mock gauge set."""
        key = f"{name}_{labels or {}}"
        self.gauges[key] = value

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Mock histogram observation."""
        key = f"{name}_{labels or {}}"
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)

    def get_metric_value(
        self, name: str, labels: dict[str, str] | None = None
    ) -> float | None:
        """Get mock metric value."""
        key = f"{name}_{labels or {}}"
        return self.counters.get(key) or self.gauges.get(key)

    def get_histogram_buckets(
        self, name: str, labels: dict[str, str] | None = None
    ) -> list[float]:
        """Get mock histogram values."""
        key = f"{name}_{labels or {}}"
        return self.histograms.get(key, [])


class MockS3Client:
    """Mock S3 client for object storage operations."""

    def __init__(self):
        self._buckets = {}

    def create_bucket(self, bucket_name: str) -> bool:
        """Mock create bucket."""
        self._buckets[bucket_name] = {}
        return True

    def put_object(self, bucket_name: str, key: str, data: bytes) -> bool:
        """Mock put object."""
        if bucket_name not in self._buckets:
            self.create_bucket(bucket_name)
        self._buckets[bucket_name][key] = data
        return True

    def get_object(self, bucket_name: str, key: str) -> bytes | None:
        """Mock get object."""
        return self._buckets.get(bucket_name, {}).get(key)

    def delete_object(self, bucket_name: str, key: str) -> bool:
        """Mock delete object."""
        if bucket_name in self._buckets and key in self._buckets[bucket_name]:
            del self._buckets[bucket_name][key]
            return True
        return False

    def list_objects(self, bucket_name: str, prefix: str = "") -> list[str]:
        """Mock list objects."""
        if bucket_name not in self._buckets:
            return []

        keys = list(self._buckets[bucket_name].keys())
        if prefix:
            keys = [key for key in keys if key.startswith(prefix)]

        return keys


class MockMessageQueue:
    """Mock message queue for async communication."""

    def __init__(self):
        self.queues = {}

    def create_queue(self, queue_name: str) -> bool:
        """Mock create queue."""
        self.queues[queue_name] = []
        return True

    def send_message(self, queue_name: str, message: dict[str, Any]) -> bool:
        """Mock send message."""
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        self.queues[queue_name].append(message)
        return True

    def receive_message(self, queue_name: str) -> dict[str, Any] | None:
        """Mock receive message."""
        if queue_name in self.queues and self.queues[queue_name]:
            return self.queues[queue_name].pop(0)
        return None

    def get_queue_size(self, queue_name: str) -> int:
        """Mock get queue size."""
        return len(self.queues.get(queue_name, []))


# Pytest fixtures for easy use in tests
@pytest.fixture
def mock_http_client():
    """Provide mock HTTP client."""
    return MockHttpClient()


@pytest.fixture
def mock_redis_client():
    """Provide mock Redis client."""
    return MockRedisClient()


@pytest.fixture
def mock_database_client():
    """Provide mock database client."""
    return MockDatabaseClient()


@pytest.fixture
def mock_prometheus_client():
    """Provide mock Prometheus client."""
    return MockPrometheusClient()


@pytest.fixture
def mock_s3_client():
    """Provide mock S3 client."""
    return MockS3Client()


@pytest.fixture
def mock_message_queue():
    """Provide mock message queue."""
    return MockMessageQueue()


@pytest.fixture
def all_external_mocks(
    mock_http_client,
    mock_redis_client,
    mock_database_client,
    mock_prometheus_client,
    mock_s3_client,
    mock_message_queue,
):
    """Provide all external service mocks."""
    return {
        "http": mock_http_client,
        "redis": mock_redis_client,
        "database": mock_database_client,
        "prometheus": mock_prometheus_client,
        "s3": mock_s3_client,
        "message_queue": mock_message_queue,
    }
