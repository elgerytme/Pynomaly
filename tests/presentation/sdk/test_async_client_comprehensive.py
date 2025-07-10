"""
Comprehensive test suite for Pynomaly SDK asynchronous client.

This module provides extensive testing coverage for the asynchronous client implementation,
including async/await patterns, concurrent operations, and async error handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponse, ClientTimeout
from aiohttp.client_exceptions import ClientConnectorError, ServerTimeoutError

from pynomaly.presentation.sdk.async_client import AsyncPynomalyClient
from pynomaly.presentation.sdk.exceptions import (
    AuthenticationError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from pynomaly.presentation.sdk.models import (
    DatasetInfo,
    DetectionRequest,
    DetectionResponse,
    HealthStatus,
    ModelInfo,
    TrainingRequest,
    TrainingResponse,
)


class TestAsyncPynomalyClientInitialization:
    """Test async client initialization and configuration."""

    def test_async_client_initialization_default(self):
        """Test async client initialization with default parameters."""
        client = AsyncPynomalyClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.retry_delay == 1.0
        assert client.api_key is None
        assert client.session is None  # Session created on demand

    def test_async_client_initialization_custom_parameters(self):
        """Test async client initialization with custom parameters."""
        client = AsyncPynomalyClient(
            base_url="https://api.example.com",
            api_key="test-key",
            timeout=60.0,
            max_retries=5,
            retry_delay=2.0,
        )
        assert client.base_url == "https://api.example.com"
        assert client.api_key == "test-key"
        assert client.timeout == 60.0
        assert client.max_retries == 5
        assert client.retry_delay == 2.0

    def test_async_client_initialization_from_config(self):
        """Test async client initialization from configuration."""
        config = {
            "base_url": "https://prod.example.com",
            "api_key": "prod-key",
            "timeout": 45.0,
            "max_retries": 4,
        }
        client = AsyncPynomalyClient.from_config(config)
        assert client.base_url == "https://prod.example.com"
        assert client.api_key == "prod-key"
        assert client.timeout == 45.0
        assert client.max_retries == 4

    def test_async_client_headers_setup(self):
        """Test proper headers setup."""
        client = AsyncPynomalyClient(api_key="test-key")
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"].startswith("pynomaly-sdk/")

    def test_async_client_headers_without_api_key(self):
        """Test headers setup without API key."""
        client = AsyncPynomalyClient()
        headers = client._get_headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"


class TestAsyncPynomalyClientHealthCheck:
    """Test async health check functionality."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_health_check_success(self, mock_get):
        """Test successful async health check."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2023-01-01T00:00:00Z",
            }
        )
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient()
        async with client:
            health = await client.health_check()

        assert isinstance(health, HealthStatus)
        assert health.status == "healthy"
        assert health.version == "1.0.0"
        mock_get.assert_called_once_with(
            "http://localhost:8000/health",
            timeout=ClientTimeout(total=30.0),
            headers=client._get_headers(),
        )

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_health_check_failure(self, mock_get):
        """Test async health check failure."""
        mock_get.side_effect = ClientConnectorError(
            connection_key=None, os_error=OSError("Connection failed")
        )

        client = AsyncPynomalyClient()
        async with client:
            with pytest.raises(NetworkError):
                await client.health_check()

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_health_check_timeout(self, mock_get):
        """Test async health check timeout."""
        mock_get.side_effect = ServerTimeoutError("Request timeout")

        client = AsyncPynomalyClient()
        async with client:
            with pytest.raises(NetworkError):
                await client.health_check()


class TestAsyncPynomalyClientAuthentication:
    """Test async authentication functionality."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_login_success(self, mock_post):
        """Test successful async login."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "access_token": "jwt-token",
                "token_type": "bearer",
                "expires_in": 3600,
            }
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient()
        async with client:
            token = await client.login("user@example.com", "password123")

        assert token == "jwt-token"
        assert client.api_key == "jwt-token"
        mock_post.assert_called_once_with(
            "http://localhost:8000/api/v1/auth/login",
            json={"email": "user@example.com", "password": "password123"},
            timeout=ClientTimeout(total=30.0),
            headers=client._get_headers(),
        )

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_login_invalid_credentials(self, mock_post):
        """Test async login with invalid credentials."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 401
        mock_response.json = AsyncMock(return_value={"detail": "Invalid credentials"})
        mock_post.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient()
        async with client:
            with pytest.raises(AuthenticationError):
                await client.login("user@example.com", "wrong-password")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_logout_success(self, mock_post):
        """Test successful async logout."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        async with client:
            await client.logout()

        assert client.api_key is None
        mock_post.assert_called_once_with(
            "http://localhost:8000/api/v1/auth/logout",
            timeout=ClientTimeout(total=30.0),
            headers={
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json",
                "User-Agent": client._get_headers()["User-Agent"],
            },
        )

    @pytest.mark.asyncio
    async def test_logout_without_token(self):
        """Test async logout without authentication token."""
        client = AsyncPynomalyClient()
        async with client:
            with pytest.raises(AuthenticationError):
                await client.logout()


class TestAsyncPynomalyClientDetection:
    """Test async detection functionality."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_detect_anomalies_success(self, mock_post):
        """Test successful async anomaly detection."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "anomaly_scores": [0.1, 0.9, 0.2],
                "anomaly_labels": [0, 1, 0],
                "execution_time": 0.123,
                "model_info": {"name": "isolation_forest", "version": "1.0"},
            }
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        request = DetectionRequest(
            data=[[1, 2], [3, 4], [5, 6]], algorithm="isolation_forest"
        )

        async with client:
            response = await client.detect_anomalies(request)

        assert isinstance(response, DetectionResponse)
        assert response.anomaly_scores == [0.1, 0.9, 0.2]
        assert response.anomaly_labels == [0, 1, 0]
        assert response.execution_time == 0.123

        mock_post.assert_called_once_with(
            "http://localhost:8000/api/v1/detection/detect",
            json=request.to_dict(),
            timeout=ClientTimeout(total=30.0),
            headers=client._get_headers(),
        )

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_detect_anomalies_validation_error(self, mock_post):
        """Test async detection with validation error."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 422
        mock_response.json = AsyncMock(
            return_value={"detail": "Invalid input data format"}
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        request = DetectionRequest(data=[], algorithm="invalid_algorithm")

        async with client:
            with pytest.raises(ValidationError):
                await client.detect_anomalies(request)

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_detect_anomalies_with_retry(self, mock_post):
        """Test async detection with retry logic."""
        # First call fails, second succeeds
        mock_response_fail = AsyncMock(spec=ClientResponse)
        mock_response_fail.status = 500
        mock_response_fail.json = AsyncMock(
            return_value={"detail": "Internal server error"}
        )

        mock_response_success = AsyncMock(spec=ClientResponse)
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(
            return_value={
                "anomaly_scores": [0.1, 0.2],
                "anomaly_labels": [0, 0],
                "execution_time": 0.1,
            }
        )

        mock_post.return_value.__aenter__.side_effect = [
            mock_response_fail,
            mock_response_success,
        ]

        client = AsyncPynomalyClient(api_key="test-token", max_retries=2)
        request = DetectionRequest(data=[[1, 2], [3, 4]], algorithm="isolation_forest")

        async with client:
            response = await client.detect_anomalies(request)

        assert isinstance(response, DetectionResponse)
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_concurrent_detection_requests(self, mock_post):
        """Test concurrent async detection requests."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "anomaly_scores": [0.1, 0.2],
                "anomaly_labels": [0, 0],
                "execution_time": 0.1,
            }
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        requests = [
            DetectionRequest(data=[[1, 2], [3, 4]], algorithm="isolation_forest")
            for _ in range(3)
        ]

        async with client:
            responses = await asyncio.gather(
                *[client.detect_anomalies(request) for request in requests]
            )

        assert len(responses) == 3
        assert all(isinstance(r, DetectionResponse) for r in responses)
        assert mock_post.call_count == 3


class TestAsyncPynomalyClientTraining:
    """Test async training functionality."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_train_model_success(self, mock_post):
        """Test successful async model training."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "job_id": "train-123",
                "status": "started",
                "model_id": "model-456",
                "estimated_duration": 300,
            }
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        request = TrainingRequest(
            data=[[1, 2], [3, 4], [5, 6]],
            algorithm="isolation_forest",
            hyperparameters={"n_estimators": 100},
        )

        async with client:
            response = await client.train_model(request)

        assert isinstance(response, TrainingResponse)
        assert response.job_id == "train-123"
        assert response.status == "started"
        assert response.model_id == "model-456"

        mock_post.assert_called_once_with(
            "http://localhost:8000/api/v1/training/train",
            json=request.to_dict(),
            timeout=ClientTimeout(total=30.0),
            headers=client._get_headers(),
        )

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_get_training_status(self, mock_get):
        """Test getting async training job status."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "job_id": "train-123",
                "status": "completed",
                "progress": 100,
                "model_id": "model-456",
            }
        )
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        async with client:
            status = await client.get_training_status("train-123")

        assert status["status"] == "completed"
        assert status["progress"] == 100

        mock_get.assert_called_once_with(
            "http://localhost:8000/api/v1/training/status/train-123",
            timeout=ClientTimeout(total=30.0),
            headers=client._get_headers(),
        )

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_wait_for_training_completion(self, mock_get):
        """Test waiting for async training completion."""
        # Mock progressive training status updates
        status_responses = [
            {"job_id": "train-123", "status": "running", "progress": 25},
            {"job_id": "train-123", "status": "running", "progress": 50},
            {"job_id": "train-123", "status": "running", "progress": 75},
            {
                "job_id": "train-123",
                "status": "completed",
                "progress": 100,
                "model_id": "model-456",
            },
        ]

        mock_responses = []
        for status_data in status_responses:
            mock_response = AsyncMock(spec=ClientResponse)
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=status_data)
            mock_responses.append(mock_response)

        mock_get.return_value.__aenter__.side_effect = mock_responses

        client = AsyncPynomalyClient(api_key="test-token")
        async with client:
            final_status = await client.wait_for_training_completion(
                "train-123", poll_interval=0.1
            )

        assert final_status["status"] == "completed"
        assert final_status["progress"] == 100
        assert final_status["model_id"] == "model-456"
        assert mock_get.call_count == 4


class TestAsyncPynomalyClientDatasets:
    """Test async dataset management functionality."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_upload_dataset_success(self, mock_post):
        """Test successful async dataset upload."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "dataset_id": "dataset-123",
                "name": "test_dataset",
                "size": 1000,
                "features": 5,
            }
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")

        with patch("aiofiles.open", mock_async_open(read_data="csv,data")):
            async with client:
                dataset_info = await client.upload_dataset("test.csv", "test_dataset")

        assert isinstance(dataset_info, DatasetInfo)
        assert dataset_info.dataset_id == "dataset-123"
        assert dataset_info.name == "test_dataset"
        assert dataset_info.size == 1000

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_list_datasets(self, mock_get):
        """Test listing async datasets."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "datasets": [
                    {"dataset_id": "dataset-1", "name": "dataset1", "size": 100},
                    {"dataset_id": "dataset-2", "name": "dataset2", "size": 200},
                ]
            }
        )
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        async with client:
            datasets = await client.list_datasets()

        assert len(datasets) == 2
        assert all(isinstance(d, DatasetInfo) for d in datasets)
        assert datasets[0].name == "dataset1"
        assert datasets[1].name == "dataset2"

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.delete")
    async def test_delete_dataset(self, mock_delete):
        """Test async dataset deletion."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_delete.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        async with client:
            await client.delete_dataset("dataset-123")

        mock_delete.assert_called_once_with(
            "http://localhost:8000/api/v1/datasets/dataset-123",
            timeout=ClientTimeout(total=30.0),
            headers=client._get_headers(),
        )


class TestAsyncPynomalyClientModels:
    """Test async model management functionality."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_list_models(self, mock_get):
        """Test listing async models."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "models": [
                    {
                        "model_id": "model-1",
                        "name": "model1",
                        "algorithm": "isolation_forest",
                    },
                    {
                        "model_id": "model-2",
                        "name": "model2",
                        "algorithm": "one_class_svm",
                    },
                ]
            }
        )
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        async with client:
            models = await client.list_models()

        assert len(models) == 2
        assert all(isinstance(m, ModelInfo) for m in models)
        assert models[0].algorithm == "isolation_forest"
        assert models[1].algorithm == "one_class_svm"

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_get_model_info(self, mock_get):
        """Test getting async model information."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "model_id": "model-123",
                "name": "test_model",
                "algorithm": "isolation_forest",
                "hyperparameters": {"n_estimators": 100},
                "metrics": {"accuracy": 0.95},
            }
        )
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        async with client:
            model_info = await client.get_model_info("model-123")

        assert isinstance(model_info, ModelInfo)
        assert model_info.model_id == "model-123"
        assert model_info.algorithm == "isolation_forest"
        assert model_info.hyperparameters["n_estimators"] == 100


class TestAsyncPynomalyClientErrorHandling:
    """Test async error handling and recovery."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_rate_limit_error(self, mock_get):
        """Test async rate limit error handling."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 429
        mock_response.json = AsyncMock(return_value={"detail": "Rate limit exceeded"})
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient()
        async with client:
            with pytest.raises(RateLimitError):
                await client.health_check()

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_not_found_error(self, mock_get):
        """Test async not found error handling."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 404
        mock_response.json = AsyncMock(return_value={"detail": "Resource not found"})
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        async with client:
            with pytest.raises(NotFoundError):
                await client.get_model_info("nonexistent-model")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_server_error(self, mock_get):
        """Test async server error handling."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value={"detail": "Internal server error"})
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(max_retries=1)
        async with client:
            with pytest.raises(ServerError):
                await client.health_check()

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_network_error(self, mock_get):
        """Test async network error handling."""
        mock_get.side_effect = ClientConnectorError(
            connection_key=None, os_error=OSError("Network unreachable")
        )

        client = AsyncPynomalyClient(max_retries=1)
        async with client:
            with pytest.raises(NetworkError):
                await client.health_check()

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_timeout_error(self, mock_get):
        """Test async timeout error handling."""
        mock_get.side_effect = ServerTimeoutError("Request timeout")

        client = AsyncPynomalyClient(timeout=1.0)
        async with client:
            with pytest.raises(NetworkError):
                await client.health_check()


class TestAsyncPynomalyClientContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager_success(self):
        """Test successful async context manager usage."""
        async with AsyncPynomalyClient() as client:
            assert client.session is not None
            assert hasattr(client, "_get_headers")

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup(self):
        """Test async context manager cleanup."""
        client = AsyncPynomalyClient()
        async with client:
            session = client.session
            assert session is not None

        # Session should be closed after context exit
        assert session.closed


class TestAsyncPynomalyClientConcurrency:
    """Test async client concurrency features."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_concurrent_health_checks(self, mock_get):
        """Test concurrent async health checks."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"status": "healthy", "version": "1.0.0"}
        )
        mock_get.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient()
        async with client:
            # Run multiple health checks concurrently
            health_checks = await asyncio.gather(
                *[client.health_check() for _ in range(5)]
            )

        assert len(health_checks) == 5
        assert all(isinstance(h, HealthStatus) for h in health_checks)
        assert all(h.status == "healthy" for h in health_checks)
        assert mock_get.call_count == 5

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_concurrent_training_jobs(self, mock_post):
        """Test concurrent async training jobs."""
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "job_id": "train-123",
                "status": "started",
                "model_id": "model-456",
            }
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        client = AsyncPynomalyClient(api_key="test-token")
        requests = [
            TrainingRequest(data=[[1, 2], [3, 4]], algorithm="isolation_forest")
            for _ in range(3)
        ]

        async with client:
            responses = await asyncio.gather(
                *[client.train_model(request) for request in requests]
            )

        assert len(responses) == 3
        assert all(isinstance(r, TrainingResponse) for r in responses)
        assert mock_post.call_count == 3


def mock_async_open(read_data=""):
    """Mock async open function for file operations."""
    mock_file = AsyncMock()
    mock_file.read = AsyncMock(return_value=read_data)
    mock_file.__aenter__ = AsyncMock(return_value=mock_file)
    mock_file.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=mock_file)


# Test fixtures
@pytest.fixture
def async_mock_client():
    """Create an async mock client for testing."""
    return AsyncPynomalyClient(api_key="test-token")


@pytest.fixture
def sample_async_detection_request():
    """Create a sample detection request for async testing."""
    return DetectionRequest(data=[[1, 2], [3, 4], [5, 6]], algorithm="isolation_forest")


@pytest.fixture
def sample_async_training_request():
    """Create a sample training request for async testing."""
    return TrainingRequest(
        data=[[1, 2], [3, 4], [5, 6]],
        algorithm="isolation_forest",
        hyperparameters={"n_estimators": 100},
    )
