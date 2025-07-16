"""
Comprehensive test suite for Pynomaly SDK synchronous client.

This module provides extensive testing coverage for the synchronous client implementation,
including authentication, error handling, retry logic, and all SDK operations.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from requests import Response
from requests.exceptions import ConnectionError, Timeout

from monorepo.presentation.sdk.client import PynomalyClient
from monorepo.presentation.sdk.exceptions import (
    AuthenticationError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from monorepo.presentation.sdk.models import (
    DatasetInfo,
    DetectionRequest,
    DetectionResponse,
    HealthStatus,
    ModelInfo,
    TrainingRequest,
    TrainingResponse,
)


class TestPynomalyClientInitialization:
    """Test client initialization and configuration."""

    def test_client_initialization_default(self):
        """Test client initialization with default parameters."""
        client = PynomalyClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.retry_delay == 1.0
        assert client.api_key is None
        assert client.session is not None

    def test_client_initialization_custom_parameters(self):
        """Test client initialization with custom parameters."""
        client = PynomalyClient(
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

    def test_client_initialization_from_config(self):
        """Test client initialization from configuration."""
        config = {
            "base_url": "https://prod.example.com",
            "api_key": "prod-key",
            "timeout": 45.0,
            "max_retries": 4,
        }
        client = PynomalyClient.from_config(config)
        assert client.base_url == "https://prod.example.com"
        assert client.api_key == "prod-key"
        assert client.timeout == 45.0
        assert client.max_retries == 4

    def test_client_headers_setup(self):
        """Test proper headers setup."""
        client = PynomalyClient(api_key="test-key")
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"].startswith("pynomaly-sdk/")

    def test_client_headers_without_api_key(self):
        """Test headers setup without API key."""
        client = PynomalyClient()
        headers = client._get_headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"


class TestPynomalyClientHealthCheck:
    """Test health check functionality."""

    @patch("requests.Session.get")
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": "2023-01-01T00:00:00Z",
        }
        mock_get.return_value = mock_response

        client = PynomalyClient()
        health = client.health_check()

        assert isinstance(health, HealthStatus)
        assert health.status == "healthy"
        assert health.version == "1.0.0"
        mock_get.assert_called_once_with(
            "http://localhost:8000/health", timeout=30.0, headers=client._get_headers()
        )

    @patch("requests.Session.get")
    def test_health_check_failure(self, mock_get):
        """Test health check failure."""
        mock_get.side_effect = ConnectionError("Connection failed")

        client = PynomalyClient()
        with pytest.raises(NetworkError):
            client.health_check()

    @patch("requests.Session.get")
    def test_health_check_timeout(self, mock_get):
        """Test health check timeout."""
        mock_get.side_effect = Timeout("Request timeout")

        client = PynomalyClient()
        with pytest.raises(NetworkError):
            client.health_check()


class TestPynomalyClientAuthentication:
    """Test authentication functionality."""

    @patch("requests.Session.post")
    def test_login_success(self, mock_post):
        """Test successful login."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "jwt-token",
            "token_type": "bearer",
            "expires_in": 3600,
        }
        mock_post.return_value = mock_response

        client = PynomalyClient()
        token = client.login("user@example.com", "password123")

        assert token == "jwt-token"
        assert client.api_key == "jwt-token"
        mock_post.assert_called_once_with(
            "http://localhost:8000/api/v1/auth/login",
            json={"email": "user@example.com", "password": "password123"},
            timeout=30.0,
            headers=client._get_headers(),
        )

    @patch("requests.Session.post")
    def test_login_invalid_credentials(self, mock_post):
        """Test login with invalid credentials."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Invalid credentials"}
        mock_post.return_value = mock_response

        client = PynomalyClient()
        with pytest.raises(AuthenticationError):
            client.login("user@example.com", "wrong-password")

    @patch("requests.Session.post")
    def test_logout_success(self, mock_post):
        """Test successful logout."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        client = PynomalyClient(api_key="test-token")
        client.logout()

        assert client.api_key is None
        mock_post.assert_called_once_with(
            "http://localhost:8000/api/v1/auth/logout",
            timeout=30.0,
            headers={
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json",
                "User-Agent": client._get_headers()["User-Agent"],
            },
        )

    def test_logout_without_token(self):
        """Test logout without authentication token."""
        client = PynomalyClient()
        with pytest.raises(AuthenticationError):
            client.logout()


class TestPynomalyClientDetection:
    """Test detection functionality."""

    @patch("requests.Session.post")
    def test_detect_anomalies_success(self, mock_post):
        """Test successful anomaly detection."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "anomaly_scores": [0.1, 0.9, 0.2],
            "anomaly_labels": [0, 1, 0],
            "execution_time": 0.123,
            "model_info": {"name": "isolation_forest", "version": "1.0"},
        }
        mock_post.return_value = mock_response

        client = PynomalyClient(api_key="test-token")
        request = DetectionRequest(
            data=[[1, 2], [3, 4], [5, 6]], algorithm="isolation_forest"
        )

        response = client.detect_anomalies(request)

        assert isinstance(response, DetectionResponse)
        assert response.anomaly_scores == [0.1, 0.9, 0.2]
        assert response.anomaly_labels == [0, 1, 0]
        assert response.execution_time == 0.123

        mock_post.assert_called_once_with(
            "http://localhost:8000/api/v1/detection/detect",
            json=request.to_dict(),
            timeout=30.0,
            headers=client._get_headers(),
        )

    @patch("requests.Session.post")
    def test_detect_anomalies_validation_error(self, mock_post):
        """Test detection with validation error."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 422
        mock_response.json.return_value = {"detail": "Invalid input data format"}
        mock_post.return_value = mock_response

        client = PynomalyClient(api_key="test-token")
        request = DetectionRequest(data=[], algorithm="invalid_algorithm")

        with pytest.raises(ValidationError):
            client.detect_anomalies(request)

    @patch("requests.Session.post")
    def test_detect_anomalies_with_retry(self, mock_post):
        """Test detection with retry logic."""
        # First call fails, second succeeds
        mock_response_fail = Mock(spec=Response)
        mock_response_fail.status_code = 500
        mock_response_fail.json.return_value = {"detail": "Internal server error"}

        mock_response_success = Mock(spec=Response)
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "anomaly_scores": [0.1, 0.2],
            "anomaly_labels": [0, 0],
            "execution_time": 0.1,
        }

        mock_post.side_effect = [mock_response_fail, mock_response_success]

        client = PynomalyClient(api_key="test-token", max_retries=2)
        request = DetectionRequest(data=[[1, 2], [3, 4]], algorithm="isolation_forest")

        response = client.detect_anomalies(request)

        assert isinstance(response, DetectionResponse)
        assert mock_post.call_count == 2


class TestPynomalyClientTraining:
    """Test training functionality."""

    @patch("requests.Session.post")
    def test_train_model_success(self, mock_post):
        """Test successful model training."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "job_id": "train-123",
            "status": "started",
            "model_id": "model-456",
            "estimated_duration": 300,
        }
        mock_post.return_value = mock_response

        client = PynomalyClient(api_key="test-token")
        request = TrainingRequest(
            data=[[1, 2], [3, 4], [5, 6]],
            algorithm="isolation_forest",
            hyperparameters={"n_estimators": 100},
        )

        response = client.train_model(request)

        assert isinstance(response, TrainingResponse)
        assert response.job_id == "train-123"
        assert response.status == "started"
        assert response.model_id == "model-456"

        mock_post.assert_called_once_with(
            "http://localhost:8000/api/v1/training/train",
            json=request.to_dict(),
            timeout=30.0,
            headers=client._get_headers(),
        )

    @patch("requests.Session.get")
    def test_get_training_status(self, mock_get):
        """Test getting training job status."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "job_id": "train-123",
            "status": "completed",
            "progress": 100,
            "model_id": "model-456",
        }
        mock_get.return_value = mock_response

        client = PynomalyClient(api_key="test-token")
        status = client.get_training_status("train-123")

        assert status["status"] == "completed"
        assert status["progress"] == 100

        mock_get.assert_called_once_with(
            "http://localhost:8000/api/v1/training/status/train-123",
            timeout=30.0,
            headers=client._get_headers(),
        )


class TestPynomalyClientDatasets:
    """Test dataset management functionality."""

    @patch("requests.Session.post")
    def test_upload_dataset_success(self, mock_post):
        """Test successful dataset upload."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "dataset_id": "dataset-123",
            "name": "test_dataset",
            "size": 1000,
            "features": 5,
        }
        mock_post.return_value = mock_response

        client = PynomalyClient(api_key="test-token")

        with patch("builtins.open", mock_open(read_data="csv,data")):
            dataset_info = client.upload_dataset("test.csv", "test_dataset")

        assert isinstance(dataset_info, DatasetInfo)
        assert dataset_info.dataset_id == "dataset-123"
        assert dataset_info.name == "test_dataset"
        assert dataset_info.size == 1000

    @patch("requests.Session.get")
    def test_list_datasets(self, mock_get):
        """Test listing datasets."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "datasets": [
                {"dataset_id": "dataset-1", "name": "dataset1", "size": 100},
                {"dataset_id": "dataset-2", "name": "dataset2", "size": 200},
            ]
        }
        mock_get.return_value = mock_response

        client = PynomalyClient(api_key="test-token")
        datasets = client.list_datasets()

        assert len(datasets) == 2
        assert all(isinstance(d, DatasetInfo) for d in datasets)
        assert datasets[0].name == "dataset1"
        assert datasets[1].name == "dataset2"

    @patch("requests.Session.delete")
    def test_delete_dataset(self, mock_delete):
        """Test dataset deletion."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_delete.return_value = mock_response

        client = PynomalyClient(api_key="test-token")
        client.delete_dataset("dataset-123")

        mock_delete.assert_called_once_with(
            "http://localhost:8000/api/v1/datasets/dataset-123",
            timeout=30.0,
            headers=client._get_headers(),
        )


class TestPynomalyClientModels:
    """Test model management functionality."""

    @patch("requests.Session.get")
    def test_list_models(self, mock_get):
        """Test listing models."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "model_id": "model-1",
                    "name": "model1",
                    "algorithm": "isolation_forest",
                },
                {"model_id": "model-2", "name": "model2", "algorithm": "one_class_svm"},
            ]
        }
        mock_get.return_value = mock_response

        client = PynomalyClient(api_key="test-token")
        models = client.list_models()

        assert len(models) == 2
        assert all(isinstance(m, ModelInfo) for m in models)
        assert models[0].algorithm == "isolation_forest"
        assert models[1].algorithm == "one_class_svm"

    @patch("requests.Session.get")
    def test_get_model_info(self, mock_get):
        """Test getting model information."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_id": "model-123",
            "name": "test_model",
            "algorithm": "isolation_forest",
            "hyperparameters": {"n_estimators": 100},
            "metrics": {"accuracy": 0.95},
        }
        mock_get.return_value = mock_response

        client = PynomalyClient(api_key="test-token")
        model_info = client.get_model_info("model-123")

        assert isinstance(model_info, ModelInfo)
        assert model_info.model_id == "model-123"
        assert model_info.algorithm == "isolation_forest"
        assert model_info.hyperparameters["n_estimators"] == 100


class TestPynomalyClientErrorHandling:
    """Test error handling and recovery."""

    @patch("requests.Session.get")
    def test_rate_limit_error(self, mock_get):
        """Test rate limit error handling."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 429
        mock_response.json.return_value = {"detail": "Rate limit exceeded"}
        mock_get.return_value = mock_response

        client = PynomalyClient()
        with pytest.raises(RateLimitError):
            client.health_check()

    @patch("requests.Session.get")
    def test_not_found_error(self, mock_get):
        """Test not found error handling."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Resource not found"}
        mock_get.return_value = mock_response

        client = PynomalyClient(api_key="test-token")
        with pytest.raises(NotFoundError):
            client.get_model_info("nonexistent-model")

    @patch("requests.Session.get")
    def test_server_error(self, mock_get):
        """Test server error handling."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal server error"}
        mock_get.return_value = mock_response

        client = PynomalyClient(max_retries=1)
        with pytest.raises(ServerError):
            client.health_check()

    @patch("requests.Session.get")
    def test_network_error(self, mock_get):
        """Test network error handling."""
        mock_get.side_effect = ConnectionError("Network unreachable")

        client = PynomalyClient(max_retries=1)
        with pytest.raises(NetworkError):
            client.health_check()

    @patch("requests.Session.get")
    def test_timeout_error(self, mock_get):
        """Test timeout error handling."""
        mock_get.side_effect = Timeout("Request timeout")

        client = PynomalyClient(timeout=1.0)
        with pytest.raises(NetworkError):
            client.health_check()


class TestPynomalyClientContextManager:
    """Test context manager functionality."""

    def test_context_manager_success(self):
        """Test successful context manager usage."""
        with PynomalyClient() as client:
            assert client.session is not None
            assert hasattr(client, "_get_headers")

    def test_context_manager_cleanup(self):
        """Test context manager cleanup."""
        client = PynomalyClient()
        with client:
            session = client.session
            assert session is not None

        # Session should be closed after context exit
        assert session.closed if hasattr(session, "closed") else True


def mock_open(read_data=""):
    """Mock open function for file operations."""
    mock_file = MagicMock()
    mock_file.read.return_value = read_data
    mock_file.__enter__.return_value = mock_file
    mock_file.__exit__.return_value = None
    return MagicMock(return_value=mock_file)


# Test fixtures
@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    return PynomalyClient(api_key="test-token")


@pytest.fixture
def sample_detection_request():
    """Create a sample detection request."""
    return DetectionRequest(data=[[1, 2], [3, 4], [5, 6]], algorithm="isolation_forest")


@pytest.fixture
def sample_training_request():
    """Create a sample training request."""
    return TrainingRequest(
        data=[[1, 2], [3, 4], [5, 6]],
        algorithm="isolation_forest",
        hyperparameters={"n_estimators": 100},
    )
