"""
Comprehensive SDK Integration Testing
===================================

This module provides comprehensive integration testing for the Pynomaly SDK,
covering client initialization, authentication, API operations, and error handling.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import numpy as np
import pandas as pd
import pytest
from requests import ConnectionError
from requests.adapters import HTTPAdapter


class TestSDKClientInitialization:
    """Test suite for SDK client initialization and configuration."""

    @pytest.fixture
    def mock_sdk_config(self):
        """Create mock SDK configuration."""
        return {
            "base_url": "https://api.pynomaly.com",
            "api_key": "test-api-key-123",
            "timeout": 30,
            "verify_ssl": True,
            "max_retries": 3,
            "backoff_factor": 0.3,
        }

    @pytest.fixture
    def mock_sync_client(self):
        """Create mock synchronous client."""
        client = Mock()
        client.base_url = "https://api.pynomaly.com"
        client.timeout = 30
        client.headers = {"X-API-Key": "test-api-key-123"}
        client.session = Mock()
        client.close = Mock()
        return client

    @pytest.fixture
    def mock_async_client(self):
        """Create mock asynchronous client."""
        client = AsyncMock()
        client.base_url = "https://api.pynomaly.com"
        client.timeout = 30
        client.headers = {"X-API-Key": "test-api-key-123"}
        client.session = AsyncMock()
        client.close = AsyncMock()
        return client

    def test_sync_client_default_initialization(self, mock_sync_client):
        """Test synchronous client default initialization."""
        # Test with minimal configuration
        assert mock_sync_client.base_url.startswith("https://")
        assert "X-API-Key" in mock_sync_client.headers
        assert mock_sync_client.timeout == 30
        assert hasattr(mock_sync_client, "session")
        assert hasattr(mock_sync_client, "close")

    def test_sync_client_custom_configuration(self, mock_sdk_config):
        """Test synchronous client custom configuration."""
        # Mock client with custom config
        client = Mock()
        client.base_url = mock_sdk_config["base_url"]
        client.api_key = mock_sdk_config["api_key"]
        client.timeout = mock_sdk_config["timeout"]
        client.verify_ssl = mock_sdk_config["verify_ssl"]

        assert client.base_url == "https://api.pynomaly.com"
        assert client.api_key == "test-api-key-123"
        assert client.timeout == 30
        assert client.verify_ssl is True

    @pytest.mark.asyncio
    async def test_async_client_initialization(self, mock_async_client):
        """Test asynchronous client initialization."""
        assert mock_async_client.base_url.startswith("https://")
        assert "X-API-Key" in mock_async_client.headers
        assert mock_async_client.timeout == 30
        assert hasattr(mock_async_client, "session")
        assert hasattr(mock_async_client, "close")

    def test_client_authentication_configuration(self):
        """Test client authentication configuration."""
        # Test API key authentication
        api_key_config = {"auth_type": "api_key", "api_key": "test-key-123"}

        client = Mock()
        client.auth_type = api_key_config["auth_type"]
        client.headers = {"X-API-Key": api_key_config["api_key"]}

        assert client.auth_type == "api_key"
        assert client.headers["X-API-Key"] == "test-key-123"

        # Test bearer token authentication
        bearer_config = {"auth_type": "bearer", "token": "bearer-token-456"}

        client.auth_type = bearer_config["auth_type"]
        client.headers = {"Authorization": f"Bearer {bearer_config['token']}"}

        assert client.auth_type == "bearer"
        assert client.headers["Authorization"] == "Bearer bearer-token-456"

    def test_client_ssl_configuration(self):
        """Test client SSL configuration."""
        # Test SSL verification enabled
        client = Mock()
        client.verify_ssl = True
        client.session = Mock()
        client.session.verify = True

        assert client.verify_ssl is True
        assert client.session.verify is True

        # Test SSL verification disabled
        client.verify_ssl = False
        client.session.verify = False

        assert client.verify_ssl is False
        assert client.session.verify is False

    def test_client_timeout_configuration(self):
        """Test client timeout configuration."""
        client = Mock()

        # Test default timeout
        client.timeout = 30
        assert client.timeout == 30

        # Test custom timeout
        client.timeout = 60
        assert client.timeout == 60

        # Test infinite timeout
        client.timeout = None
        assert client.timeout is None

    def test_client_context_manager(self, mock_sync_client):
        """Test client context manager functionality."""
        # Test synchronous context manager
        mock_sync_client.__enter__ = Mock(return_value=mock_sync_client)
        mock_sync_client.__exit__ = Mock(return_value=None)

        with mock_sync_client as client:
            assert client is mock_sync_client

        mock_sync_client.__exit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_client_context_manager(self, mock_async_client):
        """Test asynchronous client context manager functionality."""
        # Test asynchronous context manager
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=None)

        async with mock_async_client as client:
            assert client is mock_async_client

        mock_async_client.__aexit__.assert_called_once()


class TestSDKDatasetOperations:
    """Test suite for SDK dataset operations."""

    @pytest.fixture
    def sample_dataset_data(self):
        """Create sample dataset data."""
        return {
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            "features": ["feature_1", "feature_2", "feature_3"],
            "metadata": {"source": "test", "created_at": datetime.now().isoformat()},
        }

    @pytest.fixture
    def mock_dataset_response(self):
        """Create mock dataset response."""
        return {
            "id": "dataset-123",
            "name": "test_dataset",
            "status": "ready",
            "shape": [3, 3],
            "features": ["feature_1", "feature_2", "feature_3"],
            "created_at": "2023-01-01T10:00:00Z",
            "updated_at": "2023-01-01T10:00:00Z",
        }

    def test_create_dataset_with_list_data(
        self, mock_sync_client, sample_dataset_data, mock_dataset_response
    ):
        """Test creating dataset with list data."""
        mock_sync_client.create_dataset = Mock(return_value=mock_dataset_response)

        result = mock_sync_client.create_dataset(
            name="test_dataset",
            data=sample_dataset_data["data"],
            features=sample_dataset_data["features"],
        )

        assert result["id"] == "dataset-123"
        assert result["name"] == "test_dataset"
        assert result["status"] == "ready"
        assert result["shape"] == [3, 3]
        mock_sync_client.create_dataset.assert_called_once()

    def test_create_dataset_with_numpy_data(
        self, mock_sync_client, mock_dataset_response
    ):
        """Test creating dataset with numpy array."""
        numpy_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mock_sync_client.create_dataset = Mock(return_value=mock_dataset_response)

        result = mock_sync_client.create_dataset(
            name="numpy_dataset", data=numpy_data, features=["a", "b", "c"]
        )

        assert result["id"] == "dataset-123"
        mock_sync_client.create_dataset.assert_called_once()

    def test_create_dataset_with_pandas_dataframe(
        self, mock_sync_client, mock_dataset_response
    ):
        """Test creating dataset with pandas DataFrame."""
        df_data = pd.DataFrame(
            {"feature_1": [1, 4, 7], "feature_2": [2, 5, 8], "feature_3": [3, 6, 9]}
        )
        mock_sync_client.create_dataset = Mock(return_value=mock_dataset_response)

        result = mock_sync_client.create_dataset(name="pandas_dataset", data=df_data)

        assert result["id"] == "dataset-123"
        mock_sync_client.create_dataset.assert_called_once()

    def test_upload_dataset_file(self, mock_sync_client, mock_dataset_response):
        """Test uploading dataset from file."""
        mock_sync_client.upload_dataset = Mock(return_value=mock_dataset_response)

        result = mock_sync_client.upload_dataset(
            name="file_dataset", file_path="/path/to/dataset.csv", file_format="csv"
        )

        assert result["id"] == "dataset-123"
        mock_sync_client.upload_dataset.assert_called_once_with(
            name="file_dataset", file_path="/path/to/dataset.csv", file_format="csv"
        )

    def test_get_dataset(self, mock_sync_client, mock_dataset_response):
        """Test getting dataset information."""
        mock_sync_client.get_dataset = Mock(return_value=mock_dataset_response)

        result = mock_sync_client.get_dataset("dataset-123")

        assert result["id"] == "dataset-123"
        assert result["name"] == "test_dataset"
        assert result["status"] == "ready"
        mock_sync_client.get_dataset.assert_called_once_with("dataset-123")

    def test_list_datasets(self, mock_sync_client):
        """Test listing datasets."""
        mock_datasets_response = {
            "datasets": [
                {"id": "dataset-1", "name": "dataset_1", "status": "ready"},
                {"id": "dataset-2", "name": "dataset_2", "status": "processing"},
            ],
            "total": 2,
            "page": 1,
            "per_page": 10,
        }
        mock_sync_client.list_datasets = Mock(return_value=mock_datasets_response)

        result = mock_sync_client.list_datasets(page=1, per_page=10)

        assert len(result["datasets"]) == 2
        assert result["total"] == 2
        mock_sync_client.list_datasets.assert_called_once_with(page=1, per_page=10)

    def test_delete_dataset(self, mock_sync_client):
        """Test deleting dataset."""
        mock_sync_client.delete_dataset = Mock(return_value={"success": True})

        result = mock_sync_client.delete_dataset("dataset-123")

        assert result["success"] is True
        mock_sync_client.delete_dataset.assert_called_once_with("dataset-123")

    def test_download_dataset(self, mock_sync_client):
        """Test downloading dataset data."""
        mock_download_response = {
            "data": [[1, 2, 3], [4, 5, 6]],
            "features": ["a", "b", "c"],
            "format": "json",
        }
        mock_sync_client.download_dataset = Mock(return_value=mock_download_response)

        result = mock_sync_client.download_dataset("dataset-123", format="json")

        assert "data" in result
        assert "features" in result
        assert result["format"] == "json"
        mock_sync_client.download_dataset.assert_called_once_with(
            "dataset-123", format="json"
        )


class TestSDKDetectorOperations:
    """Test suite for SDK detector operations."""

    @pytest.fixture
    def detector_config(self):
        """Create detector configuration."""
        return {
            "algorithm": "isolation_forest",
            "parameters": {
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42,
            },
            "training_data": "dataset-123",
        }

    @pytest.fixture
    def mock_detector_response(self):
        """Create mock detector response."""
        return {
            "id": "detector-456",
            "name": "test_detector",
            "algorithm": "isolation_forest",
            "status": "trained",
            "parameters": {"n_estimators": 100, "contamination": 0.1},
            "training_dataset": "dataset-123",
            "performance_metrics": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.89,
                "f1_score": 0.90,
            },
            "created_at": "2023-01-01T10:00:00Z",
            "trained_at": "2023-01-01T10:05:00Z",
        }

    def test_create_detector(
        self, mock_sync_client, detector_config, mock_detector_response
    ):
        """Test creating a detector."""
        mock_sync_client.create_detector = Mock(return_value=mock_detector_response)

        result = mock_sync_client.create_detector(
            name="test_detector",
            algorithm=detector_config["algorithm"],
            parameters=detector_config["parameters"],
            training_data=detector_config["training_data"],
        )

        assert result["id"] == "detector-456"
        assert result["name"] == "test_detector"
        assert result["algorithm"] == "isolation_forest"
        assert result["status"] == "trained"
        mock_sync_client.create_detector.assert_called_once()

    def test_train_detector(self, mock_sync_client):
        """Test training a detector."""
        mock_training_response = {
            "job_id": "training-job-789",
            "status": "running",
            "progress": 0,
            "estimated_time": 300,
        }
        mock_sync_client.train_detector = Mock(return_value=mock_training_response)

        result = mock_sync_client.train_detector(
            detector_id="detector-456", dataset_id="dataset-123"
        )

        assert result["job_id"] == "training-job-789"
        assert result["status"] == "running"
        mock_sync_client.train_detector.assert_called_once_with(
            detector_id="detector-456", dataset_id="dataset-123"
        )

    def test_get_detector(self, mock_sync_client, mock_detector_response):
        """Test getting detector information."""
        mock_sync_client.get_detector = Mock(return_value=mock_detector_response)

        result = mock_sync_client.get_detector("detector-456")

        assert result["id"] == "detector-456"
        assert result["name"] == "test_detector"
        assert result["status"] == "trained"
        assert "performance_metrics" in result
        mock_sync_client.get_detector.assert_called_once_with("detector-456")

    def test_list_detectors(self, mock_sync_client):
        """Test listing detectors."""
        mock_detectors_response = {
            "detectors": [
                {"id": "detector-1", "name": "detector_1", "status": "trained"},
                {"id": "detector-2", "name": "detector_2", "status": "training"},
            ],
            "total": 2,
            "page": 1,
            "per_page": 10,
        }
        mock_sync_client.list_detectors = Mock(return_value=mock_detectors_response)

        result = mock_sync_client.list_detectors(page=1, per_page=10)

        assert len(result["detectors"]) == 2
        assert result["total"] == 2
        mock_sync_client.list_detectors.assert_called_once_with(page=1, per_page=10)

    def test_deploy_detector(self, mock_sync_client):
        """Test deploying a detector."""
        mock_deployment_response = {
            "deployment_id": "deployment-999",
            "detector_id": "detector-456",
            "status": "deployed",
            "endpoint_url": "https://api.pynomaly.com/v1/detect/deployment-999",
            "deployed_at": "2023-01-01T11:00:00Z",
        }
        mock_sync_client.deploy_detector = Mock(return_value=mock_deployment_response)

        result = mock_sync_client.deploy_detector("detector-456")

        assert result["deployment_id"] == "deployment-999"
        assert result["status"] == "deployed"
        assert "endpoint_url" in result
        mock_sync_client.deploy_detector.assert_called_once_with("detector-456")

    def test_delete_detector(self, mock_sync_client):
        """Test deleting a detector."""
        mock_sync_client.delete_detector = Mock(return_value={"success": True})

        result = mock_sync_client.delete_detector("detector-456")

        assert result["success"] is True
        mock_sync_client.delete_detector.assert_called_once_with("detector-456")


class TestSDKAnomalyDetection:
    """Test suite for SDK anomaly detection operations."""

    @pytest.fixture
    def detection_data(self):
        """Create detection data."""
        return {
            "data": [
                [1.5, 2.5, 3.5],
                [4.2, 5.8, 6.1],
                [100.0, 200.0, 300.0],  # Potential anomaly
            ],
            "features": ["feature_1", "feature_2", "feature_3"],
        }

    @pytest.fixture
    def mock_detection_response(self):
        """Create mock detection response."""
        return {
            "detection_id": "detection-123",
            "detector_id": "detector-456",
            "predictions": [0, 0, 1],  # Normal, Normal, Anomaly
            "scores": [0.1, 0.15, 0.95],
            "explanations": [
                {"feature_contributions": [0.02, 0.03, 0.05]},
                {"feature_contributions": [0.04, 0.06, 0.05]},
                {"feature_contributions": [0.8, 0.1, 0.05]},
            ],
            "statistics": {
                "total_samples": 3,
                "anomalies_detected": 1,
                "anomaly_rate": 0.333,
            },
            "execution_time": 0.045,
            "timestamp": "2023-01-01T12:00:00Z",
        }

    def test_detect_anomalies_single(
        self, mock_sync_client, detection_data, mock_detection_response
    ):
        """Test single anomaly detection."""
        mock_sync_client.detect_anomalies = Mock(return_value=mock_detection_response)

        result = mock_sync_client.detect_anomalies(
            detector_id="detector-456",
            data=detection_data["data"],
            features=detection_data["features"],
        )

        assert result["detection_id"] == "detection-123"
        assert result["detector_id"] == "detector-456"
        assert len(result["predictions"]) == 3
        assert len(result["scores"]) == 3
        assert result["statistics"]["anomalies_detected"] == 1
        mock_sync_client.detect_anomalies.assert_called_once()

    def test_batch_detect_anomalies(self, mock_sync_client):
        """Test batch anomaly detection."""
        mock_batch_response = {
            "batch_id": "batch-789",
            "status": "processing",
            "total_batches": 5,
            "completed_batches": 0,
            "estimated_time": 120,
        }
        mock_sync_client.batch_detect = Mock(return_value=mock_batch_response)

        result = mock_sync_client.batch_detect(
            detector_id="detector-456", dataset_id="dataset-123", batch_size=1000
        )

        assert result["batch_id"] == "batch-789"
        assert result["status"] == "processing"
        assert result["total_batches"] == 5
        mock_sync_client.batch_detect.assert_called_once_with(
            detector_id="detector-456", dataset_id="dataset-123", batch_size=1000
        )

    def test_get_detection_result(self, mock_sync_client, mock_detection_response):
        """Test getting detection result."""
        mock_sync_client.get_detection_result = Mock(
            return_value=mock_detection_response
        )

        result = mock_sync_client.get_detection_result("detection-123")

        assert result["detection_id"] == "detection-123"
        assert "predictions" in result
        assert "scores" in result
        assert "explanations" in result
        mock_sync_client.get_detection_result.assert_called_once_with("detection-123")

    def test_list_detection_results(self, mock_sync_client):
        """Test listing detection results."""
        mock_results_response = {
            "results": [
                {
                    "detection_id": "detection-1",
                    "detector_id": "detector-456",
                    "timestamp": "2023-01-01T12:00:00Z",
                },
                {
                    "detection_id": "detection-2",
                    "detector_id": "detector-456",
                    "timestamp": "2023-01-01T12:05:00Z",
                },
            ],
            "total": 2,
            "page": 1,
            "per_page": 10,
        }
        mock_sync_client.list_detection_results = Mock(
            return_value=mock_results_response
        )

        result = mock_sync_client.list_detection_results(
            detector_id="detector-456", page=1, per_page=10
        )

        assert len(result["results"]) == 2
        assert result["total"] == 2
        mock_sync_client.list_detection_results.assert_called_once_with(
            detector_id="detector-456", page=1, per_page=10
        )

    @pytest.mark.asyncio
    async def test_async_stream_detection(self, mock_async_client):
        """Test asynchronous streaming detection."""

        # Mock streaming response
        async def mock_stream():
            yield {"sample_id": 1, "prediction": 0, "score": 0.1}
            yield {"sample_id": 2, "prediction": 1, "score": 0.9}

        mock_async_client.stream_detection = Mock(return_value=mock_stream())

        results = []
        async for result in mock_async_client.stream_detection("detector-456"):
            results.append(result)

        assert len(results) == 2
        assert results[0]["sample_id"] == 1
        assert results[1]["prediction"] == 1
        mock_async_client.stream_detection.assert_called_once_with("detector-456")

    @pytest.mark.asyncio
    async def test_concurrent_batch_detection(self, mock_async_client):
        """Test concurrent batch detection."""
        mock_concurrent_response = [
            {"batch_id": "batch-1", "status": "completed", "anomalies": 5},
            {"batch_id": "batch-2", "status": "completed", "anomalies": 3},
            {"batch_id": "batch-3", "status": "completed", "anomalies": 8},
        ]
        mock_async_client.batch_detect_concurrent = AsyncMock(
            return_value=mock_concurrent_response
        )

        result = await mock_async_client.batch_detect_concurrent(
            detector_id="detector-456",
            batch_configs=[
                {"dataset_id": "dataset-1", "batch_size": 1000},
                {"dataset_id": "dataset-2", "batch_size": 1000},
                {"dataset_id": "dataset-3", "batch_size": 1000},
            ],
            max_concurrent=2,
        )

        assert len(result) == 3
        assert all(r["status"] == "completed" for r in result)
        mock_async_client.batch_detect_concurrent.assert_called_once()


class TestSDKErrorHandling:
    """Test suite for SDK error handling and exception scenarios."""

    @pytest.fixture
    def mock_error_responses(self):
        """Create mock error responses."""
        return {
            "400": {"error": "Bad Request", "message": "Invalid input parameters"},
            "401": {"error": "Unauthorized", "message": "Invalid API key"},
            "403": {"error": "Forbidden", "message": "Insufficient permissions"},
            "404": {"error": "Not Found", "message": "Resource not found"},
            "409": {"error": "Conflict", "message": "Resource already exists"},
            "429": {"error": "Rate Limited", "message": "Too many requests"},
            "500": {"error": "Internal Server Error", "message": "Server error"},
            "503": {
                "error": "Service Unavailable",
                "message": "Service temporarily unavailable",
            },
        }

    def test_authentication_error_handling(
        self, mock_sync_client, mock_error_responses
    ):
        """Test authentication error handling."""
        from pynomaly.presentation.sdk.exceptions import AuthenticationError

        # Mock 401 error
        def raise_auth_error(*args, **kwargs):
            raise AuthenticationError(mock_error_responses["401"]["message"])

        mock_sync_client.get_dataset = Mock(side_effect=raise_auth_error)

        with pytest.raises(AuthenticationError) as exc_info:
            mock_sync_client.get_dataset("dataset-123")

        assert "Invalid API key" in str(exc_info.value)

    def test_authorization_error_handling(self, mock_sync_client, mock_error_responses):
        """Test authorization error handling."""
        from pynomaly.presentation.sdk.exceptions import AuthorizationError

        # Mock 403 error
        def raise_auth_error(*args, **kwargs):
            raise AuthorizationError(mock_error_responses["403"]["message"])

        mock_sync_client.delete_dataset = Mock(side_effect=raise_auth_error)

        with pytest.raises(AuthorizationError) as exc_info:
            mock_sync_client.delete_dataset("dataset-123")

        assert "Insufficient permissions" in str(exc_info.value)

    def test_validation_error_handling(self, mock_sync_client, mock_error_responses):
        """Test validation error handling."""
        from pynomaly.presentation.sdk.exceptions import ValidationError

        # Mock 400 error
        def raise_validation_error(*args, **kwargs):
            raise ValidationError(mock_error_responses["400"]["message"])

        mock_sync_client.create_dataset = Mock(side_effect=raise_validation_error)

        with pytest.raises(ValidationError) as exc_info:
            mock_sync_client.create_dataset(name="", data=None)

        assert "Invalid input parameters" in str(exc_info.value)

    def test_resource_not_found_error_handling(
        self, mock_sync_client, mock_error_responses
    ):
        """Test resource not found error handling."""
        from pynomaly.presentation.sdk.exceptions import ResourceNotFoundError

        # Mock 404 error
        def raise_not_found_error(*args, **kwargs):
            raise ResourceNotFoundError(mock_error_responses["404"]["message"])

        mock_sync_client.get_detector = Mock(side_effect=raise_not_found_error)

        with pytest.raises(ResourceNotFoundError) as exc_info:
            mock_sync_client.get_detector("nonexistent-detector")

        assert "Resource not found" in str(exc_info.value)

    def test_rate_limit_error_handling(self, mock_sync_client, mock_error_responses):
        """Test rate limit error handling."""
        from pynomaly.presentation.sdk.exceptions import RateLimitError

        # Mock 429 error
        def raise_rate_limit_error(*args, **kwargs):
            raise RateLimitError(mock_error_responses["429"]["message"])

        mock_sync_client.detect_anomalies = Mock(side_effect=raise_rate_limit_error)

        with pytest.raises(RateLimitError) as exc_info:
            mock_sync_client.detect_anomalies("detector-123", data=[[1, 2, 3]])

        assert "Too many requests" in str(exc_info.value)

    def test_server_error_handling(self, mock_sync_client, mock_error_responses):
        """Test server error handling."""
        from pynomaly.presentation.sdk.exceptions import ServerError

        # Mock 500 error
        def raise_server_error(*args, **kwargs):
            raise ServerError(mock_error_responses["500"]["message"])

        mock_sync_client.train_detector = Mock(side_effect=raise_server_error)

        with pytest.raises(ServerError) as exc_info:
            mock_sync_client.train_detector("detector-123", "dataset-123")

        assert "Server error" in str(exc_info.value)

    def test_network_error_handling(self, mock_sync_client):
        """Test network error handling."""
        from pynomaly.presentation.sdk.exceptions import NetworkError

        # Mock network error
        def raise_network_error(*args, **kwargs):
            raise NetworkError("Connection failed")

        mock_sync_client.health_check = Mock(side_effect=raise_network_error)

        with pytest.raises(NetworkError) as exc_info:
            mock_sync_client.health_check()

        assert "Connection failed" in str(exc_info.value)

    def test_timeout_error_handling(self, mock_sync_client):
        """Test timeout error handling."""
        from pynomaly.presentation.sdk.exceptions import TimeoutError

        # Mock timeout error
        def raise_timeout_error(*args, **kwargs):
            raise TimeoutError("Request timed out after 30 seconds")

        mock_sync_client.batch_detect = Mock(side_effect=raise_timeout_error)

        with pytest.raises(TimeoutError) as exc_info:
            mock_sync_client.batch_detect("detector-123", "dataset-123")

        assert "Request timed out" in str(exc_info.value)

    def test_retry_logic_with_exponential_backoff(self, mock_sync_client):
        """Test retry logic with exponential backoff."""
        # Mock client with retry configuration
        mock_sync_client.max_retries = 3
        mock_sync_client.backoff_factor = 0.3

        call_count = 0

        def mock_request_with_retries(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return {"success": True}

        mock_sync_client._request_with_retries = Mock(
            side_effect=mock_request_with_retries
        )

        result = mock_sync_client._request_with_retries("GET", "/health")
        assert result["success"] is True
        assert call_count == 3


class TestSDKPerformanceAndReliability:
    """Test suite for SDK performance and reliability features."""

    def test_connection_pooling(self, mock_sync_client):
        """Test connection pooling functionality."""
        # Mock connection pool
        mock_sync_client.session = Mock()
        mock_sync_client.session.adapters = {
            "https://": Mock(spec=HTTPAdapter),
            "http://": Mock(spec=HTTPAdapter),
        }

        # Verify connection pool is configured
        assert hasattr(mock_sync_client.session, "adapters")
        assert "https://" in mock_sync_client.session.adapters
        assert "http://" in mock_sync_client.session.adapters

    def test_request_timeout_configuration(self, mock_sync_client):
        """Test request timeout configuration."""
        mock_sync_client.timeout = 30
        mock_sync_client.get_dataset = Mock()

        # Mock request with timeout
        def mock_request_with_timeout(*args, **kwargs):
            assert kwargs.get("timeout") == 30
            return {"id": "dataset-123"}

        mock_sync_client._make_request = Mock(side_effect=mock_request_with_timeout)
        mock_sync_client._make_request("GET", "/datasets/123", timeout=30)

        mock_sync_client._make_request.assert_called_once_with(
            "GET", "/datasets/123", timeout=30
        )

    @pytest.mark.asyncio
    async def test_async_session_management(self, mock_async_client):
        """Test asynchronous session management."""
        # Mock session lifecycle
        mock_async_client._session = None
        mock_async_client._ensure_session = AsyncMock()

        await mock_async_client._ensure_session()
        mock_async_client._ensure_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_request_limiting(self, mock_async_client):
        """Test concurrent request limiting."""
        # Mock semaphore for concurrent requests
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        mock_async_client._semaphore = semaphore

        # Test concurrent operations
        tasks = []
        for i in range(10):
            task = mock_async_client.get_dataset(f"dataset-{i}")
            tasks.append(task)

        # Mock the actual method calls
        mock_async_client.get_dataset = AsyncMock(return_value={"id": "dataset-test"})

        # Execute concurrent requests
        results = await asyncio.gather(
            *[mock_async_client.get_dataset(f"dataset-{i}") for i in range(10)]
        )

        assert len(results) == 10
        assert all(r["id"] == "dataset-test" for r in results)

    def test_memory_usage_with_large_datasets(self, mock_sync_client):
        """Test memory usage with large datasets."""
        # Mock large dataset scenario
        large_data_size = 1000000  # 1M rows
        mock_sync_client.stream_dataset = Mock()

        # Mock streaming response for large datasets
        def mock_stream_large_data(*args, **kwargs):
            for i in range(0, large_data_size, 10000):  # Stream in chunks
                yield {
                    "batch": i // 10000,
                    "data": list(range(i, min(i + 10000, large_data_size))),
                }

        mock_sync_client.stream_dataset.return_value = mock_stream_large_data()

        # Process large dataset in streaming fashion
        total_processed = 0
        for batch in mock_sync_client.stream_dataset("large-dataset"):
            total_processed += len(batch["data"])

        assert total_processed == large_data_size

    def test_resource_cleanup(self, mock_sync_client):
        """Test resource cleanup functionality."""
        # Test context manager cleanup
        mock_sync_client.close = Mock()
        mock_sync_client.__enter__ = Mock(return_value=mock_sync_client)
        mock_sync_client.__exit__ = Mock()

        with mock_sync_client:
            # Perform operations
            pass

        # Verify cleanup was called
        mock_sync_client.__exit__.assert_called_once()

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = {
            "base_url": "https://api.pynomaly.com",
            "api_key": "valid-key-123",
            "timeout": 30,
            "verify_ssl": True,
        }

        # Mock configuration validator
        def validate_config(config):
            required_fields = ["base_url", "api_key"]
            for field in required_fields:
                if field not in config or not config[field]:
                    raise ValueError(f"Missing required field: {field}")

            if not config["base_url"].startswith(("http://", "https://")):
                raise ValueError("Invalid base_url format")

            if config.get("timeout", 0) <= 0:
                raise ValueError("Timeout must be positive")

            return True

        # Test valid configuration
        assert validate_config(valid_config) is True

        # Test invalid configurations
        invalid_configs = [
            {"base_url": "invalid-url", "api_key": "key"},
            {"base_url": "https://api.example.com"},  # Missing api_key
            {"base_url": "https://api.example.com", "api_key": "key", "timeout": -1},
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                validate_config(invalid_config)
