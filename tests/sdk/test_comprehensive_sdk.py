"""
Comprehensive SDK Testing Suite

Complete testing for the Python SDK, including client initialization, API interactions,
error handling, authentication, caching, and async operations.
"""

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import numpy as np
import pytest

from pynomaly.sdk import AsyncPynomalaClient, PynomalaClient
from pynomaly.sdk.exceptions import (
    AuthenticationError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    SDKError,
    ValidationError,
)


class TestSDKClientInitialization:
    """Test SDK client initialization and configuration."""

    def test_client_initialization_with_defaults(self):
        """Test client initialization with default settings."""
        client = PynomalaClient()

        assert client.base_url == "http://localhost:8000"
        assert client.api_version == "v1"
        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.api_key is None

    def test_client_initialization_with_custom_config(self):
        """Test client initialization with custom configuration."""
        config = {
            "base_url": "https://api.pynomaly.com",
            "api_version": "v2",
            "timeout": 60.0,
            "max_retries": 5,
            "api_key": "test_api_key_123",
        }

        client = PynomalaClient(**config)

        assert client.base_url == config["base_url"]
        assert client.api_version == config["api_version"]
        assert client.timeout == config["timeout"]
        assert client.max_retries == config["max_retries"]
        assert client.api_key == config["api_key"]

    def test_client_initialization_from_env(self):
        """Test client initialization from environment variables."""
        env_vars = {
            "PYNOMALY_BASE_URL": "https://env.pynomaly.com",
            "PYNOMALY_API_KEY": "env_api_key_456",
            "PYNOMALY_TIMEOUT": "45",
        }

        with patch.dict(os.environ, env_vars):
            client = PynomalaClient.from_env()

            assert client.base_url == env_vars["PYNOMALY_BASE_URL"]
            assert client.api_key == env_vars["PYNOMALY_API_KEY"]
            assert client.timeout == 45.0

    def test_client_initialization_from_config_file(self):
        """Test client initialization from configuration file."""
        config_data = {
            "base_url": "https://config.pynomaly.com",
            "api_key": "config_api_key_789",
            "timeout": 90,
            "max_retries": 2,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            client = PynomalaClient.from_config_file(config_file)

            assert client.base_url == config_data["base_url"]
            assert client.api_key == config_data["api_key"]
            assert client.timeout == config_data["timeout"]
            assert client.max_retries == config_data["max_retries"]
        finally:
            os.unlink(config_file)

    def test_async_client_initialization(self):
        """Test async client initialization."""
        client = AsyncPynomalaClient(
            base_url="https://async.pynomaly.com", api_key="async_key_123"
        )

        assert client.base_url == "https://async.pynomaly.com"
        assert client.api_key == "async_key_123"
        assert hasattr(client, "_session")

    def test_client_context_manager(self):
        """Test client as context manager."""
        with PynomalaClient() as client:
            assert client is not None
            assert hasattr(client, "_session")

    async def test_async_client_context_manager(self):
        """Test async client as context manager."""
        async with AsyncPynomalaClient() as client:
            assert client is not None
            assert hasattr(client, "_session")


class TestSDKDatasetOperations:
    """Test SDK dataset operations."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client for testing."""
        client = PynomalaClient()
        client._session = Mock()
        return client

    @pytest.fixture
    def sample_dataset_data(self):
        """Sample dataset data for testing."""
        return {
            "name": "Test Dataset",
            "description": "Dataset for SDK testing",
            "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "features": ["feature_1", "feature_2", "feature_3"],
            "metadata": {"source": "sdk_test"},
        }

    def test_create_dataset(self, mock_client, sample_dataset_data):
        """Test dataset creation via SDK."""
        expected_response = {
            "id": "dataset_123",
            "status": "created",
            **sample_dataset_data,
        }

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = expected_response
        mock_client._session.post.return_value = mock_response

        result = mock_client.create_dataset(**sample_dataset_data)

        # Verify API call
        mock_client._session.post.assert_called_once()
        call_args = mock_client._session.post.call_args
        assert "/datasets" in call_args[0][0]

        # Verify result
        assert result["id"] == "dataset_123"
        assert result["name"] == sample_dataset_data["name"]

    def test_get_dataset(self, mock_client):
        """Test getting dataset by ID."""
        dataset_id = "dataset_456"
        expected_response = {
            "id": dataset_id,
            "name": "Retrieved Dataset",
            "status": "ready",
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_client._session.get.return_value = mock_response

        result = mock_client.get_dataset(dataset_id)

        # Verify API call
        mock_client._session.get.assert_called_once()
        call_args = mock_client._session.get.call_args
        assert f"/datasets/{dataset_id}" in call_args[0][0]

        # Verify result
        assert result["id"] == dataset_id
        assert result["name"] == "Retrieved Dataset"

    def test_list_datasets(self, mock_client):
        """Test listing datasets."""
        expected_response = {
            "datasets": [
                {"id": "dataset_1", "name": "Dataset 1"},
                {"id": "dataset_2", "name": "Dataset 2"},
            ],
            "total": 2,
            "page": 1,
            "per_page": 10,
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_client._session.get.return_value = mock_response

        result = mock_client.list_datasets(page=1, per_page=10)

        # Verify API call
        mock_client._session.get.assert_called_once()
        call_args = mock_client._session.get.call_args
        assert "/datasets" in call_args[0][0]

        # Verify result
        assert len(result["datasets"]) == 2
        assert result["total"] == 2

    def test_update_dataset(self, mock_client):
        """Test updating dataset."""
        dataset_id = "dataset_789"
        update_data = {"description": "Updated description"}

        expected_response = {
            "id": dataset_id,
            "description": update_data["description"],
            "status": "updated",
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_client._session.patch.return_value = mock_response

        result = mock_client.update_dataset(dataset_id, **update_data)

        # Verify API call
        mock_client._session.patch.assert_called_once()
        call_args = mock_client._session.patch.call_args
        assert f"/datasets/{dataset_id}" in call_args[0][0]

        # Verify result
        assert result["description"] == update_data["description"]

    def test_delete_dataset(self, mock_client):
        """Test deleting dataset."""
        dataset_id = "dataset_delete"

        mock_response = Mock()
        mock_response.status_code = 204
        mock_client._session.delete.return_value = mock_response

        result = mock_client.delete_dataset(dataset_id)

        # Verify API call
        mock_client._session.delete.assert_called_once()
        call_args = mock_client._session.delete.call_args
        assert f"/datasets/{dataset_id}" in call_args[0][0]

        # Verify result
        assert result is True

    def test_upload_dataset_from_file(self, mock_client):
        """Test uploading dataset from file."""
        # Create temporary CSV file
        csv_data = "feature_1,feature_2,feature_3\n1,2,3\n4,5,6\n7,8,9"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            csv_file = f.name

        try:
            expected_response = {
                "id": "dataset_upload",
                "name": "Uploaded Dataset",
                "status": "processing",
            }

            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = expected_response
            mock_client._session.post.return_value = mock_response

            result = mock_client.upload_dataset_from_file(
                file_path=csv_file, name="Uploaded Dataset", format="csv"
            )

            # Verify API call
            mock_client._session.post.assert_called_once()

            # Verify result
            assert result["id"] == "dataset_upload"
            assert result["status"] == "processing"

        finally:
            os.unlink(csv_file)


class TestSDKDetectorOperations:
    """Test SDK detector operations."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client for testing."""
        client = PynomalaClient()
        client._session = Mock()
        return client

    @pytest.fixture
    def sample_detector_data(self):
        """Sample detector data for testing."""
        return {
            "name": "Test Detector",
            "algorithm": "IsolationForest",
            "parameters": {"n_estimators": 100, "contamination": 0.1},
            "description": "Detector for SDK testing",
        }

    def test_create_detector(self, mock_client, sample_detector_data):
        """Test detector creation via SDK."""
        expected_response = {
            "id": "detector_123",
            "status": "created",
            **sample_detector_data,
        }

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = expected_response
        mock_client._session.post.return_value = mock_response

        result = mock_client.create_detector(**sample_detector_data)

        # Verify API call
        mock_client._session.post.assert_called_once()
        call_args = mock_client._session.post.call_args
        assert "/detectors" in call_args[0][0]

        # Verify result
        assert result["id"] == "detector_123"
        assert result["algorithm"] == sample_detector_data["algorithm"]

    def test_train_detector(self, mock_client):
        """Test detector training via SDK."""
        detector_id = "detector_train"
        dataset_id = "dataset_train"

        expected_response = {
            "job_id": "training_job_123",
            "detector_id": detector_id,
            "status": "training",
            "estimated_completion": "2024-01-01T12:00:00Z",
        }

        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.json.return_value = expected_response
        mock_client._session.post.return_value = mock_response

        result = mock_client.train_detector(
            detector_id=detector_id,
            dataset_id=dataset_id,
            parameters={"n_estimators": 200},
        )

        # Verify API call
        mock_client._session.post.assert_called_once()
        call_args = mock_client._session.post.call_args
        assert f"/detectors/{detector_id}/train" in call_args[0][0]

        # Verify result
        assert result["job_id"] == "training_job_123"
        assert result["status"] == "training"

    def test_get_training_status(self, mock_client):
        """Test getting training status."""
        detector_id = "detector_status"

        expected_response = {
            "detector_id": detector_id,
            "status": "trained",
            "progress": 100,
            "performance_metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
            },
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_client._session.get.return_value = mock_response

        result = mock_client.get_training_status(detector_id)

        # Verify API call
        mock_client._session.get.assert_called_once()
        call_args = mock_client._session.get.call_args
        assert f"/detectors/{detector_id}/status" in call_args[0][0]

        # Verify result
        assert result["status"] == "trained"
        assert result["progress"] == 100

    def test_list_detectors(self, mock_client):
        """Test listing detectors with filters."""
        expected_response = {
            "detectors": [
                {"id": "det_1", "name": "Detector 1", "algorithm": "IsolationForest"},
                {
                    "id": "det_2",
                    "name": "Detector 2",
                    "algorithm": "LocalOutlierFactor",
                },
            ],
            "total": 2,
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_client._session.get.return_value = mock_response

        result = mock_client.list_detectors(
            algorithm="IsolationForest", status="trained"
        )

        # Verify API call
        mock_client._session.get.assert_called_once()
        call_args = mock_client._session.get.call_args
        assert "/detectors" in call_args[0][0]

        # Verify result
        assert len(result["detectors"]) == 2


class TestSDKDetectionOperations:
    """Test SDK anomaly detection operations."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client for testing."""
        client = PynomalaClient()
        client._session = Mock()
        return client

    def test_detect_anomalies(self, mock_client):
        """Test anomaly detection via SDK."""
        detector_id = "detector_detect"
        test_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        expected_response = {
            "predictions": [0, 1, 0],
            "anomaly_scores": [0.1, 0.9, 0.2],
            "processing_time": 0.05,
            "metadata": {"detector_id": detector_id, "samples_processed": 3},
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_client._session.post.return_value = mock_response

        result = mock_client.detect_anomalies(
            detector_id=detector_id,
            data=test_data,
            return_scores=True,
            return_explanations=False,
        )

        # Verify API call
        mock_client._session.post.assert_called_once()
        call_args = mock_client._session.post.call_args
        assert "/detection/detect" in call_args[0][0]

        # Verify result
        assert len(result["predictions"]) == 3
        assert len(result["anomaly_scores"]) == 3
        assert result["processing_time"] == 0.05

    def test_batch_detection(self, mock_client):
        """Test batch anomaly detection."""
        detector_id = "detector_batch"
        batch_data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

        expected_response = {
            "batch_results": [
                {"predictions": [0, 1], "anomaly_scores": [0.1, 0.8]},
                {"predictions": [1, 0], "anomaly_scores": [0.9, 0.2]},
            ],
            "total_batches": 2,
            "processing_time": 0.12,
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_client._session.post.return_value = mock_response

        result = mock_client.batch_detection(
            detector_id=detector_id, batch_data=batch_data
        )

        # Verify API call
        mock_client._session.post.assert_called_once()
        call_args = mock_client._session.post.call_args
        assert "/detection/batch" in call_args[0][0]

        # Verify result
        assert len(result["batch_results"]) == 2
        assert result["total_batches"] == 2

    def test_streaming_detection(self, mock_client):
        """Test streaming detection setup."""
        detector_id = "detector_stream"
        stream_config = {
            "buffer_size": 100,
            "threshold": 0.8,
            "callback_url": "https://webhook.example.com/anomalies",
        }

        expected_response = {
            "stream_id": "stream_123",
            "detector_id": detector_id,
            "status": "active",
            "endpoint": "wss://api.pynomaly.com/stream/stream_123",
        }

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = expected_response
        mock_client._session.post.return_value = mock_response

        result = mock_client.setup_streaming_detection(
            detector_id=detector_id, **stream_config
        )

        # Verify API call
        mock_client._session.post.assert_called_once()
        call_args = mock_client._session.post.call_args
        assert "/detection/stream" in call_args[0][0]

        # Verify result
        assert result["stream_id"] == "stream_123"
        assert result["status"] == "active"

    def test_explain_anomalies(self, mock_client):
        """Test anomaly explanation via SDK."""
        detector_id = "detector_explain"
        anomaly_data = [[7, 8, 9]]  # Anomalous sample

        expected_response = {
            "explanations": [
                {
                    "sample_id": 0,
                    "anomaly_score": 0.95,
                    "feature_contributions": {
                        "feature_0": 0.4,
                        "feature_1": 0.3,
                        "feature_2": 0.25,
                    },
                    "explanation_method": "SHAP",
                }
            ]
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_client._session.post.return_value = mock_response

        result = mock_client.explain_anomalies(
            detector_id=detector_id, data=anomaly_data, explanation_method="SHAP"
        )

        # Verify API call
        mock_client._session.post.assert_called_once()
        call_args = mock_client._session.post.call_args
        assert "/detection/explain" in call_args[0][0]

        # Verify result
        assert len(result["explanations"]) == 1
        assert "feature_contributions" in result["explanations"][0]


class TestSDKErrorHandling:
    """Test SDK error handling and exception management."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client for testing."""
        client = PynomalaClient()
        client._session = Mock()
        return client

    def test_authentication_error(self, mock_client):
        """Test authentication error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": "authentication_failed",
            "message": "Invalid API key",
        }
        mock_client._session.get.return_value = mock_response

        with pytest.raises(AuthenticationError) as exc_info:
            mock_client.get_dataset("test_id")

        assert "Invalid API key" in str(exc_info.value)

    def test_validation_error(self, mock_client):
        """Test validation error handling."""
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "error": "validation_error",
            "message": "Invalid data format",
            "details": {"field": "data", "issue": "Must be a list of numbers"},
        }
        mock_client._session.post.return_value = mock_response

        with pytest.raises(ValidationError) as exc_info:
            mock_client.create_dataset(name="test", data="invalid")

        assert "Invalid data format" in str(exc_info.value)

    def test_not_found_error(self, mock_client):
        """Test not found error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": "not_found",
            "message": "Dataset not found",
        }
        mock_client._session.get.return_value = mock_response

        with pytest.raises(NotFoundError) as exc_info:
            mock_client.get_dataset("nonexistent_id")

        assert "Dataset not found" in str(exc_info.value)

    def test_rate_limit_error(self, mock_client):
        """Test rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": "rate_limit_exceeded",
            "message": "Too many requests",
            "retry_after": 60,
        }
        mock_client._session.get.return_value = mock_response

        with pytest.raises(RateLimitError) as exc_info:
            mock_client.list_datasets()

        assert "Too many requests" in str(exc_info.value)
        assert exc_info.value.retry_after == 60

    def test_network_error(self, mock_client):
        """Test network error handling."""
        mock_client._session.get.side_effect = httpx.ConnectError("Connection failed")

        with pytest.raises(NetworkError) as exc_info:
            mock_client.get_dataset("test_id")

        assert "Connection failed" in str(exc_info.value)

    def test_generic_sdk_error(self, mock_client):
        """Test generic SDK error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": "internal_server_error",
            "message": "Something went wrong",
        }
        mock_client._session.get.return_value = mock_response

        with pytest.raises(SDKError) as exc_info:
            mock_client.get_dataset("test_id")

        assert "Something went wrong" in str(exc_info.value)

    def test_retry_mechanism(self, mock_client):
        """Test automatic retry mechanism."""
        # First two calls fail, third succeeds
        failed_response = Mock()
        failed_response.status_code = 503
        failed_response.json.return_value = {"error": "service_unavailable"}

        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"id": "success"}

        mock_client._session.get.side_effect = [
            failed_response,
            failed_response,
            success_response,
        ]

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = mock_client.get_dataset("test_id")

        assert result["id"] == "success"
        assert mock_client._session.get.call_count == 3


class TestSDKAsyncOperations:
    """Test SDK async operations."""

    @pytest.fixture
    def mock_async_client(self):
        """Create mock async client for testing."""
        client = AsyncPynomalaClient()
        client._session = AsyncMock()
        return client

    async def test_async_create_dataset(self, mock_async_client):
        """Test async dataset creation."""
        dataset_data = {
            "name": "Async Dataset",
            "data": [[1, 2], [3, 4]],
            "features": ["x", "y"],
        }

        expected_response = {"id": "async_dataset_123", **dataset_data}

        mock_response = AsyncMock()
        mock_response.status_code = 201
        mock_response.json.return_value = expected_response
        mock_async_client._session.post.return_value = mock_response

        result = await mock_async_client.create_dataset(**dataset_data)

        # Verify API call
        mock_async_client._session.post.assert_called_once()

        # Verify result
        assert result["id"] == "async_dataset_123"

    async def test_async_detect_anomalies(self, mock_async_client):
        """Test async anomaly detection."""
        detector_id = "async_detector"
        test_data = [[1, 2], [3, 4]]

        expected_response = {
            "predictions": [0, 1],
            "anomaly_scores": [0.1, 0.9],
            "processing_time": 0.03,
        }

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_async_client._session.post.return_value = mock_response

        result = await mock_async_client.detect_anomalies(
            detector_id=detector_id, data=test_data
        )

        # Verify result
        assert len(result["predictions"]) == 2
        assert len(result["anomaly_scores"]) == 2

    async def test_async_concurrent_operations(self, mock_async_client):
        """Test concurrent async operations."""
        # Mock multiple successful responses
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_async_client._session.get.return_value = mock_response

        # Execute concurrent operations
        tasks = [mock_async_client.get_dataset(f"dataset_{i}") for i in range(5)]

        results = await asyncio.gather(*tasks)

        # Verify all operations completed
        assert len(results) == 5
        assert all(r["status"] == "success" for r in results)
        assert mock_async_client._session.get.call_count == 5

    async def test_async_error_handling(self, mock_async_client):
        """Test async error handling."""
        mock_async_client._session.get.side_effect = httpx.ConnectError(
            "Async connection failed"
        )

        with pytest.raises(NetworkError) as exc_info:
            await mock_async_client.get_dataset("test_id")

        assert "Async connection failed" in str(exc_info.value)


class TestSDKCaching:
    """Test SDK caching mechanisms."""

    @pytest.fixture
    def cached_client(self):
        """Create client with caching enabled."""
        client = PynomalaClient(enable_caching=True, cache_ttl=300)
        client._session = Mock()
        client._cache = {}
        return client

    def test_response_caching(self, cached_client):
        """Test response caching functionality."""
        dataset_id = "cached_dataset"
        expected_response = {"id": dataset_id, "name": "Cached Dataset"}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        cached_client._session.get.return_value = mock_response

        # First call - should hit API
        result1 = cached_client.get_dataset(dataset_id)
        assert cached_client._session.get.call_count == 1

        # Second call - should use cache
        result2 = cached_client.get_dataset(dataset_id)
        assert cached_client._session.get.call_count == 1  # No additional call

        # Results should be identical
        assert result1 == result2

    def test_cache_expiration(self, cached_client):
        """Test cache expiration."""
        dataset_id = "expiring_dataset"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": dataset_id}
        cached_client._session.get.return_value = mock_response

        # Set very short TTL
        cached_client.cache_ttl = 0.1

        # First call
        cached_client.get_dataset(dataset_id)
        assert cached_client._session.get.call_count == 1

        # Wait for cache to expire
        time.sleep(0.2)

        # Second call - should hit API again
        cached_client.get_dataset(dataset_id)
        assert cached_client._session.get.call_count == 2

    def test_cache_invalidation(self, cached_client):
        """Test manual cache invalidation."""
        dataset_id = "invalidated_dataset"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": dataset_id}
        cached_client._session.get.return_value = mock_response

        # Cache response
        cached_client.get_dataset(dataset_id)
        assert cached_client._session.get.call_count == 1

        # Invalidate cache
        cached_client.invalidate_cache(f"dataset_{dataset_id}")

        # Next call should hit API
        cached_client.get_dataset(dataset_id)
        assert cached_client._session.get.call_count == 2


class TestSDKAuthentication:
    """Test SDK authentication mechanisms."""

    def test_api_key_authentication(self):
        """Test API key authentication."""
        api_key = "test_api_key_123"
        client = PynomalaClient(api_key=api_key)
        client._session = Mock()

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"authenticated": True}
        client._session.get.return_value = mock_response

        client.get_dataset("test_id")

        # Verify API key was included in headers
        call_kwargs = client._session.get.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert "Authorization" in headers
        assert f"Bearer {api_key}" in headers["Authorization"]

    def test_token_refresh(self):
        """Test token refresh mechanism."""
        client = PynomalaClient()
        client._session = Mock()

        # Mock token refresh
        refresh_response = Mock()
        refresh_response.status_code = 200
        refresh_response.json.return_value = {
            "access_token": "new_token_456",
            "expires_in": 3600,
        }

        with patch.object(client, "_refresh_token", return_value="new_token_456"):
            new_token = client.refresh_token()
            assert new_token == "new_token_456"

    def test_authentication_failure_retry(self):
        """Test authentication failure and retry."""
        client = PynomalaClient(api_key="expired_key")
        client._session = Mock()

        # First call fails with 401
        auth_failed_response = Mock()
        auth_failed_response.status_code = 401
        auth_failed_response.json.return_value = {"error": "token_expired"}

        # Second call succeeds after token refresh
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"id": "success"}

        client._session.get.side_effect = [auth_failed_response, success_response]

        with patch.object(client, "_refresh_token", return_value="refreshed_token"):
            result = client.get_dataset("test_id")
            assert result["id"] == "success"
            assert client._session.get.call_count == 2


class TestSDKUtilities:
    """Test SDK utility functions and helpers."""

    def test_data_format_validation(self):
        """Test data format validation utilities."""
        from pynomaly.sdk.utils import validate_data_format

        # Valid data
        valid_data = [[1, 2, 3], [4, 5, 6]]
        assert validate_data_format(valid_data) is True

        # Invalid data
        invalid_data = ["not", "a", "matrix"]
        assert validate_data_format(invalid_data) is False

        # Empty data
        empty_data = []
        assert validate_data_format(empty_data) is False

    def test_response_parsing(self):
        """Test response parsing utilities."""
        from pynomaly.sdk.utils import parse_api_response

        # Successful response
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"data": "success"}

        result = parse_api_response(success_response)
        assert result["data"] == "success"

        # Error response
        error_response = Mock()
        error_response.status_code = 400
        error_response.json.return_value = {"error": "bad_request"}

        with pytest.raises(SDKError):
            parse_api_response(error_response)

    def test_url_building(self):
        """Test URL building utilities."""
        from pynomaly.sdk.utils import build_api_url

        base_url = "https://api.pynomaly.com"
        api_version = "v1"
        endpoint = "datasets"

        url = build_api_url(base_url, api_version, endpoint)
        assert url == "https://api.pynomaly.com/api/v1/datasets"

        # With parameters
        params = {"page": 1, "per_page": 10}
        url_with_params = build_api_url(base_url, api_version, endpoint, params)
        assert "page=1" in url_with_params
        assert "per_page=10" in url_with_params

    def test_data_serialization(self):
        """Test data serialization utilities."""
        from pynomaly.sdk.utils import serialize_numpy_data

        # NumPy array
        numpy_data = np.array([[1, 2], [3, 4]])
        serialized = serialize_numpy_data(numpy_data)
        assert isinstance(serialized, list)
        assert serialized == [[1, 2], [3, 4]]

        # Regular list (should pass through)
        list_data = [[1, 2], [3, 4]]
        serialized_list = serialize_numpy_data(list_data)
        assert serialized_list == list_data

    def test_configuration_validation(self):
        """Test configuration validation."""
        from pynomaly.sdk.utils import validate_client_config

        # Valid config
        valid_config = {
            "base_url": "https://api.pynomaly.com",
            "api_key": "valid_key",
            "timeout": 30,
        }
        assert validate_client_config(valid_config) is True

        # Invalid config
        invalid_config = {"base_url": "not_a_url", "timeout": -1}
        assert validate_client_config(invalid_config) is False


class TestSDKIntegrationScenarios:
    """Test complete SDK integration scenarios."""

    @pytest.fixture
    def integration_client(self):
        """Create client for integration testing."""
        client = PynomalaClient()
        client._session = Mock()
        return client

    def test_complete_workflow(self, integration_client):
        """Test complete anomaly detection workflow."""
        # Mock responses for each step
        dataset_response = Mock()
        dataset_response.status_code = 201
        dataset_response.json.return_value = {"id": "workflow_dataset"}

        detector_response = Mock()
        detector_response.status_code = 201
        detector_response.json.return_value = {"id": "workflow_detector"}

        training_response = Mock()
        training_response.status_code = 202
        training_response.json.return_value = {
            "job_id": "training_123",
            "status": "training",
        }

        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "trained", "progress": 100}

        detection_response = Mock()
        detection_response.status_code = 200
        detection_response.json.return_value = {
            "predictions": [0, 1, 0],
            "anomaly_scores": [0.1, 0.9, 0.2],
        }

        # Configure mock to return appropriate responses
        integration_client._session.post.side_effect = [
            dataset_response,
            detector_response,
            training_response,
            detection_response,
        ]
        integration_client._session.get.return_value = status_response

        # Execute workflow
        # 1. Create dataset
        dataset = integration_client.create_dataset(
            name="Workflow Dataset", data=[[1, 2], [3, 4], [5, 6]], features=["x", "y"]
        )
        assert dataset["id"] == "workflow_dataset"

        # 2. Create detector
        detector = integration_client.create_detector(
            name="Workflow Detector", algorithm="IsolationForest"
        )
        assert detector["id"] == "workflow_detector"

        # 3. Train detector
        training_job = integration_client.train_detector(
            detector_id=detector["id"], dataset_id=dataset["id"]
        )
        assert training_job["status"] == "training"

        # 4. Check training status
        status = integration_client.get_training_status(detector["id"])
        assert status["status"] == "trained"

        # 5. Detect anomalies
        results = integration_client.detect_anomalies(
            detector_id=detector["id"], data=[[7, 8], [9, 10], [1, 2]]
        )
        assert len(results["predictions"]) == 3

        # Verify all API calls were made
        assert integration_client._session.post.call_count == 4
        assert integration_client._session.get.call_count == 1

    def test_error_recovery_workflow(self, integration_client):
        """Test workflow with error recovery."""
        # Simulate transient error followed by success
        error_response = Mock()
        error_response.status_code = 503
        error_response.json.return_value = {"error": "service_unavailable"}

        success_response = Mock()
        success_response.status_code = 201
        success_response.json.return_value = {"id": "recovered_dataset"}

        integration_client._session.post.side_effect = [
            error_response,
            success_response,
        ]

        with patch("time.sleep"):  # Speed up retry delays
            # Should recover from transient error
            result = integration_client.create_dataset(
                name="Recovery Dataset", data=[[1, 2]], features=["x"]
            )

        assert result["id"] == "recovered_dataset"
        assert integration_client._session.post.call_count == 2

    def test_concurrent_operations_workflow(self, integration_client):
        """Test concurrent operations in workflow."""
        # Mock responses for concurrent operations
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        integration_client._session.get.return_value = mock_response

        # Execute concurrent dataset retrievals
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(integration_client.get_dataset, f"dataset_{i}")
                for i in range(5)
            ]

            results = [future.result() for future in futures]

        # All operations should succeed
        assert len(results) == 5
        assert all(r["status"] == "success" for r in results)
        assert integration_client._session.get.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
