"""
Comprehensive integration test suite for Pynomaly SDK.

This module provides end-to-end testing of the SDK, including
client-server integration, workflow testing, and performance validation.
"""

import asyncio
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pynomaly.presentation.sdk.async_client import AsyncPynomalyClient
from pynomaly.presentation.sdk.client import PynomalyClient
from pynomaly.presentation.sdk.config import PynomalyConfig, load_config_from_file
from pynomaly.presentation.sdk.exceptions import AuthenticationError, NetworkError
from pynomaly.presentation.sdk.models import (
    DetectionRequest,
    DetectionResponse,
    TrainingRequest,
)


class TestSDKEndToEndWorkflows:
    """Test complete SDK workflows from start to finish."""

    def test_complete_anomaly_detection_workflow(self):
        """Test complete anomaly detection workflow."""
        # Mock server responses
        with (
            patch("requests.Session.post") as mock_post,
            patch("requests.Session.get") as mock_get,
        ):
            # Mock login response
            login_response = Mock()
            login_response.status_code = 200
            login_response.json.return_value = {
                "access_token": "test-token",
                "token_type": "bearer",
            }

            # Mock detection response
            detection_response = Mock()
            detection_response.status_code = 200
            detection_response.json.return_value = {
                "anomaly_scores": [0.1, 0.9, 0.2, 0.8],
                "anomaly_labels": [0, 1, 0, 1],
                "execution_time": 0.123,
                "model_info": {"name": "isolation_forest", "version": "1.0"},
            }

            # Mock health check response
            health_response = Mock()
            health_response.status_code = 200
            health_response.json.return_value = {
                "status": "healthy",
                "version": "1.0.0",
            }

            mock_post.side_effect = [login_response, detection_response]
            mock_get.return_value = health_response

            # Execute workflow
            client = PynomalyClient()

            # Step 1: Check health
            health = client.health_check()
            assert health.status == "healthy"

            # Step 2: Login
            token = client.login("user@example.com", "password")
            assert token == "test-token"
            assert client.api_key == "test-token"

            # Step 3: Detect anomalies
            request = DetectionRequest(
                data=[[1, 2], [3, 4], [5, 6], [7, 8]], algorithm="isolation_forest"
            )
            response = client.detect_anomalies(request)

            assert isinstance(response, DetectionResponse)
            assert len(response.anomaly_scores) == 4
            assert response.anomaly_count == 2
            assert response.execution_time == 0.123

    def test_complete_model_training_workflow(self):
        """Test complete model training workflow."""
        with (
            patch("requests.Session.post") as mock_post,
            patch("requests.Session.get") as mock_get,
        ):
            # Mock responses
            login_response = Mock()
            login_response.status_code = 200
            login_response.json.return_value = {"access_token": "test-token"}

            training_response = Mock()
            training_response.status_code = 200
            training_response.json.return_value = {
                "job_id": "train-123",
                "status": "started",
                "model_id": "model-456",
                "estimated_duration": 300,
            }

            # Mock training status progression
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

            status_mocks = []
            for status_data in status_responses:
                status_mock = Mock()
                status_mock.status_code = 200
                status_mock.json.return_value = status_data
                status_mocks.append(status_mock)

            mock_post.side_effect = [login_response, training_response]
            mock_get.side_effect = status_mocks

            # Execute workflow
            client = PynomalyClient()

            # Step 1: Login
            client.login("user@example.com", "password")

            # Step 2: Start training
            request = TrainingRequest(
                data=[[1, 2], [3, 4], [5, 6], [7, 8]],
                algorithm="isolation_forest",
                hyperparameters={"n_estimators": 100},
            )
            response = client.train_model(request)

            assert response.job_id == "train-123"
            assert response.status == "started"

            # Step 3: Monitor training progress
            for expected_progress in [25, 50, 75, 100]:
                status = client.get_training_status("train-123")
                assert status["progress"] == expected_progress

                if status["status"] == "completed":
                    assert status["model_id"] == "model-456"
                    break

    def test_complete_dataset_management_workflow(self):
        """Test complete dataset management workflow."""
        with (
            patch("requests.Session.post") as mock_post,
            patch("requests.Session.get") as mock_get,
            patch("requests.Session.delete") as mock_delete,
            patch("builtins.open", create=True) as mock_open,
        ):
            # Mock file content
            mock_file = Mock()
            mock_file.read.return_value = b"col1,col2,col3\n1,2,3\n4,5,6\n"
            mock_open.return_value.__enter__.return_value = mock_file

            # Mock responses
            login_response = Mock()
            login_response.status_code = 200
            login_response.json.return_value = {"access_token": "test-token"}

            upload_response = Mock()
            upload_response.status_code = 200
            upload_response.json.return_value = {
                "dataset_id": "dataset-123",
                "name": "test_dataset",
                "size": 1000,
                "features": 3,
            }

            list_response = Mock()
            list_response.status_code = 200
            list_response.json.return_value = {
                "datasets": [
                    {"dataset_id": "dataset-123", "name": "test_dataset", "size": 1000}
                ]
            }

            delete_response = Mock()
            delete_response.status_code = 200

            mock_post.side_effect = [login_response, upload_response]
            mock_get.return_value = list_response
            mock_delete.return_value = delete_response

            # Execute workflow
            client = PynomalyClient()

            # Step 1: Login
            client.login("user@example.com", "password")

            # Step 2: Upload dataset
            dataset_info = client.upload_dataset("test.csv", "test_dataset")
            assert dataset_info.dataset_id == "dataset-123"
            assert dataset_info.name == "test_dataset"

            # Step 3: List datasets
            datasets = client.list_datasets()
            assert len(datasets) == 1
            assert datasets[0].name == "test_dataset"

            # Step 4: Delete dataset
            client.delete_dataset("dataset-123")
            # Should not raise an exception


class TestAsyncSDKIntegration:
    """Test async SDK integration scenarios."""

    @pytest.mark.asyncio
    async def test_async_concurrent_operations(self):
        """Test concurrent async operations."""
        with (
            patch("aiohttp.ClientSession.post") as mock_post,
            patch("aiohttp.ClientSession.get") as mock_get,
        ):
            # Mock responses
            login_response = AsyncMock()
            login_response.status = 200
            login_response.json = AsyncMock(return_value={"access_token": "test-token"})

            detection_response = AsyncMock()
            detection_response.status = 200
            detection_response.json = AsyncMock(
                return_value={
                    "anomaly_scores": [0.1, 0.9],
                    "anomaly_labels": [0, 1],
                    "execution_time": 0.1,
                }
            )

            health_response = AsyncMock()
            health_response.status = 200
            health_response.json = AsyncMock(
                return_value={"status": "healthy", "version": "1.0.0"}
            )

            mock_post.return_value.__aenter__.side_effect = [
                login_response,
                detection_response,
                detection_response,
            ]
            mock_get.return_value.__aenter__.return_value = health_response

            # Execute concurrent operations
            client = AsyncPynomalyClient()

            async with client:
                # Login first
                await client.login("user@example.com", "password")

                # Create multiple detection requests
                requests = [
                    DetectionRequest(
                        data=[[1, 2], [3, 4]], algorithm="isolation_forest"
                    )
                    for _ in range(2)
                ]

                # Execute concurrently
                results = await asyncio.gather(
                    client.health_check(),
                    client.detect_anomalies(requests[0]),
                    client.detect_anomalies(requests[1]),
                )

                health, response1, response2 = results

                assert health.status == "healthy"
                assert isinstance(response1, DetectionResponse)
                assert isinstance(response2, DetectionResponse)
                assert len(response1.anomaly_scores) == 2
                assert len(response2.anomaly_scores) == 2

    @pytest.mark.asyncio
    async def test_async_error_handling_workflow(self):
        """Test async error handling workflow."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            # Mock error responses
            error_response = AsyncMock()
            error_response.status = 401
            error_response.json = AsyncMock(return_value={"detail": "Unauthorized"})

            mock_post.return_value.__aenter__.return_value = error_response

            client = AsyncPynomalyClient()

            async with client:
                # Test authentication error
                with pytest.raises(AuthenticationError):
                    await client.login("user@example.com", "wrong_password")


class TestSDKConfigurationIntegration:
    """Test SDK configuration integration."""

    def test_config_file_integration(self):
        """Test SDK integration with configuration files."""
        # Create temporary config file
        config_data = {
            "base_url": "https://config.api.com",
            "api_key": "config-api-key",
            "timeout": 45.0,
            "max_retries": 5,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file_path = f.name

        try:
            # Load config and create client
            config = load_config_from_file(config_file_path)
            client = PynomalyClient.from_config(config.to_dict())

            assert client.base_url == "https://config.api.com"
            assert client.api_key == "config-api-key"
            assert client.timeout == 45.0
            assert client.max_retries == 5
        finally:
            os.unlink(config_file_path)

    def test_environment_config_integration(self):
        """Test SDK integration with environment configuration."""
        env_vars = {
            "PYNOMALY_BASE_URL": "https://env.api.com",
            "PYNOMALY_API_KEY": "env-api-key",
            "PYNOMALY_TIMEOUT": "60.0",
        }

        with patch.dict(os.environ, env_vars):
            from pynomaly.presentation.sdk.config import load_config_from_env

            config = load_config_from_env()
            client = PynomalyClient.from_config(config.to_dict())

            assert client.base_url == "https://env.api.com"
            assert client.api_key == "env-api-key"
            assert client.timeout == 60.0


class TestSDKPerformanceIntegration:
    """Test SDK performance characteristics."""

    def test_client_performance_metrics(self):
        """Test client performance metrics."""
        with patch("requests.Session.get") as mock_get:
            # Mock fast response
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"status": "healthy", "version": "1.0.0"}
            mock_get.return_value = response

            client = PynomalyClient()

            # Measure response time
            start_time = time.time()
            health = client.health_check()
            end_time = time.time()

            response_time = end_time - start_time

            assert health.status == "healthy"
            assert response_time < 1.0  # Should be fast with mocked response

    def test_concurrent_client_performance(self):
        """Test concurrent client performance."""
        with patch("requests.Session.get") as mock_get:
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"status": "healthy", "version": "1.0.0"}
            mock_get.return_value = response

            def make_request():
                client = PynomalyClient()
                return client.health_check()

            # Execute concurrent requests
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(10)]
                results = [future.result() for future in futures]
            end_time = time.time()

            total_time = end_time - start_time

            assert len(results) == 10
            assert all(r.status == "healthy" for r in results)
            assert total_time < 5.0  # Should complete reasonably quickly

    @pytest.mark.asyncio
    async def test_async_client_performance(self):
        """Test async client performance."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            response = AsyncMock()
            response.status = 200
            response.json = AsyncMock(
                return_value={"status": "healthy", "version": "1.0.0"}
            )
            mock_get.return_value.__aenter__.return_value = response

            client = AsyncPynomalyClient()

            async with client:
                # Execute multiple concurrent requests
                start_time = time.time()
                tasks = [client.health_check() for _ in range(10)]
                results = await asyncio.gather(*tasks)
                end_time = time.time()

                total_time = end_time - start_time

                assert len(results) == 10
                assert all(r.status == "healthy" for r in results)
                assert total_time < 2.0  # Async should be faster


class TestSDKErrorRecoveryIntegration:
    """Test SDK error recovery and resilience."""

    def test_retry_mechanism_integration(self):
        """Test retry mechanism integration."""
        with (
            patch("requests.Session.get") as mock_get,
            patch("time.sleep") as mock_sleep,
        ):  # Speed up tests
            # First two calls fail, third succeeds
            fail_response = Mock()
            fail_response.status_code = 500
            fail_response.json.return_value = {"detail": "Server error"}

            success_response = Mock()
            success_response.status_code = 200
            success_response.json.return_value = {
                "status": "healthy",
                "version": "1.0.0",
            }

            mock_get.side_effect = [fail_response, fail_response, success_response]
            mock_sleep.return_value = None  # Don't actually sleep

            client = PynomalyClient(max_retries=3, retry_delay=0.1)

            # Should succeed on third attempt
            health = client.health_check()
            assert health.status == "healthy"
            assert mock_get.call_count == 3

    def test_network_error_handling_integration(self):
        """Test network error handling integration."""
        from requests.exceptions import ConnectionError

        with patch("requests.Session.get") as mock_get:
            mock_get.side_effect = ConnectionError("Network unreachable")

            client = PynomalyClient(max_retries=1)

            with pytest.raises(NetworkError):
                client.health_check()

    def test_timeout_handling_integration(self):
        """Test timeout handling integration."""
        from requests.exceptions import Timeout

        with patch("requests.Session.get") as mock_get:
            mock_get.side_effect = Timeout("Request timeout")

            client = PynomalyClient(timeout=1.0, max_retries=1)

            with pytest.raises(NetworkError):
                client.health_check()


class TestSDKDataFlowIntegration:
    """Test data flow through SDK components."""

    def test_request_response_data_flow(self):
        """Test data flow from request to response."""
        with patch("requests.Session.post") as mock_post:
            # Mock response with specific data
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "anomaly_scores": [0.1, 0.2, 0.9, 0.1],
                "anomaly_labels": [0, 0, 1, 0],
                "execution_time": 0.456,
                "model_info": {
                    "name": "test_model",
                    "version": "2.0",
                    "algorithm": "isolation_forest",
                },
                "metadata": {"n_samples": 4, "n_features": 3, "contamination": 0.1},
            }
            mock_post.return_value = response

            client = PynomalyClient(api_key="test-token")

            # Create request with specific data
            request = DetectionRequest(
                data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                algorithm="isolation_forest",
                parameters={"contamination": 0.1, "n_estimators": 100},
            )

            # Execute request
            detection_response = client.detect_anomalies(request)

            # Verify data flow
            assert isinstance(detection_response, DetectionResponse)
            assert detection_response.anomaly_scores == [0.1, 0.2, 0.9, 0.1]
            assert detection_response.anomaly_labels == [0, 0, 1, 0]
            assert detection_response.execution_time == 0.456
            assert detection_response.model_info["name"] == "test_model"
            assert detection_response.metadata["n_samples"] == 4

            # Verify request was sent correctly
            call_args = mock_post.call_args
            assert call_args[1]["json"]["data"] == request.data
            assert call_args[1]["json"]["algorithm"] == request.algorithm
            assert call_args[1]["json"]["parameters"] == request.parameters

    def test_model_serialization_integration(self):
        """Test model serialization throughout the flow."""
        # Test request serialization
        request = DetectionRequest(
            data=[[1, 2], [3, 4]],
            algorithm="isolation_forest",
            parameters={"contamination": 0.1},
        )

        serialized_request = request.to_dict()
        assert serialized_request["data"] == [[1, 2], [3, 4]]

        # Test response deserialization
        response_data = {
            "anomaly_scores": [0.1, 0.9],
            "anomaly_labels": [0, 1],
            "execution_time": 0.123,
        }

        response = DetectionResponse.from_dict(response_data)
        assert response.anomaly_scores == [0.1, 0.9]
        assert response.anomaly_labels == [0, 1]
        assert response.execution_time == 0.123

    def test_configuration_data_flow(self):
        """Test configuration data flow."""
        # Create config
        config = PynomalyConfig(
            base_url="https://test.api.com",
            api_key="test-key",
            timeout=30.0,
            max_retries=3,
        )

        # Convert to dict and back
        config_dict = config.to_dict()
        reconstructed_config = PynomalyConfig.from_dict(config_dict)

        assert reconstructed_config.base_url == config.base_url
        assert reconstructed_config.api_key == config.api_key
        assert reconstructed_config.timeout == config.timeout
        assert reconstructed_config.max_retries == config.max_retries


class TestSDKSecurityIntegration:
    """Test SDK security features integration."""

    def test_authentication_flow_integration(self):
        """Test authentication flow integration."""
        with patch("requests.Session.post") as mock_post:
            # Mock login response
            login_response = Mock()
            login_response.status_code = 200
            login_response.json.return_value = {
                "access_token": "secure-jwt-token",
                "token_type": "bearer",
                "expires_in": 3600,
            }

            # Mock authenticated request
            auth_response = Mock()
            auth_response.status_code = 200
            auth_response.json.return_value = {
                "anomaly_scores": [0.1, 0.9],
                "anomaly_labels": [0, 1],
                "execution_time": 0.123,
            }

            mock_post.side_effect = [login_response, auth_response]

            client = PynomalyClient()

            # Login and get token
            token = client.login("user@example.com", "secure_password")
            assert token == "secure-jwt-token"
            assert client.api_key == "secure-jwt-token"

            # Make authenticated request
            request = DetectionRequest(
                data=[[1, 2], [3, 4]], algorithm="isolation_forest"
            )
            response = client.detect_anomalies(request)

            # Verify authentication header was sent
            auth_call = mock_post.call_args_list[1]
            headers = auth_call[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer secure-jwt-token"

    def test_ssl_verification_integration(self):
        """Test SSL verification integration."""
        with patch("requests.Session.post") as mock_post:
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"access_token": "test-token"}
            mock_post.return_value = response

            # Test with SSL verification enabled (default)
            client = PynomalyClient(verify_ssl=True)
            client.login("user@example.com", "password")

            # Verify SSL verification is enabled in session
            session_verify = mock_post.call_args[1].get("verify", True)
            assert session_verify is True

            # Test with SSL verification disabled
            client_no_ssl = PynomalyClient(verify_ssl=False)
            client_no_ssl.login("user@example.com", "password")

            # Verify SSL verification is disabled
            session_verify = mock_post.call_args[1].get("verify", True)
            assert session_verify is False


# Integration test fixtures
@pytest.fixture
def sample_workflow_data():
    """Create sample data for workflow testing."""
    return {
        "training_data": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        "detection_data": [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
        "algorithm": "isolation_forest",
        "parameters": {"contamination": 0.1, "n_estimators": 100},
    }


@pytest.fixture
def mock_server_responses():
    """Create mock server responses for testing."""
    return {
        "login": {
            "access_token": "test-jwt-token",
            "token_type": "bearer",
            "expires_in": 3600,
        },
        "health": {"status": "healthy", "version": "1.0.0", "uptime": 3600},
        "detection": {
            "anomaly_scores": [0.1, 0.9, 0.2],
            "anomaly_labels": [0, 1, 0],
            "execution_time": 0.123,
            "model_info": {"name": "isolation_forest", "version": "1.0"},
        },
        "training": {
            "job_id": "train-123",
            "status": "started",
            "model_id": "model-456",
            "estimated_duration": 300,
        },
    }


@pytest.fixture
def performance_test_data():
    """Create data for performance testing."""
    return {
        "small_dataset": [[i, i + 1, i + 2] for i in range(100)],
        "medium_dataset": [[i, i + 1, i + 2] for i in range(1000)],
        "large_dataset": [[i, i + 1, i + 2] for i in range(10000)],
    }
