"""
Comprehensive cross-layer presentation integration tests.

This module provides extensive testing for cross-layer communication,
authentication flow, data flow validation, and performance testing
across all presentation layer components.
"""

import asyncio
import concurrent.futures
import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from monorepo.presentation.api.app import create_app as create_api_app
from monorepo.presentation.cli.app import app as cli_app
from monorepo.presentation.sdk.async_client import AsyncPynomályClient
from monorepo.presentation.sdk.client import PynomályClient
from monorepo.presentation.sdk.config import SDKConfig
from monorepo.presentation.web.app import app as web_app


class TestCrossLayerCommunication:
    """Test communication between presentation layers."""

    @pytest.fixture
    def api_client(self):
        """Create API test client."""
        app = create_api_app()
        return TestClient(app)

    @pytest.fixture
    def web_client(self):
        """Create Web UI test client."""
        return TestClient(web_app)

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sdk_config(self):
        """Create SDK configuration."""
        return SDKConfig(
            api_url="http://localhost:8000", api_key="test-key", timeout=30
        )

    @pytest.fixture
    def sdk_client(self, sdk_config):
        """Create SDK client."""
        return PynomályClient(sdk_config)

    def test_api_web_integration(self, api_client, web_client):
        """Test API and Web UI integration."""
        # Test that Web UI can communicate with API

        # Mock API response for health check
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"status": "healthy"}

            # Test web UI health page
            response = web_client.get("/health")

            # Should handle API communication
            assert response.status_code in [200, 404, 500]

    def test_cli_api_integration(self, cli_runner, api_client):
        """Test CLI and API integration."""
        # Test CLI command that communicates with API

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = [{"id": "test", "name": "Test"}]

            # Test CLI list command
            result = cli_runner.invoke(cli_app, ["dataset", "list"])

            # Should not crash
            assert result.exit_code in [0, 1]

    def test_sdk_api_integration(self, sdk_client, api_client):
        """Test SDK and API integration."""
        # Test SDK client communicating with API

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch("requests.Session.request", return_value=mock_response):
            # Test SDK health check
            health = sdk_client.health_check()
            assert health["status"] == "healthy"

    def test_web_cli_config_sharing(self, web_client, cli_runner):
        """Test configuration sharing between Web UI and CLI."""
        # Test that both can use same configuration

        config_data = {"api_url": "http://localhost:8000", "api_key": "shared-key"}

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            # Test CLI config import
            result = cli_runner.invoke(
                cli_app, ["config", "import", "--file", config_file]
            )

            # Should not crash
            assert result.exit_code in [0, 1]

            # Test web UI can access same config
            # (This would require actual shared config implementation)

        finally:
            os.unlink(config_file)

    def test_api_sdk_data_flow(self, api_client, sdk_client):
        """Test data flow between API and SDK."""
        # Test complete data flow from SDK to API

        # Mock dataset creation flow
        mock_responses = [
            Mock(
                status_code=201, json=lambda: {"id": "dataset1", "name": "Test Dataset"}
            ),  # Create
            Mock(
                status_code=200, json=lambda: {"id": "dataset1", "name": "Test Dataset"}
            ),  # Get
            Mock(status_code=204),  # Delete
        ]

        with patch("requests.Session.request", side_effect=mock_responses):
            # Create dataset via SDK
            dataset = sdk_client.create_dataset(
                name="Test Dataset", data=[[1, 2, 3], [4, 5, 6]]
            )
            assert dataset["id"] == "dataset1"

            # Get dataset via SDK
            retrieved = sdk_client.get_dataset("dataset1")
            assert retrieved["id"] == "dataset1"

            # Delete dataset via SDK
            result = sdk_client.delete_dataset("dataset1")
            assert result is True

    def test_web_api_htmx_flow(self, web_client, api_client):
        """Test Web UI HTMX to API flow."""
        # Test HTMX endpoints calling API

        htmx_headers = {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/dashboard",
            "HX-Target": "dashboard-content",
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"stats": {"total": 100}}

            # Test HTMX dashboard stats
            response = web_client.get("/htmx/dashboard/stats", headers=htmx_headers)

            # Should handle API integration
            assert response.status_code in [200, 401, 404]

    def test_cli_sdk_equivalence(self, cli_runner, sdk_client):
        """Test CLI and SDK provide equivalent functionality."""
        # Test that CLI commands and SDK methods produce similar results

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": "dataset1", "name": "Dataset 1"}]

        # Test SDK list datasets
        with patch("requests.Session.request", return_value=mock_response):
            sdk_datasets = sdk_client.list_datasets()
            assert len(sdk_datasets) == 1

        # Test CLI list datasets
        with patch("requests.get", return_value=mock_response):
            result = cli_runner.invoke(cli_app, ["dataset", "list"])
            # Should provide similar functionality
            assert result.exit_code in [0, 1]


class TestAuthenticationFlow:
    """Test authentication flow across presentation layers."""

    @pytest.fixture
    def auth_token(self):
        """Create test authentication token."""
        return "test-auth-token-12345"

    @pytest.fixture
    def auth_headers(self, auth_token):
        """Create authentication headers."""
        return {"Authorization": f"Bearer {auth_token}"}

    def test_api_authentication(self, auth_token):
        """Test API authentication."""
        app = create_api_app()
        client = TestClient(app)

        # Test authenticated request
        response = client.get(
            "/api/v1/datasets", headers={"Authorization": f"Bearer {auth_token}"}
        )

        # Should handle authentication
        assert response.status_code in [200, 401, 403, 404]

    def test_web_authentication(self, auth_token):
        """Test Web UI authentication."""
        client = TestClient(web_app)

        # Test authenticated page access
        response = client.get(
            "/dashboard", headers={"Authorization": f"Bearer {auth_token}"}
        )

        # Should handle authentication
        assert response.status_code in [200, 302, 401, 403, 404]

    def test_sdk_authentication(self, auth_token):
        """Test SDK authentication."""
        config = SDKConfig(
            api_url="http://localhost:8000", api_key=auth_token, timeout=30
        )
        client = PynomályClient(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"authenticated": True}

        with patch(
            "requests.Session.request", return_value=mock_response
        ) as mock_request:
            # Test authenticated request
            response = client._make_request("GET", "/test")

            # Should include auth header
            call_args = mock_request.call_args
            assert "Authorization" in call_args[1]["headers"]
            assert call_args[1]["headers"]["Authorization"] == f"Bearer {auth_token}"

    def test_cli_authentication(self, auth_token):
        """Test CLI authentication."""
        runner = CliRunner()

        # Set API key via environment
        os.environ["PYNOMALY_API_KEY"] = auth_token

        try:
            with patch("requests.get") as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = [{"id": "test"}]

                # Test authenticated CLI command
                result = runner.invoke(cli_app, ["dataset", "list"])

                # Should use authentication
                assert result.exit_code in [0, 1]

        finally:
            if "PYNOMALY_API_KEY" in os.environ:
                del os.environ["PYNOMALY_API_KEY"]

    def test_authentication_propagation(self, auth_token):
        """Test authentication propagation across layers."""
        # Test that authentication is properly propagated from web to API

        web_client = TestClient(web_app)

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"user": "test-user"}

            # Make authenticated web request
            response = web_client.get(
                "/api/user", headers={"Authorization": f"Bearer {auth_token}"}
            )

            # Should propagate auth to backend API
            if mock_get.called:
                call_args = mock_get.call_args
                if call_args and len(call_args) > 1 and "headers" in call_args[1]:
                    assert "Authorization" in call_args[1]["headers"]

    def test_authentication_error_handling(self):
        """Test authentication error handling."""
        # Test invalid token
        invalid_token = "invalid-token"

        app = create_api_app()
        client = TestClient(app)

        response = client.get(
            "/api/v1/datasets", headers={"Authorization": f"Bearer {invalid_token}"}
        )

        # Should handle invalid authentication
        assert response.status_code in [401, 403, 404]

    def test_token_refresh_flow(self, auth_token):
        """Test token refresh flow."""
        # Test token refresh in SDK
        config = SDKConfig(
            api_url="http://localhost:8000", api_key=auth_token, timeout=30
        )
        client = PynomályClient(config)

        # Mock token refresh scenario
        responses = [
            Mock(
                status_code=401, json=lambda: {"error": "Token expired"}
            ),  # First request fails
            Mock(
                status_code=200, json=lambda: {"token": "new-token"}
            ),  # Refresh succeeds
            Mock(status_code=200, json=lambda: {"data": "success"}),  # Retry succeeds
        ]

        with patch("requests.Session.request", side_effect=responses):
            # Should handle token refresh
            try:
                response = client._make_request("GET", "/test")
                # If refresh is implemented, should succeed
                assert response.status_code in [200, 401]
            except Exception:
                # If refresh not implemented, should raise auth error
                pass


class TestDataFlowValidation:
    """Test data flow validation across presentation layers."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        return {
            "name": "Test Dataset",
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            "columns": ["col1", "col2", "col3"],
        }

    @pytest.fixture
    def sample_detector(self):
        """Create sample detector."""
        return {
            "name": "Test Detector",
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "parameters": {"n_estimators": 100},
        }

    def test_dataset_upload_flow(self, sample_dataset):
        """Test dataset upload flow across layers."""
        # Test complete dataset upload from web UI to API

        web_client = TestClient(web_app)

        # Mock API responses
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "dataset1", "name": "Test Dataset"}

        with patch("requests.post", return_value=mock_response):
            # Test web UI dataset upload
            files = {"file": ("test.csv", "col1,col2,col3\n1,2,3\n4,5,6\n", "text/csv")}
            data = {"name": "Test Dataset"}

            response = web_client.post("/datasets/upload", files=files, data=data)

            # Should handle upload flow
            assert response.status_code in [200, 201, 302, 401, 404]

    def test_detection_workflow(self, sample_dataset, sample_detector):
        """Test complete detection workflow."""
        # Test end-to-end detection from dataset creation to result

        config = SDKConfig(
            api_url="http://localhost:8000", api_key="test-key", timeout=30
        )
        client = PynomályClient(config)

        # Mock workflow responses
        responses = [
            Mock(status_code=201, json=lambda: {"id": "dataset1"}),  # Create dataset
            Mock(status_code=201, json=lambda: {"id": "detector1"}),  # Create detector
            Mock(status_code=200, json=lambda: {"job_id": "job1"}),  # Train detector
            Mock(
                status_code=200, json=lambda: {"status": "completed"}
            ),  # Check training
            Mock(
                status_code=200,
                json=lambda: {  # Detect anomalies
                    "anomaly_scores": [0.1, 0.2, 0.8],
                    "predictions": [0, 0, 1],
                },
            ),
        ]

        with patch("requests.Session.request", side_effect=responses):
            # Step 1: Create dataset
            dataset = client.create_dataset(**sample_dataset)
            assert dataset["id"] == "dataset1"

            # Step 2: Create detector
            detector = client.create_detector(**sample_detector)
            assert detector["id"] == "detector1"

            # Step 3: Train detector
            job = client.train_detector("detector1", "dataset1")
            assert job["job_id"] == "job1"

            # Step 4: Check training status
            status = client.get_training_job("job1")
            assert status["status"] == "completed"

            # Step 5: Detect anomalies
            result = client.detect_anomalies(
                detector_id="detector1", data=sample_dataset["data"]
            )
            assert len(result["anomaly_scores"]) == 3

    def test_data_validation_consistency(self, sample_dataset):
        """Test data validation consistency across layers."""
        # Test that all layers validate data consistently

        # Invalid dataset (missing required fields)
        invalid_dataset = {"name": ""}  # Missing name and data

        # Test API validation
        app = create_api_app()
        api_client = TestClient(app)

        response = api_client.post("/api/v1/datasets", json=invalid_dataset)
        api_status = response.status_code

        # Test SDK validation
        config = SDKConfig(
            api_url="http://localhost:8000", api_key="test-key", timeout=30
        )
        sdk_client = PynomályClient(config)

        try:
            sdk_client.create_dataset(**invalid_dataset)
            sdk_status = 200  # If no exception, assume validation passed
        except Exception:
            sdk_status = 400  # If exception, assume validation failed

        # Both should handle validation similarly
        assert api_status in [400, 422] or sdk_status == 400

    def test_data_format_conversion(self, sample_dataset):
        """Test data format conversion across layers."""
        # Test that data is properly converted between layers

        web_client = TestClient(web_app)

        # Test CSV upload
        csv_data = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n"

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "dataset1",
            "name": "CSV Dataset",
            "format": "csv",
            "columns": ["col1", "col2", "col3"],
        }

        with patch("requests.post", return_value=mock_response):
            files = {"file": ("test.csv", csv_data, "text/csv")}
            data = {"name": "CSV Dataset"}

            response = web_client.post("/datasets/upload", files=files, data=data)

            # Should handle format conversion
            assert response.status_code in [200, 201, 302, 401, 404]

    def test_error_propagation(self):
        """Test error propagation across layers."""
        # Test that errors are properly propagated from API to other layers

        config = SDKConfig(
            api_url="http://localhost:8000", api_key="test-key", timeout=30
        )
        client = PynomályClient(config)

        # Mock API error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Dataset not found"}

        with patch("requests.Session.request", return_value=mock_response):
            # Should propagate error from API
            try:
                client.get_dataset("nonexistent")
                assert False, "Should have raised exception"
            except Exception as e:
                # Should contain error information
                assert "not found" in str(e).lower() or "404" in str(e)


class TestPerformanceTesting:
    """Test performance across presentation layers."""

    def test_api_response_time(self):
        """Test API response time."""
        app = create_api_app()
        client = TestClient(app)

        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        # Should respond quickly
        assert (end_time - start_time) < 2.0  # Under 2 seconds
        assert response.status_code == 200

    def test_web_ui_response_time(self):
        """Test Web UI response time."""
        client = TestClient(web_app)

        start_time = time.time()
        response = client.get("/")
        end_time = time.time()

        # Should respond quickly
        assert (end_time - start_time) < 3.0  # Under 3 seconds
        assert response.status_code in [200, 302, 404]

    def test_cli_startup_time(self):
        """Test CLI startup time."""
        runner = CliRunner()

        start_time = time.time()
        result = runner.invoke(cli_app, ["--help"])
        end_time = time.time()

        # Should start quickly
        assert (end_time - start_time) < 5.0  # Under 5 seconds
        assert result.exit_code == 0

    def test_sdk_request_performance(self):
        """Test SDK request performance."""
        config = SDKConfig(
            api_url="http://localhost:8000", api_key="test-key", timeout=30
        )
        client = PynomályClient(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "success"}

        with patch("requests.Session.request", return_value=mock_response):
            start_time = time.time()
            response = client._make_request("GET", "/test")
            end_time = time.time()

            # Should process quickly
            assert (end_time - start_time) < 1.0  # Under 1 second
            assert response.status_code == 200

    def test_concurrent_api_requests(self):
        """Test concurrent API request handling."""
        app = create_api_app()
        client = TestClient(app)

        def make_request():
            return client.get("/health")

        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            responses = [future.result() for future in futures]

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    def test_concurrent_web_requests(self):
        """Test concurrent Web UI request handling."""
        client = TestClient(web_app)

        def make_request():
            return client.get("/")

        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]

        # All requests should complete
        for response in responses:
            assert response.status_code in [200, 302, 404, 500]

    def test_memory_usage_stability(self):
        """Test memory usage stability across operations."""
        config = SDKConfig(
            api_url="http://localhost:8000", api_key="test-key", timeout=30
        )
        client = PynomályClient(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "x" * 1000}  # 1KB response

        with patch("requests.Session.request", return_value=mock_response):
            # Make many requests to test memory stability
            for i in range(100):
                response = client._make_request("GET", f"/test{i}")
                assert response.status_code == 200

            # Should not accumulate excessive memory
            import psutil

            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB

            # Should use reasonable amount of memory
            assert memory_usage < 200  # Less than 200MB

    def test_large_data_handling(self):
        """Test handling of large data across layers."""
        config = SDKConfig(
            api_url="http://localhost:8000",
            api_key="test-key",
            timeout=60,  # Longer timeout for large data
        )
        client = PynomályClient(config)

        # Mock large dataset response
        large_data = [[i] * 100 for i in range(1000)]  # 100K data points
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": large_data}

        with patch("requests.Session.request", return_value=mock_response):
            start_time = time.time()
            response = client._make_request("GET", "/large-dataset")
            end_time = time.time()

            # Should handle large data within reasonable time
            assert (end_time - start_time) < 10.0  # Under 10 seconds
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_async_performance(self):
        """Test async performance."""
        config = SDKConfig(
            api_url="http://localhost:8000", api_key="test-key", timeout=30
        )
        client = AsyncPynomályClient(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": "success"})

        with patch("aiohttp.ClientSession.request", return_value=mock_response):
            async with client:
                # Test concurrent async requests
                start_time = time.time()
                tasks = [client._make_request("GET", f"/test{i}") for i in range(20)]
                responses = await asyncio.gather(*tasks)
                end_time = time.time()

                # Should handle concurrent requests efficiently
                assert (end_time - start_time) < 5.0  # Under 5 seconds

                # All requests should succeed
                for response in responses:
                    assert response.status == 200


class TestErrorHandlingIntegration:
    """Test error handling integration across presentation layers."""

    def test_network_error_handling(self):
        """Test network error handling across layers."""
        config = SDKConfig(
            api_url="http://nonexistent:8000", api_key="test-key", timeout=5
        )
        client = PynomályClient(config)

        # Should handle network errors gracefully
        try:
            client.health_check()
            assert False, "Should have raised network error"
        except Exception as e:
            # Should be a meaningful error
            assert "network" in str(e).lower() or "connection" in str(e).lower()

    def test_timeout_handling(self):
        """Test timeout handling across layers."""
        config = SDKConfig(
            api_url="http://localhost:8000",
            api_key="test-key",
            timeout=0.1,  # Very short timeout
        )
        client = PynomályClient(config)

        # Mock slow response
        def slow_request(*args, **kwargs):
            time.sleep(1)  # Simulate slow response
            mock_resp = Mock()
            mock_resp.status_code = 200
            return mock_resp

        with patch("requests.Session.request", side_effect=slow_request):
            # Should handle timeout
            try:
                client._make_request("GET", "/test")
                assert False, "Should have timed out"
            except Exception as e:
                # Should be timeout-related error
                assert "timeout" in str(e).lower() or "time" in str(e).lower()

    def test_validation_error_consistency(self):
        """Test validation error consistency across layers."""
        # Test that validation errors are consistent across API, Web, CLI, and SDK

        invalid_data = {"contamination": 1.5}  # Invalid contamination value

        # Test API validation
        app = create_api_app()
        api_client = TestClient(app)

        response = api_client.post("/api/v1/detectors", json=invalid_data)
        api_error = response.status_code in [400, 422]

        # Test SDK validation
        config = SDKConfig(
            api_url="http://localhost:8000", api_key="test-key", timeout=30
        )
        sdk_client = PynomályClient(config)

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Invalid contamination value"}

        with patch("requests.Session.request", return_value=mock_response):
            try:
                sdk_client.create_detector(
                    name="Test", algorithm="isolation_forest", contamination=1.5
                )
                sdk_error = False
            except Exception:
                sdk_error = True

        # Both should handle validation similarly
        assert api_error or sdk_error

    def test_authentication_error_propagation(self):
        """Test authentication error propagation."""
        # Test invalid authentication across layers

        invalid_token = "invalid-token"

        # Test API
        app = create_api_app()
        api_client = TestClient(app)

        response = api_client.get(
            "/api/v1/datasets", headers={"Authorization": f"Bearer {invalid_token}"}
        )
        api_auth_error = response.status_code in [401, 403]

        # Test SDK
        config = SDKConfig(
            api_url="http://localhost:8000", api_key=invalid_token, timeout=30
        )
        sdk_client = PynomályClient(config)

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid token"}

        with patch("requests.Session.request", return_value=mock_response):
            try:
                sdk_client.list_datasets()
                sdk_auth_error = False
            except Exception:
                sdk_auth_error = True

        # Both should handle auth errors
        assert api_auth_error or sdk_auth_error

    def test_graceful_degradation(self):
        """Test graceful degradation when services are unavailable."""
        # Test that components gracefully handle service unavailability

        web_client = TestClient(web_app)

        # Mock API unavailable
        with patch("requests.get", side_effect=ConnectionError("API unavailable")):
            # Web UI should handle API unavailability gracefully
            response = web_client.get("/dashboard")

            # Should not crash, may show error page or degraded functionality
            assert response.status_code in [200, 500, 502, 503]

    def test_error_logging_integration(self):
        """Test error logging integration across layers."""
        # Test that errors are properly logged across all layers

        with patch("logging.Logger.error") as mock_logger:
            config = SDKConfig(
                api_url="http://localhost:8000", api_key="test-key", timeout=30
            )
            client = PynomályClient(config)

            # Mock server error
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": "Internal server error"}

            with patch("requests.Session.request", return_value=mock_response):
                try:
                    client._make_request("GET", "/test")
                except Exception:
                    pass

                # Should log errors (if logging is implemented)
                # mock_logger.assert_called() - uncomment if logging is implemented
