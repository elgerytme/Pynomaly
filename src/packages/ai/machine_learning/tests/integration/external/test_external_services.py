"""Integration tests for external service integrations."""

import pytest
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from test_utilities.integration_test_base import IntegrationTestBase
from test_utilities.fixtures.external_services import (
    mock_mlflow_client,
    mock_s3_client,
    mock_redis_client,
    mock_database_connection
)


class TestExternalServiceIntegrations(IntegrationTestBase):
    """Integration tests for external service dependencies."""

    @pytest.fixture
    def sample_model_data(self):
        """Sample model data for testing."""
        return {
            "model_id": str(uuid4()),
            "name": "test_model",
            "version": "1.0.0",
            "algorithm": "IsolationForest",
            "parameters": {"contamination": 0.1, "n_estimators": 100},
            "metrics": {"accuracy": 0.95, "f1_score": 0.92},
            "artifacts": {"model_file": "model.pkl", "config_file": "config.json"}
        }

    @pytest.fixture
    def sample_experiment_data(self):
        """Sample experiment data for testing."""
        return {
            "experiment_id": str(uuid4()),
            "name": "test_experiment",
            "description": "Integration test experiment",
            "parameters": {"learning_rate": 0.01, "batch_size": 32},
            "metrics": {"loss": 0.05, "accuracy": 0.98}
        }

    @pytest.mark.asyncio
    async def test_mlflow_integration_experiment_creation(
        self,
        mock_mlflow_client: AsyncMock,
        sample_experiment_data: Dict[str, Any]
    ):
        """Test MLflow experiment creation integration."""
        # Mock MLflow client responses
        mock_mlflow_client.create_experiment.return_value = {
            "experiment_id": sample_experiment_data["experiment_id"]
        }
        mock_mlflow_client.start_run.return_value = {
            "run_id": str(uuid4()),
            "experiment_id": sample_experiment_data["experiment_id"]
        }
        
        # Test experiment creation through service
        from machine_learning.infrastructure.external.mlflow_adapter import MLflowAdapter
        
        adapter = MLflowAdapter(client=mock_mlflow_client)
        
        experiment_id = await adapter.create_experiment(
            name=sample_experiment_data["name"],
            description=sample_experiment_data["description"]
        )
        
        assert experiment_id == sample_experiment_data["experiment_id"]
        mock_mlflow_client.create_experiment.assert_called_once_with(
            name=sample_experiment_data["name"],
            tags={"description": sample_experiment_data["description"]}
        )

    @pytest.mark.asyncio
    async def test_mlflow_integration_run_tracking(
        self,
        mock_mlflow_client: AsyncMock,
        sample_experiment_data: Dict[str, Any]
    ):
        """Test MLflow run tracking integration."""
        run_id = str(uuid4())
        mock_mlflow_client.start_run.return_value = {"run_id": run_id}
        mock_mlflow_client.log_param.return_value = None
        mock_mlflow_client.log_metric.return_value = None
        mock_mlflow_client.end_run.return_value = None
        
        from machine_learning.infrastructure.external.mlflow_adapter import MLflowAdapter
        
        adapter = MLflowAdapter(client=mock_mlflow_client)
        
        # Start run
        started_run_id = await adapter.start_run(
            experiment_id=sample_experiment_data["experiment_id"],
            run_name="test_run"
        )
        
        assert started_run_id == run_id
        
        # Log parameters
        for key, value in sample_experiment_data["parameters"].items():
            await adapter.log_parameter(run_id, key, value)
        
        # Log metrics
        for key, value in sample_experiment_data["metrics"].items():
            await adapter.log_metric(run_id, key, value)
        
        # End run
        await adapter.end_run(run_id, status="FINISHED")
        
        # Verify all calls were made
        mock_mlflow_client.start_run.assert_called_once()
        assert mock_mlflow_client.log_param.call_count == len(sample_experiment_data["parameters"])
        assert mock_mlflow_client.log_metric.call_count == len(sample_experiment_data["metrics"])
        mock_mlflow_client.end_run.assert_called_once_with(run_id, "FINISHED")

    @pytest.mark.asyncio
    async def test_s3_integration_model_storage(
        self,
        mock_s3_client: AsyncMock,
        sample_model_data: Dict[str, Any]
    ):
        """Test S3 model storage integration."""
        # Mock S3 responses
        mock_s3_client.put_object.return_value = {
            "ETag": "test-etag",
            "VersionId": "test-version"
        }
        mock_s3_client.get_object.return_value = {
            "Body": AsyncMock(read=AsyncMock(return_value=b"model_data"))
        }
        mock_s3_client.head_object.return_value = {
            "ContentLength": 1024,
            "LastModified": "2024-01-01T00:00:00Z"
        }
        
        from machine_learning.infrastructure.external.s3_storage_adapter import S3StorageAdapter
        
        adapter = S3StorageAdapter(client=mock_s3_client, bucket_name="ml-models")
        
        # Test model upload
        model_key = f"models/{sample_model_data['model_id']}/model.pkl"
        upload_result = await adapter.upload_model(
            model_id=sample_model_data["model_id"],
            model_data=b"serialized_model_data",
            metadata={"version": sample_model_data["version"]}
        )
        
        assert upload_result["success"] is True
        assert "s3_key" in upload_result
        
        # Test model download
        download_result = await adapter.download_model(
            model_id=sample_model_data["model_id"]
        )
        
        assert download_result["success"] is True
        assert download_result["data"] == b"model_data"
        
        # Verify S3 calls
        mock_s3_client.put_object.assert_called_once()
        mock_s3_client.get_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_integration_caching(
        self,
        mock_redis_client: AsyncMock
    ):
        """Test Redis caching integration."""
        # Mock Redis responses
        mock_redis_client.set.return_value = True
        mock_redis_client.get.return_value = json.dumps({"cached": "data"}).encode()
        mock_redis_client.delete.return_value = 1
        mock_redis_client.exists.return_value = True
        
        from machine_learning.infrastructure.external.redis_cache_adapter import RedisCacheAdapter
        
        adapter = RedisCacheAdapter(client=mock_redis_client)
        
        # Test cache set
        cache_key = "model:predictions:user123"
        cache_data = {"predictions": [0.1, 0.9], "model_version": "1.0.0"}
        
        success = await adapter.set_cache(
            key=cache_key,
            data=cache_data,
            ttl_seconds=3600
        )
        
        assert success is True
        
        # Test cache get
        retrieved_data = await adapter.get_cache(cache_key)
        
        assert retrieved_data == {"cached": "data"}
        
        # Test cache delete
        deleted = await adapter.delete_cache(cache_key)
        
        assert deleted is True
        
        # Verify Redis calls
        mock_redis_client.set.assert_called_once()
        mock_redis_client.get.assert_called_once()
        mock_redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_integration_model_metadata(
        self,
        mock_database_connection: AsyncMock,
        sample_model_data: Dict[str, Any]
    ):
        """Test database integration for model metadata."""
        # Mock database responses
        mock_cursor = AsyncMock()
        mock_cursor.fetchone.return_value = sample_model_data
        mock_cursor.fetchall.return_value = [sample_model_data]
        mock_cursor.rowcount = 1
        
        mock_database_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_database_connection.commit.return_value = None
        
        from machine_learning.infrastructure.repositories.model_metadata_repository import ModelMetadataRepository
        
        repository = ModelMetadataRepository(connection=mock_database_connection)
        
        # Test model creation
        created_model = await repository.create_model(
            name=sample_model_data["name"],
            version=sample_model_data["version"],
            algorithm=sample_model_data["algorithm"],
            parameters=sample_model_data["parameters"],
            metrics=sample_model_data["metrics"]
        )
        
        assert created_model["model_id"] is not None
        
        # Test model retrieval
        retrieved_model = await repository.get_model(sample_model_data["model_id"])
        
        assert retrieved_model == sample_model_data
        
        # Test model listing
        models = await repository.list_models(limit=10)
        
        assert len(models) == 1
        assert models[0] == sample_model_data
        
        # Verify database calls
        assert mock_cursor.execute.call_count >= 3

    @pytest.mark.asyncio
    async def test_http_webhook_integration(self):
        """Test HTTP webhook integration for model notifications."""
        webhook_url = "https://api.example.com/webhooks/model-update"
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"received": True}
            mock_post.return_value.__aenter__.return_value = mock_response
            
            from machine_learning.infrastructure.external.webhook_adapter import WebhookAdapter
            
            adapter = WebhookAdapter()
            
            # Test webhook notification
            result = await adapter.send_model_update_notification(
                webhook_url=webhook_url,
                model_id=str(uuid4()),
                event_type="model_deployed",
                payload={
                    "model_name": "test_model",
                    "version": "1.0.0",
                    "status": "active"
                }
            )
            
            assert result["success"] is True
            assert result["status_code"] == 200
            
            # Verify HTTP call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert webhook_url in str(call_args)

    @pytest.mark.asyncio
    async def test_external_service_circuit_breaker(
        self,
        mock_mlflow_client: AsyncMock
    ):
        """Test circuit breaker pattern for external service failures."""
        # Simulate service failures
        mock_mlflow_client.create_experiment.side_effect = [
            Exception("Service unavailable"),
            Exception("Service unavailable"),
            Exception("Service unavailable"),
            {"experiment_id": "exp_123"}  # Recovery
        ]
        
        from machine_learning.infrastructure.external.circuit_breaker import CircuitBreaker
        from machine_learning.infrastructure.external.mlflow_adapter import MLflowAdapter
        
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,
            expected_exception=Exception
        )
        
        adapter = MLflowAdapter(
            client=mock_mlflow_client,
            circuit_breaker=circuit_breaker
        )
        
        # First 3 calls should fail and trigger circuit breaker
        for _ in range(3):
            with pytest.raises(Exception):
                await adapter.create_experiment("test_experiment")
        
        # Circuit should be open now
        assert circuit_breaker.state == "OPEN"
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should succeed and close circuit
        result = await adapter.create_experiment("test_experiment")
        assert result == {"experiment_id": "exp_123"}
        assert circuit_breaker.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_external_service_retry_mechanism(
        self,
        mock_s3_client: AsyncMock
    ):
        """Test retry mechanism for transient failures."""
        # Simulate transient failures followed by success
        mock_s3_client.put_object.side_effect = [
            Exception("Temporary network error"),
            Exception("Temporary network error"),
            {"ETag": "success-etag"}
        ]
        
        from machine_learning.infrastructure.external.retry_decorator import with_retry
        from machine_learning.infrastructure.external.s3_storage_adapter import S3StorageAdapter
        
        adapter = S3StorageAdapter(client=mock_s3_client, bucket_name="ml-models")
        
        # Wrap upload method with retry decorator
        adapter.upload_model = with_retry(
            max_attempts=3,
            backoff_factor=0.1,
            exceptions=(Exception,)
        )(adapter.upload_model)
        
        # Should succeed after retries
        result = await adapter.upload_model(
            model_id=str(uuid4()),
            model_data=b"test_data"
        )
        
        assert result["success"] is True
        assert mock_s3_client.put_object.call_count == 3

    @pytest.mark.asyncio
    async def test_service_health_checks(self):
        """Test health checks for external services."""
        from machine_learning.infrastructure.external.health_checker import ExternalServiceHealthChecker
        
        health_checker = ExternalServiceHealthChecker()
        
        # Mock service endpoints
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Healthy service
            mock_response_healthy = AsyncMock()
            mock_response_healthy.status = 200
            mock_response_healthy.json.return_value = {"status": "healthy"}
            
            # Unhealthy service
            mock_response_unhealthy = AsyncMock()
            mock_response_unhealthy.status = 503
            
            mock_get.side_effect = [
                mock_response_healthy.__aenter__.return_value,
                mock_response_unhealthy.__aenter__.return_value
            ]
            
            # Check healthy service
            mlflow_health = await health_checker.check_mlflow_health()
            assert mlflow_health["status"] == "healthy"
            
            # Check unhealthy service
            s3_health = await health_checker.check_s3_health()
            assert s3_health["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_service_authentication_integration(
        self,
        mock_mlflow_client: AsyncMock
    ):
        """Test authentication integration with external services."""
        from machine_learning.infrastructure.external.auth_manager import ExternalServiceAuthManager
        
        auth_manager = ExternalServiceAuthManager()
        
        # Test token-based authentication
        with patch.object(auth_manager, 'get_access_token', return_value="test_token"):
            token = await auth_manager.get_mlflow_token()
            assert token == "test_token"
        
        # Test service account authentication
        with patch.object(auth_manager, 'get_service_account_credentials') as mock_creds:
            mock_creds.return_value = {
                "access_key": "test_access_key",
                "secret_key": "test_secret_key"
            }
            
            credentials = await auth_manager.get_s3_credentials()
            assert credentials["access_key"] == "test_access_key"

    @pytest.mark.asyncio
    async def test_service_configuration_management(self):
        """Test external service configuration management."""
        from machine_learning.infrastructure.external.config_manager import ExternalServiceConfigManager
        
        config_manager = ExternalServiceConfigManager()
        
        # Test configuration loading
        config = await config_manager.load_service_config("mlflow")
        
        assert "endpoint_url" in config
        assert "timeout" in config
        assert "max_retries" in config
        
        # Test configuration validation
        is_valid = config_manager.validate_config("mlflow", config)
        assert is_valid is True
        
        # Test invalid configuration
        invalid_config = {"endpoint_url": ""}
        is_valid = config_manager.validate_config("mlflow", invalid_config)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_service_monitoring_metrics(
        self,
        mock_mlflow_client: AsyncMock
    ):
        """Test monitoring metrics for external service calls."""
        from machine_learning.infrastructure.external.metrics_collector import ServiceMetricsCollector
        
        metrics_collector = ServiceMetricsCollector()
        
        # Mock successful service call
        mock_mlflow_client.create_experiment.return_value = {"experiment_id": "exp_123"}
        
        from machine_learning.infrastructure.external.mlflow_adapter import MLflowAdapter
        
        adapter = MLflowAdapter(
            client=mock_mlflow_client,
            metrics_collector=metrics_collector
        )
        
        # Make service call
        result = await adapter.create_experiment("test_experiment")
        
        # Verify metrics were collected
        metrics = metrics_collector.get_metrics()
        
        assert "mlflow_calls_total" in metrics
        assert "mlflow_call_duration" in metrics
        assert "mlflow_success_rate" in metrics
        
        assert metrics["mlflow_calls_total"] >= 1
        assert metrics["mlflow_success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_batch_external_operations(
        self,
        mock_s3_client: AsyncMock
    ):
        """Test batch operations with external services."""
        # Mock batch upload responses
        mock_s3_client.put_object.return_value = {"ETag": "test-etag"}
        
        from machine_learning.infrastructure.external.s3_storage_adapter import S3StorageAdapter
        
        adapter = S3StorageAdapter(client=mock_s3_client, bucket_name="ml-models")
        
        # Batch upload multiple models
        model_data_list = [
            {"model_id": str(uuid4()), "data": b"model_1_data"},
            {"model_id": str(uuid4()), "data": b"model_2_data"},
            {"model_id": str(uuid4()), "data": b"model_3_data"}
        ]
        
        results = await adapter.batch_upload_models(model_data_list)
        
        assert len(results) == 3
        assert all(result["success"] for result in results)
        assert mock_s3_client.put_object.call_count == 3

    @pytest.mark.asyncio
    async def test_service_connection_pooling(self):
        """Test connection pooling for external services."""
        from machine_learning.infrastructure.external.connection_pool import ExternalServiceConnectionPool
        
        connection_pool = ExternalServiceConnectionPool()
        
        # Test getting multiple connections
        connections = []
        for _ in range(5):
            conn = await connection_pool.get_connection("mlflow")
            connections.append(conn)
        
        # All connections should be valid
        for conn in connections:
            assert conn is not None
            assert hasattr(conn, 'client')
        
        # Return connections to pool
        for conn in connections:
            await connection_pool.return_connection("mlflow", conn)
        
        # Pool should be able to reuse connections
        reused_conn = await connection_pool.get_connection("mlflow")
        assert reused_conn is not None

    @pytest.mark.asyncio
    async def test_external_service_timeouts(
        self,
        mock_mlflow_client: AsyncMock
    ):
        """Test timeout handling for external service calls."""
        # Simulate timeout
        mock_mlflow_client.create_experiment.side_effect = asyncio.TimeoutError()
        
        from machine_learning.infrastructure.external.mlflow_adapter import MLflowAdapter
        
        adapter = MLflowAdapter(client=mock_mlflow_client, timeout=1.0)
        
        # Should handle timeout gracefully
        with pytest.raises(asyncio.TimeoutError):
            await adapter.create_experiment("test_experiment")

    @pytest.mark.asyncio
    async def test_service_rate_limiting(
        self,
        mock_s3_client: AsyncMock
    ):
        """Test rate limiting for external service calls."""
        from machine_learning.infrastructure.external.rate_limiter import RateLimiter
        from machine_learning.infrastructure.external.s3_storage_adapter import S3StorageAdapter
        
        rate_limiter = RateLimiter(max_calls=2, time_window=1.0)
        adapter = S3StorageAdapter(
            client=mock_s3_client,
            bucket_name="ml-models",
            rate_limiter=rate_limiter
        )
        
        mock_s3_client.put_object.return_value = {"ETag": "test-etag"}
        
        # First two calls should succeed immediately
        start_time = asyncio.get_event_loop().time()
        
        await adapter.upload_model(str(uuid4()), b"data1")
        await adapter.upload_model(str(uuid4()), b"data2")
        
        first_two_duration = asyncio.get_event_loop().time() - start_time
        assert first_two_duration < 0.1  # Should be very fast
        
        # Third call should be rate limited
        start_time = asyncio.get_event_loop().time()
        await adapter.upload_model(str(uuid4()), b"data3")
        rate_limited_duration = asyncio.get_event_loop().time() - start_time
        
        assert rate_limited_duration >= 1.0  # Should wait for rate limit reset