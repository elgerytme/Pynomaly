"""Integration tests for ML Lifecycle API endpoints."""

import pytest
from datetime import datetime, UTC
from typing import Any, Dict
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from test_utilities.api_test_base import APITestBase
from test_utilities.fixtures.api import (
    api_client,
    authenticated_client,
    mock_container,
    mock_auth_dependencies
)
from test_utilities.fixtures.data import sample_experiments, sample_model_data


class TestMLLifecycleEndpointsIntegration(APITestBase):
    """Integration tests for ML Lifecycle API endpoints."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with ML Lifecycle router."""
        from mlops.presentation.api.endpoints.advanced_ml_lifecycle import router
        
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def mock_ml_lifecycle_service(self):
        """Mock ML Lifecycle service."""
        mock = AsyncMock()
        mock.start_experiment.return_value = "exp_123"
        mock.start_run.return_value = "run_456"
        mock.log_parameter.return_value = None
        mock.log_metric.return_value = None
        mock.log_artifact.return_value = "/artifacts/test_artifact.pkl"
        mock.create_model_version.return_value = "model_v1.0.0"
        mock.promote_model_version.return_value = {
            "success": True,
            "model_version_id": "model_v1.0.0",
            "new_stage": "production",
            "new_status": "active",
            "validation_results": {"accuracy": 0.95},
            "promoted_by": "test_user",
            "promoted_at": datetime.now(UTC).isoformat()
        }
        mock.search_models.return_value = [
            {
                "id": "model_1",
                "name": "test_model", 
                "version": "1.0.0",
                "stage": "production"
            }
        ]
        mock.get_model_registry_stats.return_value = {
            "total_models": 10,
            "total_versions": 25,
            "average_versions_per_model": 2.5,
            "model_status_distribution": {"active": 8, "archived": 2},
            "version_status_distribution": {"production": 5, "staging": 10, "development": 10},
            "recent_models": [],
            "recent_versions": [],
            "performance_trends": {},
            "registry_health": {"status": "healthy"}
        }
        return mock

    @pytest.fixture
    def sample_experiment_request(self):
        """Sample experiment start request."""
        return {
            "name": "test_experiment",
            "description": "Test ML experiment",
            "experiment_type": "HYPERPARAMETER_TUNING",
            "objective": "maximize_accuracy",
            "auto_log_parameters": True,
            "auto_log_metrics": True,
            "auto_log_artifacts": True,
            "tags": ["test", "experiment"],
            "metadata": {"project": "test_project"}
        }

    @pytest.fixture
    def sample_run_request(self):
        """Sample run start request."""
        return {
            "run_name": "test_run",
            "detector_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "parameters": {"learning_rate": 0.01, "batch_size": 32},
            "tags": ["test", "run"],
            "description": "Test ML run"
        }

    @pytest.mark.asyncio
    async def test_start_experiment_success(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock,
        sample_experiment_request: Dict[str, Any]
    ):
        """Test successful experiment creation."""
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                "/advanced-ml-lifecycle/experiments/start",
                json=sample_experiment_request
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["experiment_id"] == "exp_123"
        assert "started successfully" in data["message"]
        
        # Verify service was called correctly
        mock_ml_lifecycle_service.start_experiment.assert_called_once()
        call_args = mock_ml_lifecycle_service.start_experiment.call_args
        assert call_args.kwargs["name"] == "test_experiment"
        assert call_args.kwargs["experiment_type"] == "HYPERPARAMETER_TUNING"

    @pytest.mark.asyncio
    async def test_start_experiment_validation_error(
        self,
        authenticated_client: TestClient
    ):
        """Test experiment creation with validation errors."""
        invalid_request = {
            "name": "",  # Empty name should fail validation
            "description": "Test",
            "objective": "test"
        }
        
        response = authenticated_client.post(
            "/advanced-ml-lifecycle/experiments/start",
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_start_run_success(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock,
        sample_run_request: Dict[str, Any]
    ):
        """Test successful run creation."""
        experiment_id = "exp_123"
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                f"/advanced-ml-lifecycle/experiments/{experiment_id}/runs/start",
                json=sample_run_request
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["run_id"] == "run_456"
        assert data["experiment_id"] == experiment_id
        assert "started successfully" in data["message"]

    @pytest.mark.asyncio
    async def test_log_parameter_success(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test successful parameter logging."""
        run_id = "run_456"
        parameter_data = {
            "key": "learning_rate",
            "value": 0.01
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                f"/advanced-ml-lifecycle/runs/{run_id}/parameters",
                json=parameter_data
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "logged successfully" in data["message"]
        mock_ml_lifecycle_service.log_parameter.assert_called_once_with(
            run_id, "learning_rate", 0.01
        )

    @pytest.mark.asyncio
    async def test_log_metric_success(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test successful metric logging."""
        run_id = "run_456"
        metric_data = {
            "key": "accuracy",
            "value": 0.95,
            "step": 100,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                f"/advanced-ml-lifecycle/runs/{run_id}/metrics",
                json=metric_data
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "logged successfully" in data["message"]
        mock_ml_lifecycle_service.log_metric.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_artifact_success(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test successful artifact logging."""
        run_id = "run_456"
        artifact_data = {
            "artifact_name": "model_weights",
            "artifact_data": {"weights": [1, 2, 3]},
            "artifact_type": "json",
            "metadata": {"size": "3KB"}
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                f"/advanced-ml-lifecycle/runs/{run_id}/artifacts",
                json=artifact_data
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["artifact_path"] == "/artifacts/test_artifact.pkl"
        assert "logged successfully" in data["message"]

    @pytest.mark.asyncio
    async def test_end_run_success(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test successful run ending."""
        run_id = "run_456"
        end_request = {
            "status": "FINISHED",
            "end_time": datetime.now(UTC).isoformat()
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                f"/advanced-ml-lifecycle/runs/{run_id}/end",
                json=end_request
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "ended successfully" in data["message"]
        mock_ml_lifecycle_service.end_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_model_version_success(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test successful model version creation."""
        version_request = {
            "model_name": "test_model",
            "run_id": "run_456",
            "model_path": "/models/test_model.pkl",
            "performance_metrics": {"accuracy": 0.95, "f1_score": 0.92},
            "description": "Test model version",
            "tags": ["test", "v1"],
            "auto_version": True
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                "/advanced-ml-lifecycle/model-versions",
                json=version_request
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_version_id"] == "model_v1.0.0"
        assert "created for" in data["message"]

    @pytest.mark.asyncio
    async def test_promote_model_version_success(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test successful model version promotion."""
        model_version_id = "model_v1.0.0"
        promotion_request = {
            "stage": "production",
            "approval_workflow": True,
            "validation_tests": ["accuracy_test", "performance_test"]
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                f"/advanced-ml-lifecycle/model-versions/{model_version_id}/promote",
                json=promotion_request
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["new_stage"] == "production"
        assert data["validation_results"]["accuracy"] == 0.95

    @pytest.mark.asyncio
    async def test_search_models_success(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test successful model search."""
        search_request = {
            "query": "test_model",
            "max_results": 10,
            "filter_dict": {"stage": "production"},
            "order_by": ["created_at"]
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                "/advanced-ml-lifecycle/models/search",
                json=search_request
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 1
        assert data[0]["name"] == "test_model"
        assert data[0]["stage"] == "production"

    @pytest.mark.asyncio
    async def test_get_model_registry_stats_success(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test successful model registry stats retrieval."""
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.get("/advanced-ml-lifecycle/registry/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_models"] == 10
        assert data["total_versions"] == 25
        assert data["average_versions_per_model"] == 2.5
        assert data["registry_health"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_health_endpoint(
        self,
        authenticated_client: TestClient
    ):
        """Test health endpoint."""
        response = authenticated_client.get("/advanced-ml-lifecycle/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "advanced_ml_lifecycle"
        assert "features" in data
        assert "experiment_tracking" in data["features"]

    @pytest.mark.asyncio
    async def test_get_capabilities_endpoint(
        self,
        authenticated_client: TestClient
    ):
        """Test capabilities endpoint."""
        response = authenticated_client.get("/advanced-ml-lifecycle/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "experiment_tracking" in data
        assert "model_versioning" in data
        assert "model_registry" in data
        assert data["experiment_tracking"]["auto_logging"] is True

    @pytest.mark.asyncio
    async def test_service_failure_handling(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock,
        sample_experiment_request: Dict[str, Any]
    ):
        """Test service failure handling."""
        mock_ml_lifecycle_service.start_experiment.side_effect = Exception("Service unavailable")
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                "/advanced-ml-lifecycle/experiments/start",
                json=sample_experiment_request
            )
        
        assert response.status_code == 400
        assert "Service unavailable" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_authentication_required(self, api_client: TestClient):
        """Test that authentication is required for all endpoints."""
        endpoints = [
            ("POST", "/advanced-ml-lifecycle/experiments/start", {"name": "test", "description": "test", "experiment_type": "A_B_TESTING", "objective": "test"}),
            ("POST", "/advanced-ml-lifecycle/experiments/exp_123/runs/start", {"run_name": "test", "detector_id": str(uuid4()), "dataset_id": str(uuid4()), "parameters": {}}),
            ("POST", "/advanced-ml-lifecycle/runs/run_456/parameters", {"key": "test", "value": 1}),
            ("POST", "/advanced-ml-lifecycle/runs/run_456/metrics", {"key": "accuracy", "value": 0.9}),
            ("POST", "/advanced-ml-lifecycle/model-versions", {"model_name": "test", "run_id": "run_456", "model_path": "/path", "performance_metrics": {}})
        ]
        
        for method, endpoint, data in endpoints:
            response = api_client.post(endpoint, json=data)
            assert response.status_code == 401  # Unauthorized

    @pytest.mark.asyncio
    async def test_permission_enforcement(
        self,
        api_client: TestClient,
        sample_experiment_request: Dict[str, Any]
    ):
        """Test that write permissions are enforced."""
        # Mock authenticated user but without write permissions
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_current_user", return_value={"username": "test_user"}):
            with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.require_write") as mock_require:
                mock_require.side_effect = Exception("Insufficient permissions")
                
                response = api_client.post(
                    "/advanced-ml-lifecycle/experiments/start",
                    json=sample_experiment_request
                )
                
                # Should be blocked by permission check
                assert response.status_code != 200

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test handling concurrent ML operations."""
        import asyncio
        
        run_id = "run_456"
        
        async def log_metric(key: str, value: float):
            metric_data = {"key": key, "value": value}
            with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
                return authenticated_client.post(
                    f"/advanced-ml-lifecycle/runs/{run_id}/metrics",
                    json=metric_data
                )
        
        # Log multiple metrics concurrently
        tasks = [
            log_metric("accuracy", 0.95),
            log_metric("precision", 0.92),
            log_metric("recall", 0.89),
            log_metric("f1_score", 0.90)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_large_artifact_handling(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test handling large artifacts."""
        run_id = "run_456"
        
        # Simulate large artifact data
        large_data = {"weights": list(range(10000))}
        artifact_data = {
            "artifact_name": "large_model",
            "artifact_data": large_data,
            "artifact_type": "json",
            "metadata": {"size": "large"}
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                f"/advanced-ml-lifecycle/runs/{run_id}/artifacts",
                json=artifact_data
            )
        
        assert response.status_code == 200
        mock_ml_lifecycle_service.log_artifact.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_version_validation(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test model version creation validation."""
        # Test with missing required fields
        invalid_request = {
            "model_name": "test_model",
            # Missing run_id, model_path, performance_metrics
        }
        
        response = authenticated_client.post(
            "/advanced-ml-lifecycle/model-versions",
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_experiment_type_validation(
        self,
        authenticated_client: TestClient
    ):
        """Test experiment type validation."""
        invalid_request = {
            "name": "test_experiment",
            "description": "Test",
            "experiment_type": "INVALID_TYPE",  # Invalid enum value
            "objective": "test"
        }
        
        response = authenticated_client.post(
            "/advanced-ml-lifecycle/experiments/start",
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_metric_timestamp_handling(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test metric logging with timestamps."""
        run_id = "run_456"
        
        # Test with explicit timestamp
        metric_with_timestamp = {
            "key": "accuracy",
            "value": 0.95,
            "step": 100,
            "timestamp": "2024-01-01T10:00:00Z"
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                f"/advanced-ml-lifecycle/runs/{run_id}/metrics",
                json=metric_with_timestamp
            )
        
        assert response.status_code == 200
        
        # Test without timestamp (should use current time)
        metric_without_timestamp = {
            "key": "loss",
            "value": 0.05,
            "step": 100
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                f"/advanced-ml-lifecycle/runs/{run_id}/metrics",
                json=metric_without_timestamp
            )
        
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_nested_run_support(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock
    ):
        """Test nested run creation."""
        experiment_id = "exp_123"
        parent_run_id = "parent_run_123"
        
        nested_run_request = {
            "run_name": "nested_run",
            "detector_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "parameters": {"nested": True},
            "parent_run_id": parent_run_id,
            "description": "Nested test run"
        }
        
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                f"/advanced-ml-lifecycle/experiments/{experiment_id}/runs/start",
                json=nested_run_request
            )
        
        assert response.status_code == 200
        
        # Verify parent_run_id was passed correctly
        call_args = mock_ml_lifecycle_service.start_run.call_args
        assert call_args.kwargs["parent_run_id"] == parent_run_id

    def test_api_documentation_coverage(self):
        """Test that all endpoints have proper documentation."""
        from mlops.presentation.api.endpoints.advanced_ml_lifecycle import router
        
        for route in router.routes:
            if hasattr(route, 'endpoint') and route.path != "/openapi.json":
                assert route.endpoint.__doc__ is not None, f"Endpoint {route.path} missing documentation"
                assert len(route.endpoint.__doc__.strip()) > 20, f"Endpoint {route.path} has insufficient documentation"

    @pytest.mark.asyncio
    async def test_response_model_compliance(
        self,
        authenticated_client: TestClient,
        mock_ml_lifecycle_service: AsyncMock,
        sample_experiment_request: Dict[str, Any]
    ):
        """Test that responses match defined models."""
        with patch("mlops.presentation.api.endpoints.advanced_ml_lifecycle.get_advanced_ml_lifecycle_service", return_value=mock_ml_lifecycle_service):
            response = authenticated_client.post(
                "/advanced-ml-lifecycle/experiments/start",
                json=sample_experiment_request
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure matches StartExperimentResponse model
        required_fields = ["experiment_id", "message"]
        for field in required_fields:
            assert field in data, f"Response missing required field: {field}"
            assert data[field] is not None, f"Response field {field} is null"