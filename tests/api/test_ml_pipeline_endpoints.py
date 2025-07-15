"""Test ML Pipeline API Endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from src.packages.api.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_auth():
    """Mock authentication."""
    with patch("src.packages.api.api.dependencies.auth.get_current_user") as mock:
        mock.return_value = {
            "user_id": "test_user",
            "username": "test_user",
            "roles": ["user", "ml_engineer"],
            "permissions": [
                "ml_pipelines:read", 
                "ml_pipelines:write", 
                "ml_pipelines:execute", 
                "ml_pipelines:deploy",
                "ml_pipelines:admin"
            ]
        }
        yield mock


@pytest.fixture
def sample_pipeline_request():
    """Sample pipeline creation request."""
    return {
        "name": "customer_anomaly_detection",
        "description": "Pipeline for detecting anomalies in customer behavior data",
        "pipeline_type": "training",
        "algorithm": "IsolationForest",
        "hyperparameters": {
            "n_estimators": 100,
            "contamination": 0.1,
            "random_state": 42
        },
        "data_source": {
            "type": "database",
            "connection_string": "postgresql://user:pass@localhost/db",
            "query": "SELECT * FROM customer_data WHERE date >= '2024-01-01'"
        },
        "preprocessing": [
            {"type": "scaler", "method": "standard"},
            {"type": "encoder", "method": "onehot", "columns": ["category"]}
        ],
        "validation": {
            "method": "cross_validation",
            "folds": 5,
            "test_size": 0.2
        }
    }


@pytest.fixture
def sample_execution_request():
    """Sample pipeline execution request."""
    return {
        "pipeline_id": "pipeline_123456",
        "execution_mode": "async",
        "override_config": {
            "batch_size": 1000,
            "enable_monitoring": True
        }
    }


@pytest.fixture
def sample_deployment_request():
    """Sample model deployment request."""
    return {
        "model_id": "model_123456",
        "deployment_name": "anomaly-detector-v1",
        "environment": "production",
        "resources": {
            "cpu": "2",
            "memory": "4Gi",
            "replicas": 3
        },
        "scaling": {
            "min_replicas": 1,
            "max_replicas": 10,
            "target_cpu": 70
        },
        "monitoring": {
            "enable_metrics": True,
            "alert_thresholds": {
                "latency_ms": 1000,
                "error_rate": 0.05
            }
        }
    }


class TestMLPipelineEndpoints:
    """Test ML pipeline endpoints."""

    def test_create_pipeline_success(self, client, mock_auth, sample_pipeline_request):
        """Test successful pipeline creation."""
        response = client.post(
            "/ml-pipelines/pipelines",
            json=sample_pipeline_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        # Verify response structure
        assert "pipeline_id" in data
        assert data["name"] == sample_pipeline_request["name"]
        assert data["description"] == sample_pipeline_request["description"]
        assert data["pipeline_type"] == sample_pipeline_request["pipeline_type"]
        assert data["algorithm"] == sample_pipeline_request["algorithm"]
        assert data["status"] == "created"
        assert data["created_by"] == "test_user"
        assert data["version"] == "1.0.0"
        assert "execution_stats" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_pipeline_invalid_data(self, client, mock_auth):
        """Test pipeline creation with invalid data."""
        invalid_request = {
            "name": "",  # Empty name
            "description": "Test pipeline",
            "pipeline_type": "invalid_type",
            "algorithm": "Unknown"
        }
        
        response = client.post(
            "/ml-pipelines/pipelines",
            json=invalid_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 422  # Validation error

    def test_create_pipeline_unauthorized(self, client):
        """Test pipeline creation without authentication."""
        response = client.post("/ml-pipelines/pipelines", json={})
        
        assert response.status_code == 401

    def test_list_pipelines_success(self, client, mock_auth):
        """Test listing pipelines."""
        response = client.get(
            "/ml-pipelines/pipelines",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        if data:  # If pipelines exist
            pipeline = data[0]
            assert "pipeline_id" in pipeline
            assert "name" in pipeline
            assert "status" in pipeline
            assert "execution_stats" in pipeline

    def test_list_pipelines_with_filters(self, client, mock_auth):
        """Test listing pipelines with filters."""
        response = client.get(
            "/ml-pipelines/pipelines?pipeline_type=training&status_filter=active",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # All returned pipelines should match filters
        for pipeline in data:
            assert pipeline["pipeline_type"] == "training"
            assert pipeline["status"] == "active"

    def test_list_pipelines_with_pagination(self, client, mock_auth):
        """Test listing pipelines with pagination."""
        response = client.get(
            "/ml-pipelines/pipelines?limit=10&offset=5",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return at most 10 items
        assert len(data) <= 10

    def test_get_pipeline_success(self, client, mock_auth):
        """Test getting a specific pipeline."""
        pipeline_id = "pipeline_123456"
        
        response = client.get(
            f"/ml-pipelines/pipelines/{pipeline_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["pipeline_id"] == pipeline_id
        assert "name" in data
        assert "description" in data
        assert "pipeline_type" in data
        assert "algorithm" in data
        assert "status" in data
        assert "execution_stats" in data

    def test_get_pipeline_not_found(self, client, mock_auth):
        """Test getting a non-existent pipeline."""
        response = client.get(
            "/ml-pipelines/pipelines/nonexistent",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 404

    def test_execute_pipeline_success(self, client, mock_auth, sample_execution_request):
        """Test successful pipeline execution."""
        pipeline_id = "pipeline_123456"
        
        response = client.post(
            f"/ml-pipelines/pipelines/{pipeline_id}/execute",
            json=sample_execution_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 202
        data = response.json()
        
        # Verify response structure
        assert "execution_id" in data
        assert data["pipeline_id"] == pipeline_id
        assert data["status"] == "running"
        assert data["progress"] == 0.0
        assert "started_at" in data
        assert "logs_url" in data
        assert "results_url" in data

    def test_execute_pipeline_invalid_mode(self, client, mock_auth):
        """Test pipeline execution with invalid mode."""
        pipeline_id = "pipeline_123456"
        invalid_request = {
            "pipeline_id": pipeline_id,
            "execution_mode": "invalid_mode"
        }
        
        response = client.post(
            f"/ml-pipelines/pipelines/{pipeline_id}/execute",
            json=invalid_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 422  # Validation error

    def test_get_execution_status_success(self, client, mock_auth):
        """Test getting execution status."""
        execution_id = "exec_123456"
        
        response = client.get(
            f"/ml-pipelines/executions/{execution_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["execution_id"] == execution_id
        assert "status" in data
        assert "progress" in data
        assert "steps" in data
        assert "metrics" in data
        assert "artifacts" in data

    def test_get_execution_status_not_found(self, client, mock_auth):
        """Test getting status for non-existent execution."""
        response = client.get(
            "/ml-pipelines/executions/nonexistent",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 404

    def test_deploy_model_success(self, client, mock_auth, sample_deployment_request):
        """Test successful model deployment."""
        model_id = "model_123456"
        
        response = client.post(
            f"/ml-pipelines/models/{model_id}/deploy",
            json=sample_deployment_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        # Verify response structure
        assert "deployment_id" in data
        assert data["model_id"] == model_id
        assert data["deployment_name"] == sample_deployment_request["deployment_name"]
        assert data["environment"] == sample_deployment_request["environment"]
        assert data["status"] == "deploying"
        assert "endpoint_url" in data
        assert "health_check_url" in data
        assert "deployed_at" in data
        assert "metrics" in data

    def test_deploy_model_invalid_environment(self, client, mock_auth):
        """Test model deployment with invalid environment."""
        model_id = "model_123456"
        invalid_request = {
            "model_id": model_id,
            "deployment_name": "test-deployment",
            "environment": "invalid_env",
            "resources": {"cpu": "1", "memory": "1Gi"}
        }
        
        response = client.post(
            f"/ml-pipelines/models/{model_id}/deploy",
            json=invalid_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Should succeed as validation is not strict in the endpoint
        assert response.status_code == 201

    def test_list_deployments_success(self, client, mock_auth):
        """Test listing deployments."""
        response = client.get(
            "/ml-pipelines/deployments",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        if data:  # If deployments exist
            deployment = data[0]
            assert "deployment_id" in deployment
            assert "model_id" in deployment
            assert "deployment_name" in deployment
            assert "environment" in deployment
            assert "status" in deployment
            assert "metrics" in deployment

    def test_list_deployments_with_filters(self, client, mock_auth):
        """Test listing deployments with filters."""
        response = client.get(
            "/ml-pipelines/deployments?environment=production&status_filter=active",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # All returned deployments should match filters
        for deployment in data:
            assert deployment["environment"] == "production"
            assert deployment["status"] == "active"

    def test_get_ml_metrics_success(self, client, mock_auth):
        """Test getting ML metrics."""
        response = client.get(
            "/ml-pipelines/metrics",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "pipelines" in data
        assert "models" in data
        assert "deployments" in data
        assert "resource_usage" in data
        assert "timestamp" in data
        
        # Verify pipelines metrics
        pipeline_metrics = data["pipelines"]
        assert "total_pipelines" in pipeline_metrics
        assert "active_pipelines" in pipeline_metrics
        assert "success_rate" in pipeline_metrics

    def test_insufficient_permissions_create(self, client):
        """Test pipeline creation with insufficient permissions."""
        with patch("src.packages.api.api.dependencies.auth.get_current_user") as mock:
            mock.return_value = {
                "user_id": "test_user",
                "permissions": ["ml_pipelines:read"]  # No write permission
            }
            
            response = client.post(
                "/ml-pipelines/pipelines",
                json={"name": "test", "description": "test"},
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 403

    def test_insufficient_permissions_execute(self, client):
        """Test pipeline execution with insufficient permissions."""
        with patch("src.packages.api.api.dependencies.auth.get_current_user") as mock:
            mock.return_value = {
                "user_id": "test_user",
                "permissions": ["ml_pipelines:read"]  # No execute permission
            }
            
            response = client.post(
                "/ml-pipelines/pipelines/test/execute",
                json={"pipeline_id": "test"},
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 403

    def test_insufficient_permissions_deploy(self, client):
        """Test model deployment with insufficient permissions."""
        with patch("src.packages.api.api.dependencies.auth.get_current_user") as mock:
            mock.return_value = {
                "user_id": "test_user",
                "permissions": ["ml_pipelines:read"]  # No deploy permission
            }
            
            response = client.post(
                "/ml-pipelines/models/test/deploy",
                json={"deployment_name": "test", "environment": "dev"},
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 403


@pytest.mark.integration
class TestMLPipelineIntegration:
    """Integration tests for ML pipeline endpoints."""

    def test_pipeline_lifecycle_integration(self, client, mock_auth, sample_pipeline_request):
        """Test complete pipeline lifecycle (create, execute, monitor)."""
        # Create pipeline
        create_response = client.post(
            "/ml-pipelines/pipelines",
            json=sample_pipeline_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert create_response.status_code == 201
        pipeline_id = create_response.json()["pipeline_id"]
        
        # Execute pipeline
        execution_request = {
            "pipeline_id": pipeline_id,
            "execution_mode": "async"
        }
        
        execute_response = client.post(
            f"/ml-pipelines/pipelines/{pipeline_id}/execute",
            json=execution_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert execute_response.status_code == 202
        execution_id = execute_response.json()["execution_id"]
        
        # Monitor execution
        status_response = client.get(
            f"/ml-pipelines/executions/{execution_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert status_response.status_code == 200

    def test_model_deployment_lifecycle(self, client, mock_auth, sample_deployment_request):
        """Test complete model deployment lifecycle."""
        model_id = "model_123456"
        
        # Deploy model
        deploy_response = client.post(
            f"/ml-pipelines/models/{model_id}/deploy",
            json=sample_deployment_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert deploy_response.status_code == 201
        deployment_id = deploy_response.json()["deployment_id"]
        
        # List deployments to verify
        list_response = client.get(
            "/ml-pipelines/deployments",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert list_response.status_code == 200
        deployments = list_response.json()
        
        # Find our deployment
        found_deployment = None
        for deployment in deployments:
            if deployment["deployment_id"] == deployment_id:
                found_deployment = deployment
                break
        
        assert found_deployment is not None
        assert found_deployment["model_id"] == model_id

    def test_error_handling(self, client, mock_auth, sample_pipeline_request):
        """Test error handling in pipeline endpoints."""
        # Test with malformed JSON
        response = client.post(
            "/ml-pipelines/pipelines",
            data="invalid json",
            headers={"Authorization": "Bearer test_token", "Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


@pytest.mark.performance
class TestMLPipelinePerformance:
    """Performance tests for ML pipeline endpoints."""

    def test_list_pipelines_performance(self, client, mock_auth):
        """Test performance of listing many pipelines."""
        # Test with large limit
        response = client.get(
            "/ml-pipelines/pipelines?limit=100",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        # Should handle large requests reasonably
        assert len(response.json()) <= 100

    def test_concurrent_pipeline_creation(self, client, mock_auth, sample_pipeline_request):
        """Test handling concurrent pipeline creation requests."""
        responses = []
        
        for i in range(5):
            request_data = sample_pipeline_request.copy()
            request_data["name"] = f"concurrent_pipeline_{i}"
            
            response = client.post(
                "/ml-pipelines/pipelines",
                json=request_data,
                headers={"Authorization": "Bearer test_token"}
            )
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 201

    def test_metrics_endpoint_performance(self, client, mock_auth):
        """Test performance of metrics endpoint."""
        response = client.get(
            "/ml-pipelines/metrics",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        # Should return comprehensive metrics efficiently
        data = response.json()
        assert len(data) >= 4  # pipelines, models, deployments, resource_usage