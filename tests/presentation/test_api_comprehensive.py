"""Comprehensive tests for API endpoints - Phase 3 Coverage."""

from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import io
import tempfile
from pathlib import Path

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app
from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore
from pynomaly.domain.exceptions import EntityNotFoundError, AuthenticationError


@pytest.fixture
def test_container():
    """Create test container with mocked dependencies."""
    container = create_container()
    return container


@pytest.fixture
def client(test_container):
    """Create test client with mocked container."""
    app = create_app(test_container)
    return TestClient(app)


@pytest.fixture
async def async_client(test_container):
    """Create async test client."""
    app = create_app(test_container)
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def auth_headers():
    """Create authentication headers for testing."""
    with patch('pynomaly.infrastructure.auth.get_current_active_user') as mock_auth:
        mock_user = Mock()
        mock_user.id = "test_user_id"
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.is_active = True
        mock_user.roles = ["user"]
        mock_auth.return_value = mock_user
        
        return {"Authorization": "Bearer test_token"}


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return "feature1,feature2,target\n1,2,0\n3,4,0\n5,6,1\n7,8,0\n9,10,1\n"


class TestHealthEndpoints:
    """Comprehensive tests for health check endpoints."""
    
    def test_health_check_basic(self, client: TestClient):
        """Test basic health endpoint."""
        response = client.get("/api/health/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data
    
    def test_health_check_with_dependencies(self, client: TestClient):
        """Test health check with dependency status."""
        with patch('pynomaly.infrastructure.monitoring.health.get_system_health') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "checks": {
                    "database": {"status": "healthy", "response_time": 0.05},
                    "cache": {"status": "healthy", "hit_rate": 0.95},
                    "memory": {"status": "healthy", "usage_percent": 45.2}
                }
            }
            
            response = client.get("/api/health/")
            
            assert response.status_code == 200
            data = response.json()
            assert data["checks"]["database"]["status"] == "healthy"
            assert data["checks"]["cache"]["hit_rate"] == 0.95
    
    def test_readiness_check(self, client: TestClient):
        """Test readiness endpoint."""
        response = client.get("/api/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert "checks" in data
        assert "timestamp" in data
    
    def test_liveness_check(self, client: TestClient):
        """Test liveness endpoint."""
        response = client.get("/api/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True
    
    def test_metrics_endpoint(self, client: TestClient):
        """Test metrics endpoint for monitoring."""
        response = client.get("/api/health/metrics")
        
        assert response.status_code == 200
        # Should return Prometheus-format metrics
        assert "text/plain" in response.headers.get("content-type", "")
    
    def test_health_check_failure_simulation(self, client: TestClient):
        """Test health check with simulated failures."""
        with patch('pynomaly.infrastructure.monitoring.health.get_system_health') as mock_health:
            mock_health.return_value = {
                "status": "unhealthy",
                "checks": {
                    "database": {"status": "unhealthy", "error": "Connection timeout"},
                    "cache": {"status": "degraded", "warning": "High latency"}
                }
            }
            
            response = client.get("/api/health/")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["checks"]["database"]["error"] == "Connection timeout"


class TestAuthenticationEndpoints:
    """Comprehensive tests for authentication endpoints."""
    
    def test_login_success(self, client: TestClient):
        """Test successful login."""
        with patch('pynomaly.infrastructure.auth.JWTAuthService') as mock_auth_service:
            mock_service = Mock()
            mock_user = Mock()
            mock_user.id = "user123"
            mock_user.username = "testuser"
            
            mock_service.authenticate_user.return_value = mock_user
            mock_service.create_access_token.return_value = {
                "access_token": "test_access_token",
                "refresh_token": "test_refresh_token",
                "token_type": "bearer"
            }
            
            with patch('pynomaly.presentation.api.endpoints.auth.get_auth', return_value=mock_service):
                response = client.post(
                    "/api/auth/login",
                    data={"username": "testuser", "password": "testpass"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["access_token"] == "test_access_token"
                assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client: TestClient):
        """Test login with invalid credentials."""
        with patch('pynomaly.infrastructure.auth.JWTAuthService') as mock_auth_service:
            mock_service = Mock()
            mock_service.authenticate_user.side_effect = AuthenticationError("Invalid credentials")
            
            with patch('pynomaly.presentation.api.endpoints.auth.get_auth', return_value=mock_service):
                response = client.post(
                    "/api/auth/login",
                    data={"username": "invalid", "password": "invalid"}
                )
                
                assert response.status_code == 401
                data = response.json()
                assert "Invalid credentials" in data["detail"]
    
    def test_user_registration(self, client: TestClient):
        """Test user registration."""
        with patch('pynomaly.infrastructure.auth.JWTAuthService') as mock_auth_service:
            mock_service = Mock()
            mock_user = Mock()
            mock_user.id = "new_user_123"
            mock_user.username = "newuser"
            mock_user.email = "new@example.com"
            mock_user.full_name = "New User"
            mock_user.is_active = True
            mock_user.roles = ["user"]
            mock_user.created_at.isoformat.return_value = "2024-01-01T00:00:00Z"
            
            mock_service.create_user.return_value = mock_user
            
            with patch('pynomaly.presentation.api.endpoints.auth.get_auth', return_value=mock_service):
                response = client.post(
                    "/api/auth/register",
                    json={
                        "username": "newuser",
                        "email": "new@example.com",
                        "password": "securepass123",
                        "full_name": "New User"
                    }
                )
                
                assert response.status_code == 201
                data = response.json()
                assert data["username"] == "newuser"
                assert data["email"] == "new@example.com"
    
    def test_refresh_token(self, client: TestClient):
        """Test token refresh."""
        with patch('pynomaly.infrastructure.auth.JWTAuthService') as mock_auth_service:
            mock_service = Mock()
            mock_service.refresh_access_token.return_value = {
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
                "token_type": "bearer"
            }
            
            with patch('pynomaly.presentation.api.endpoints.auth.get_auth', return_value=mock_service):
                response = client.post(
                    "/api/auth/refresh",
                    json={"refresh_token": "valid_refresh_token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["access_token"] == "new_access_token"
    
    def test_get_current_user_profile(self, client: TestClient, auth_headers):
        """Test getting current user profile."""
        response = client.get("/api/auth/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
    
    def test_create_api_key(self, client: TestClient, auth_headers):
        """Test API key creation."""
        with patch('pynomaly.infrastructure.auth.JWTAuthService') as mock_auth_service:
            mock_service = Mock()
            mock_service.create_api_key.return_value = "api_key_123456"
            
            with patch('pynomaly.presentation.api.endpoints.auth.get_auth', return_value=mock_service):
                response = client.post(
                    "/api/auth/api-keys",
                    json={
                        "name": "Test API Key",
                        "description": "For testing purposes"
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 201
                data = response.json()
                assert data["api_key"] == "api_key_123456"
                assert data["name"] == "Test API Key"
    
    def test_revoke_api_key(self, client: TestClient, auth_headers):
        """Test API key revocation."""
        with patch('pynomaly.infrastructure.auth.JWTAuthService') as mock_auth_service:
            mock_service = Mock()
            mock_service.revoke_api_key.return_value = True
            
            with patch('pynomaly.presentation.api.endpoints.auth.get_auth', return_value=mock_service):
                with patch('pynomaly.infrastructure.auth.get_current_active_user') as mock_user:
                    mock_user.return_value.api_keys = ["api_key_to_revoke"]
                    
                    response = client.delete(
                        "/api/auth/api-keys/api_key_to_revoke",
                        headers=auth_headers
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "revoked successfully" in data["message"]
    
    def test_logout(self, client: TestClient, auth_headers):
        """Test user logout."""
        response = client.post("/api/auth/logout", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "Logged out successfully" in data["message"]


class TestDetectorEndpoints:
    """Comprehensive tests for detector management endpoints."""
    
    def test_create_detector_basic(self, client: TestClient, auth_headers):
        """Test basic detector creation."""
        detector_data = {
            "name": "Test Detector",
            "algorithm": "IsolationForest",
            "description": "Test description",
            "parameters": {"contamination": 0.1}
        }
        
        response = client.post("/api/detectors/", json=detector_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Detector"
        assert data["algorithm"] == "IsolationForest"
        assert "id" in data
        assert not data["is_fitted"]
    
    def test_create_detector_with_advanced_parameters(self, client: TestClient, auth_headers):
        """Test detector creation with advanced parameters."""
        detector_data = {
            "name": "Advanced Detector",
            "algorithm": "LOF",
            "description": "Local Outlier Factor detector",
            "parameters": {
                "n_neighbors": 20,
                "contamination": 0.05,
                "leaf_size": 30,
                "metric": "minkowski"
            },
            "tags": ["outlier", "unsupervised"],
            "metadata": {
                "created_by": "test_user",
                "purpose": "fraud_detection"
            }
        }
        
        response = client.post("/api/detectors/", json=detector_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["algorithm"] == "LOF"
        assert data["parameters"]["n_neighbors"] == 20
        assert "outlier" in data["tags"]
    
    def test_list_detectors_with_pagination(self, client: TestClient, auth_headers):
        """Test listing detectors with pagination."""
        # Create multiple detectors first
        for i in range(5):
            client.post("/api/detectors/", json={
                "name": f"Detector {i}",
                "algorithm": "IsolationForest"
            }, headers=auth_headers)
        
        # Test pagination
        response = client.get("/api/detectors/?skip=1&limit=2", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 2
        assert data["total"] >= 5
        assert data["skip"] == 1
        assert data["limit"] == 2
    
    def test_list_detectors_with_filters(self, client: TestClient, auth_headers):
        """Test listing detectors with filters."""
        # Create detectors with different algorithms
        algorithms = ["IsolationForest", "LOF", "OCSVM"]
        for algo in algorithms:
            client.post("/api/detectors/", json={
                "name": f"{algo} Detector",
                "algorithm": algo
            }, headers=auth_headers)
        
        # Filter by algorithm
        response = client.get("/api/detectors/?algorithm=LOF", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert all(detector["algorithm"] == "LOF" for detector in data["items"])
    
    def test_get_detector_details(self, client: TestClient, auth_headers):
        """Test getting detector details."""
        # Create detector
        create_response = client.post("/api/detectors/", json={
            "name": "Detail Test",
            "algorithm": "IsolationForest",
            "parameters": {"contamination": 0.15}
        }, headers=auth_headers)
        detector_id = create_response.json()["id"]
        
        # Get detector details
        response = client.get(f"/api/detectors/{detector_id}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == detector_id
        assert data["name"] == "Detail Test"
        assert data["parameters"]["contamination"] == 0.15
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_update_detector(self, client: TestClient, auth_headers):
        """Test updating detector."""
        # Create detector
        create_response = client.post("/api/detectors/", json={
            "name": "Update Test",
            "algorithm": "IsolationForest"
        }, headers=auth_headers)
        detector_id = create_response.json()["id"]
        
        # Update detector
        update_data = {
            "name": "Updated Name",
            "description": "Updated description",
            "parameters": {"contamination": 0.2, "n_estimators": 200}
        }
        response = client.put(f"/api/detectors/{detector_id}", json=update_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Updated description"
        assert data["parameters"]["contamination"] == 0.2
    
    def test_detector_validation_endpoints(self, client: TestClient, auth_headers):
        """Test detector validation."""
        # Test with invalid algorithm
        response = client.post("/api/detectors/", json={
            "name": "Invalid Detector",
            "algorithm": "NonExistentAlgorithm"
        }, headers=auth_headers)
        
        assert response.status_code == 400
        data = response.json()
        assert "algorithm" in data["detail"].lower()
        
        # Test parameter validation
        response = client.post("/api/detectors/", json={
            "name": "Invalid Parameters",
            "algorithm": "IsolationForest",
            "parameters": {"contamination": 1.5}  # Invalid contamination rate
        }, headers=auth_headers)
        
        assert response.status_code == 400
    
    def test_delete_detector(self, client: TestClient, auth_headers):
        """Test deleting detector."""
        # Create detector
        create_response = client.post("/api/detectors/", json={
            "name": "Delete Test",
            "algorithm": "LOF"
        }, headers=auth_headers)
        detector_id = create_response.json()["id"]
        
        # Delete detector
        response = client.delete(f"/api/detectors/{detector_id}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify deletion
        get_response = client.get(f"/api/detectors/{detector_id}", headers=auth_headers)
        assert get_response.status_code == 404
    
    def test_detector_clone(self, client: TestClient, auth_headers):
        """Test cloning detector."""
        # Create original detector
        create_response = client.post("/api/detectors/", json={
            "name": "Original Detector",
            "algorithm": "IsolationForest",
            "parameters": {"contamination": 0.1}
        }, headers=auth_headers)
        detector_id = create_response.json()["id"]
        
        # Clone detector
        response = client.post(f"/api/detectors/{detector_id}/clone", json={
            "name": "Cloned Detector"
        }, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Cloned Detector"
        assert data["algorithm"] == "IsolationForest"
        assert data["id"] != detector_id  # Different ID
    
    def test_detector_export_import(self, client: TestClient, auth_headers):
        """Test detector export and import."""
        # Create detector
        create_response = client.post("/api/detectors/", json={
            "name": "Export Test",
            "algorithm": "LOF",
            "parameters": {"n_neighbors": 25}
        }, headers=auth_headers)
        detector_id = create_response.json()["id"]
        
        # Export detector
        export_response = client.get(f"/api/detectors/{detector_id}/export", headers=auth_headers)
        
        assert export_response.status_code == 200
        assert "application/json" in export_response.headers.get("content-type", "")
        
        # Import detector
        import_data = export_response.json()
        import_data["name"] = "Imported Detector"
        
        import_response = client.post("/api/detectors/import", json=import_data, headers=auth_headers)
        
        assert import_response.status_code == 200
        data = import_response.json()
        assert data["name"] == "Imported Detector"
        assert data["algorithm"] == "LOF"


class TestDatasetEndpoints:
    """Comprehensive tests for dataset management endpoints."""
    
    def test_upload_csv_dataset(self, client: TestClient, auth_headers, sample_csv_data):
        """Test uploading CSV dataset."""
        csv_file = io.BytesIO(sample_csv_data.encode())
        
        files = {"file": ("test.csv", csv_file, "text/csv")}
        data = {
            "name": "Test CSV Dataset",
            "target_column": "target",
            "description": "Test dataset for validation"
        }
        
        response = client.post("/api/datasets/upload", files=files, data=data, headers=auth_headers)
        
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "Test CSV Dataset"
        assert result["n_samples"] == 5
        assert result["n_features"] == 2
        assert result["has_target"] is True
        assert "id" in result
    
    def test_upload_parquet_dataset(self, client: TestClient, auth_headers):
        """Test uploading Parquet dataset."""
        import pandas as pd
        import tempfile
        
        # Create sample DataFrame and save as Parquet
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1],
            "label": [0, 0, 1, 0, 1]
        })
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name)
            
            with open(tmp.name, "rb") as f:
                files = {"file": ("test.parquet", f, "application/octet-stream")}
                data = {
                    "name": "Parquet Dataset",
                    "target_column": "label"
                }
                
                response = client.post("/api/datasets/upload", files=files, data=data, headers=auth_headers)
                
                assert response.status_code == 200
                result = response.json()
                assert result["name"] == "Parquet Dataset"
                assert result["n_samples"] == 5
    
    def test_list_datasets_with_metadata(self, client: TestClient, auth_headers, sample_csv_data):
        """Test listing datasets with metadata."""
        # Upload multiple datasets
        for i in range(3):
            csv_file = io.BytesIO(sample_csv_data.encode())
            files = {"file": (f"test_{i}.csv", csv_file, "text/csv")}
            data = {"name": f"Dataset {i}"}
            
            client.post("/api/datasets/upload", files=files, data=data, headers=auth_headers)
        
        # List datasets
        response = client.get("/api/datasets/", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) >= 3
        
        # Check metadata
        dataset = data["items"][0]
        assert "name" in dataset
        assert "n_samples" in dataset
        assert "n_features" in dataset
        assert "file_size" in dataset
        assert "created_at" in dataset
    
    def test_dataset_sample_with_options(self, client: TestClient, auth_headers, sample_csv_data):
        """Test getting dataset sample with various options."""
        # Upload dataset
        csv_file = io.BytesIO(sample_csv_data.encode())
        upload_response = client.post(
            "/api/datasets/upload",
            files={"file": ("sample.csv", csv_file, "text/csv")},
            data={"name": "Sample Dataset"},
            headers=auth_headers
        )
        dataset_id = upload_response.json()["id"]
        
        # Test different sample options
        test_cases = [
            ("?n=3", 3),
            ("?n=10", 5),  # More than available, should return all
            ("?random=true&n=2", 2),
            ("?columns=feature1,feature2", None)
        ]
        
        for query, expected_size in test_cases:
            response = client.get(f"/api/datasets/{dataset_id}/sample{query}", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            if expected_size:
                assert data["sample_size"] == expected_size
    
    def test_dataset_statistics(self, client: TestClient, auth_headers, sample_csv_data):
        """Test dataset statistics endpoint."""
        # Upload dataset
        csv_file = io.BytesIO(sample_csv_data.encode())
        upload_response = client.post(
            "/api/datasets/upload",
            files={"file": ("stats.csv", csv_file, "text/csv")},
            data={"name": "Stats Dataset"},
            headers=auth_headers
        )
        dataset_id = upload_response.json()["id"]
        
        # Get statistics
        response = client.get(f"/api/datasets/{dataset_id}/statistics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "basic_stats" in data
        assert "correlation_matrix" in data
        assert "feature_types" in data
        assert "missing_values" in data
        assert "outlier_detection" in data
        
        # Check basic stats structure
        basic_stats = data["basic_stats"]
        assert "feature1" in basic_stats
        assert "mean" in basic_stats["feature1"]
        assert "std" in basic_stats["feature1"]
    
    def test_dataset_quality_analysis(self, client: TestClient, auth_headers):
        """Test comprehensive dataset quality analysis."""
        # Create dataset with quality issues
        problematic_data = "x,y,z\n1,2,3\n4,,6\n7,8,9\n1,2,3\n,5,6\n10,20,30\n"
        csv_file = io.BytesIO(problematic_data.encode())
        
        upload_response = client.post(
            "/api/datasets/upload",
            files={"file": ("quality.csv", csv_file, "text/csv")},
            data={"name": "Quality Test"},
            headers=auth_headers
        )
        dataset_id = upload_response.json()["id"]
        
        # Analyze quality
        response = client.get(f"/api/datasets/{dataset_id}/quality", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check quality metrics
        assert data["missing_values"]["y"] > 0  # Missing values in y column
        assert data["missing_values"]["x"] > 0  # Missing values in x column
        assert data["duplicate_rows"] > 0       # Duplicate rows
        assert len(data["suggestions"]) > 0     # Should have suggestions
        assert "data_quality_score" in data     # Overall quality score
    
    def test_dataset_preprocessing_options(self, client: TestClient, auth_headers, sample_csv_data):
        """Test dataset preprocessing options."""
        # Upload dataset
        csv_file = io.BytesIO(sample_csv_data.encode())
        upload_response = client.post(
            "/api/datasets/upload",
            files={"file": ("preprocess.csv", csv_file, "text/csv")},
            data={"name": "Preprocess Dataset"},
            headers=auth_headers
        )
        dataset_id = upload_response.json()["id"]
        
        # Test preprocessing
        preprocessing_config = {
            "remove_duplicates": True,
            "handle_missing": "mean",
            "scale_features": "standard",
            "remove_outliers": True,
            "outlier_method": "iqr"
        }
        
        response = client.post(
            f"/api/datasets/{dataset_id}/preprocess",
            json=preprocessing_config,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "preprocessing_summary" in data
        assert "transformed_dataset_id" in data
    
    def test_dataset_export_formats(self, client: TestClient, auth_headers, sample_csv_data):
        """Test dataset export in different formats."""
        # Upload dataset
        csv_file = io.BytesIO(sample_csv_data.encode())
        upload_response = client.post(
            "/api/datasets/upload",
            files={"file": ("export.csv", csv_file, "text/csv")},
            data={"name": "Export Dataset"},
            headers=auth_headers
        )
        dataset_id = upload_response.json()["id"]
        
        # Test different export formats
        formats = ["csv", "json", "parquet", "excel"]
        
        for format_type in formats:
            response = client.get(
                f"/api/datasets/{dataset_id}/export?format={format_type}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            
            # Check content type
            if format_type == "csv":
                assert "text/csv" in response.headers.get("content-type", "")
            elif format_type == "json":
                assert "application/json" in response.headers.get("content-type", "")
            elif format_type in ["parquet", "excel"]:
                assert "application/octet-stream" in response.headers.get("content-type", "")
    
    def test_dataset_delete_with_dependencies(self, client: TestClient, auth_headers, sample_csv_data):
        """Test dataset deletion with dependency checking."""
        # Upload dataset
        csv_file = io.BytesIO(sample_csv_data.encode())
        upload_response = client.post(
            "/api/datasets/upload",
            files={"file": ("delete.csv", csv_file, "text/csv")},
            data={"name": "Delete Dataset"},
            headers=auth_headers
        )
        dataset_id = upload_response.json()["id"]
        
        # Create detector and use dataset (simulate dependency)
        detector_response = client.post("/api/detectors/", json={
            "name": "Dependent Detector",
            "algorithm": "IsolationForest"
        }, headers=auth_headers)
        
        # Try to delete dataset (should check dependencies)
        response = client.delete(f"/api/datasets/{dataset_id}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify deletion
        get_response = client.get(f"/api/datasets/{dataset_id}", headers=auth_headers)
        assert get_response.status_code == 404


class TestDetectionEndpoints:
    """Comprehensive tests for detection endpoints."""
    
    @pytest.mark.asyncio
    async def test_detector_training_workflow(self, async_client: AsyncClient, auth_headers, sample_csv_data):
        """Test complete detector training workflow."""
        # Create detector
        detector_response = await async_client.post("/api/detectors/", json={
            "name": "Training Test",
            "algorithm": "IsolationForest",
            "parameters": {"contamination": 0.1}
        }, headers=auth_headers)
        detector_id = detector_response.json()["id"]
        
        # Upload dataset
        csv_file = io.BytesIO(sample_csv_data.encode())
        dataset_response = await async_client.post(
            "/api/datasets/upload",
            files={"file": ("train.csv", csv_file, "text/csv")},
            data={"name": "Training Data"},
            headers=auth_headers
        )
        dataset_id = dataset_response.json()["id"]
        
        # Train detector
        train_response = await async_client.post("/api/detection/train", json={
            "detector_id": detector_id,
            "dataset_id": dataset_id,
            "validation_split": 0.2,
            "save_model": True,
            "training_options": {
                "cross_validation": True,
                "cv_folds": 5
            }
        }, headers=auth_headers)
        
        assert train_response.status_code == 200
        data = train_response.json()
        assert data["success"] is True
        assert data["training_time_ms"] > 0
        assert "model_metrics" in data
        assert "validation_results" in data
    
    @pytest.mark.asyncio
    async def test_anomaly_prediction(self, async_client: AsyncClient, auth_headers, sample_csv_data):
        """Test anomaly prediction on new data."""
        # Create and train detector (simplified)
        detector_response = await async_client.post("/api/detectors/", json={
            "name": "Prediction Test",
            "algorithm": "IsolationForest"
        }, headers=auth_headers)
        detector_id = detector_response.json()["id"]
        
        # Upload training dataset
        csv_file = io.BytesIO(sample_csv_data.encode())
        train_dataset_response = await async_client.post(
            "/api/datasets/upload",
            files={"file": ("train.csv", csv_file, "text/csv")},
            data={"name": "Training Data"},
            headers=auth_headers
        )
        train_dataset_id = train_dataset_response.json()["id"]
        
        # Train detector
        await async_client.post("/api/detection/train", json={
            "detector_id": detector_id,
            "dataset_id": train_dataset_id
        }, headers=auth_headers)
        
        # Upload test dataset
        test_data = "feature1,feature2\n100,200\n1,2\n999,888\n3,4\n"
        test_csv_file = io.BytesIO(test_data.encode())
        test_dataset_response = await async_client.post(
            "/api/datasets/upload",
            files={"file": ("test.csv", test_csv_file, "text/csv")},
            data={"name": "Test Data"},
            headers=auth_headers
        )
        test_dataset_id = test_dataset_response.json()["id"]
        
        # Predict anomalies
        prediction_response = await async_client.post("/api/detection/predict", json={
            "detector_id": detector_id,
            "dataset_id": test_dataset_id,
            "include_scores": True,
            "include_explanations": True
        }, headers=auth_headers)
        
        assert prediction_response.status_code == 200
        data = prediction_response.json()
        assert "predictions" in data
        assert "anomaly_scores" in data
        assert "summary" in data
        assert len(data["predictions"]) == 4  # 4 test samples
    
    @pytest.mark.asyncio
    async def test_batch_detection(self, async_client: AsyncClient, auth_headers):
        """Test batch detection across multiple datasets."""
        # Create detector
        detector_response = await async_client.post("/api/detectors/", json={
            "name": "Batch Test",
            "algorithm": "LOF"
        }, headers=auth_headers)
        detector_id = detector_response.json()["id"]
        
        # Upload multiple datasets
        dataset_ids = []
        for i in range(3):
            test_data = f"x,y\n{i+1},{i*2+1}\n{i+2},{i*2+2}\n{i+100},{i*200}\n"
            csv_file = io.BytesIO(test_data.encode())
            
            dataset_response = await async_client.post(
                "/api/datasets/upload",
                files={"file": (f"batch_{i}.csv", csv_file, "text/csv")},
                data={"name": f"Batch Dataset {i}"},
                headers=auth_headers
            )
            dataset_ids.append(dataset_response.json()["id"])
        
        # Train on first dataset
        await async_client.post("/api/detection/train", json={
            "detector_id": detector_id,
            "dataset_id": dataset_ids[0]
        }, headers=auth_headers)
        
        # Batch prediction
        batch_response = await async_client.post("/api/detection/batch", json={
            "detector_id": detector_id,
            "dataset_ids": dataset_ids[1:],
            "batch_options": {
                "parallel_processing": True,
                "max_workers": 2
            }
        }, headers=auth_headers)
        
        assert batch_response.status_code == 200
        data = batch_response.json()
        assert "batch_id" in data
        assert "results" in data
        assert len(data["results"]) == 2
    
    @pytest.mark.asyncio
    async def test_ensemble_detection(self, async_client: AsyncClient, auth_headers, sample_csv_data):
        """Test ensemble detection with multiple algorithms."""
        # Create multiple detectors
        algorithms = ["IsolationForest", "LOF", "OCSVM"]
        detector_ids = []
        
        for algo in algorithms:
            detector_response = await async_client.post("/api/detectors/", json={
                "name": f"Ensemble {algo}",
                "algorithm": algo
            }, headers=auth_headers)
            detector_ids.append(detector_response.json()["id"])
        
        # Upload dataset
        csv_file = io.BytesIO(sample_csv_data.encode())
        dataset_response = await async_client.post(
            "/api/datasets/upload",
            files={"file": ("ensemble.csv", csv_file, "text/csv")},
            data={"name": "Ensemble Data"},
            headers=auth_headers
        )
        dataset_id = dataset_response.json()["id"]
        
        # Train all detectors
        for detector_id in detector_ids:
            await async_client.post("/api/detection/train", json={
                "detector_id": detector_id,
                "dataset_id": dataset_id
            }, headers=auth_headers)
        
        # Ensemble prediction
        ensemble_response = await async_client.post("/api/detection/ensemble", json={
            "detector_ids": detector_ids,
            "dataset_id": dataset_id,
            "ensemble_method": "voting",
            "voting_strategy": "majority",
            "weights": [0.4, 0.3, 0.3]
        }, headers=auth_headers)
        
        assert ensemble_response.status_code == 200
        data = ensemble_response.json()
        assert "ensemble_predictions" in data
        assert "individual_predictions" in data
        assert "consensus_score" in data
        assert len(data["individual_predictions"]) == 3
    
    @pytest.mark.asyncio
    async def test_detection_results_management(self, async_client: AsyncClient, auth_headers):
        """Test detection results storage and retrieval."""
        # Get recent results
        response = await async_client.get("/api/detection/results", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "pagination" in data
        
        # Test filtering by detector
        response = await async_client.get("/api/detection/results?detector_id=test_id", headers=auth_headers)
        assert response.status_code == 200
        
        # Test filtering by date range
        response = await async_client.get(
            "/api/detection/results?start_date=2024-01-01&end_date=2024-12-31",
            headers=auth_headers
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_detection_performance_metrics(self, async_client: AsyncClient, auth_headers):
        """Test detection performance monitoring."""
        response = await async_client.get("/api/detection/metrics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "performance_summary" in data
        assert "algorithm_performance" in data
        assert "recent_activity" in data
        
        # Test detailed metrics for specific detector
        response = await async_client.get("/api/detection/metrics/detector_123", headers=auth_headers)
        # Would return 404 for non-existent detector, which is expected
        assert response.status_code in [200, 404]


class TestExperimentEndpoints:
    """Comprehensive tests for experiment management endpoints."""
    
    def test_create_experiment(self, client: TestClient, auth_headers):
        """Test experiment creation."""
        experiment_data = {
            "name": "Fraud Detection Experiment",
            "description": "Testing various algorithms for fraud detection",
            "tags": ["fraud", "ensemble", "comparison"],
            "metadata": {
                "dataset_type": "financial",
                "evaluation_metric": "precision_recall"
            }
        }
        
        response = client.post("/api/experiments/", json=experiment_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Fraud Detection Experiment"
        assert "fraud" in data["tags"]
        assert "id" in data
        assert data["status"] == "created"
    
    def test_experiment_workflow_management(self, client: TestClient, auth_headers):
        """Test experiment workflow execution."""
        # Create experiment
        create_response = client.post("/api/experiments/", json={
            "name": "Workflow Test",
            "description": "Testing experiment workflow"
        }, headers=auth_headers)
        experiment_id = create_response.json()["id"]
        
        # Add detectors to experiment
        detector_config = {
            "detectors": [
                {"algorithm": "IsolationForest", "parameters": {"contamination": 0.1}},
                {"algorithm": "LOF", "parameters": {"n_neighbors": 20}},
                {"algorithm": "OCSVM", "parameters": {"nu": 0.1}}
            ]
        }
        
        response = client.post(
            f"/api/experiments/{experiment_id}/detectors",
            json=detector_config,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["detectors"]) == 3
    
    def test_experiment_execution(self, client: TestClient, auth_headers, sample_csv_data):
        """Test experiment execution."""
        # Create experiment with detectors
        create_response = client.post("/api/experiments/", json={
            "name": "Execution Test",
            "detectors": [
                {"algorithm": "IsolationForest"},
                {"algorithm": "LOF"}
            ]
        }, headers=auth_headers)
        experiment_id = create_response.json()["id"]
        
        # Upload dataset
        csv_file = io.BytesIO(sample_csv_data.encode())
        dataset_response = client.post(
            "/api/datasets/upload",
            files={"file": ("experiment.csv", csv_file, "text/csv")},
            data={"name": "Experiment Data"},
            headers=auth_headers
        )
        dataset_id = dataset_response.json()["id"]
        
        # Execute experiment
        execution_config = {
            "dataset_id": dataset_id,
            "evaluation_metrics": ["precision", "recall", "f1_score", "roc_auc"],
            "cross_validation": {"enabled": True, "folds": 5},
            "test_split": 0.2
        }
        
        response = client.post(
            f"/api/experiments/{experiment_id}/execute",
            json=execution_config,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "execution_id" in data
    
    def test_experiment_results_analysis(self, client: TestClient, auth_headers):
        """Test experiment results and analysis."""
        # Create completed experiment (mocked)
        create_response = client.post("/api/experiments/", json={
            "name": "Results Test",
            "status": "completed"
        }, headers=auth_headers)
        experiment_id = create_response.json()["id"]
        
        # Get experiment results
        response = client.get(f"/api/experiments/{experiment_id}/results", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "detector_results" in data
        assert "comparative_analysis" in data
        
        # Get leaderboard
        response = client.get(f"/api/experiments/{experiment_id}/leaderboard", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "rankings" in data
        assert "metrics" in data
    
    def test_experiment_visualization_data(self, client: TestClient, auth_headers):
        """Test experiment visualization data endpoints."""
        create_response = client.post("/api/experiments/", json={
            "name": "Viz Test"
        }, headers=auth_headers)
        experiment_id = create_response.json()["id"]
        
        # Get ROC curves data
        response = client.get(f"/api/experiments/{experiment_id}/visualizations/roc", headers=auth_headers)
        assert response.status_code == 200
        
        # Get precision-recall curves
        response = client.get(f"/api/experiments/{experiment_id}/visualizations/pr", headers=auth_headers)
        assert response.status_code == 200
        
        # Get confusion matrices
        response = client.get(f"/api/experiments/{experiment_id}/visualizations/confusion", headers=auth_headers)
        assert response.status_code == 200


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    def test_authentication_required(self, client: TestClient):
        """Test endpoints require authentication."""
        protected_endpoints = [
            "/api/detectors/",
            "/api/datasets/",
            "/api/detection/train",
            "/api/experiments/"
        ]
        
        for endpoint in protected_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401
    
    def test_invalid_json_handling(self, client: TestClient, auth_headers):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/detectors/",
            data="invalid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_rate_limiting(self, client: TestClient, auth_headers):
        """Test rate limiting functionality."""
        # Make multiple rapid requests
        for _ in range(100):  # Assuming rate limit is lower
            response = client.get("/api/health/", headers=auth_headers)
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        assert response.status_code in [200, 429]
        
        if response.status_code == 429:
            assert "rate limit" in response.json()["detail"].lower()
    
    def test_large_file_upload_handling(self, client: TestClient, auth_headers):
        """Test handling of large file uploads."""
        # Create large CSV content (simulated)
        large_data = "x,y\n" + "\n".join(f"{i},{i*2}" for i in range(10000))
        large_csv = io.BytesIO(large_data.encode())
        
        files = {"file": ("large.csv", large_csv, "text/csv")}
        data = {"name": "Large Dataset"}
        
        response = client.post("/api/datasets/upload", files=files, data=data, headers=auth_headers)
        
        # Should either succeed or return appropriate error
        assert response.status_code in [200, 413, 422]
    
    def test_concurrent_request_handling(self, client: TestClient, auth_headers):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get("/api/health/", headers=auth_headers)
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(result.status_code == 200 for result in results)
    
    def test_database_connection_failure_handling(self, client: TestClient, auth_headers):
        """Test handling of database connection failures."""
        with patch('pynomaly.infrastructure.persistence.database.get_session') as mock_session:
            mock_session.side_effect = Exception("Database connection failed")
            
            response = client.get("/api/detectors/", headers=auth_headers)
            
            # Should return service unavailable
            assert response.status_code == 503
            data = response.json()
            assert "database" in data["detail"].lower()


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    def test_response_time_limits(self, client: TestClient, auth_headers):
        """Test API response time limits."""
        import time
        
        start_time = time.time()
        response = client.get("/api/health/", headers=auth_headers)
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_pagination_performance(self, client: TestClient, auth_headers):
        """Test pagination performance with large datasets."""
        # Create multiple items
        for i in range(50):
            client.post("/api/detectors/", json={
                "name": f"Perf Test {i}",
                "algorithm": "IsolationForest"
            }, headers=auth_headers)
        
        # Test large page sizes
        response = client.get("/api/detectors/?limit=100", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 100
    
    def test_memory_usage_with_large_datasets(self, client: TestClient, auth_headers):
        """Test memory usage with large dataset operations."""
        # Create reasonably large dataset
        large_data = "x,y,z\n" + "\n".join(f"{i},{i*2},{i*3}" for i in range(1000))
        csv_file = io.BytesIO(large_data.encode())
        
        files = {"file": ("memory_test.csv", csv_file, "text/csv")}
        data = {"name": "Memory Test Dataset"}
        
        response = client.post("/api/datasets/upload", files=files, data=data, headers=auth_headers)
        
        assert response.status_code == 200
        result = response.json()
        assert result["n_samples"] == 1000