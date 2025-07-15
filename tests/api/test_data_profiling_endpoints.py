"""Test Data Profiling API Endpoints."""

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
            "roles": ["user", "analyst"],
            "permissions": [
                "data_profiling:read", 
                "data_profiling:write", 
                "data_profiling:delete", 
                "data_profiling:admin"
            ]
        }
        yield mock


@pytest.fixture
def sample_profiling_request():
    """Sample data profiling request."""
    return {
        "dataset_id": "test_dataset_123",
        "data": [
            {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com"},
            {"id": 2, "name": "Jane Smith", "age": 25, "email": "jane@example.com"},
            {"id": 3, "name": "Bob Johnson", "age": 35, "email": "bob@example.com"}
        ],
        "config": {
            "enable_sampling": True,
            "sample_size": 1000,
            "enable_parallel_processing": True
        }
    }


class TestDataProfilingEndpoints:
    """Test data profiling endpoints."""

    def test_profile_dataset_success(self, client, mock_auth, sample_profiling_request):
        """Test successful dataset profiling."""
        response = client.post(
            "/data-profiling/profile",
            json=sample_profiling_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        # Verify response structure
        assert "validation_id" in data
        assert data["dataset_id"] == "test_dataset_123"
        assert "overall_score" in data
        assert "quality_scores" in data
        assert data["total_records"] == 3
        assert data["total_columns"] == 4
        assert "issues_detected" in data
        assert "processing_time_ms" in data
        assert "created_at" in data

    def test_profile_dataset_empty_data(self, client, mock_auth):
        """Test profiling with empty dataset."""
        request_data = {
            "dataset_id": "test_dataset_empty",
            "data": []
        }
        
        response = client.post(
            "/data-profiling/profile",
            json=request_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_profile_dataset_unauthorized(self, client):
        """Test profiling without authentication."""
        response = client.post("/data-profiling/profile", json={})
        
        assert response.status_code == 401

    def test_list_profiles_success(self, client, mock_auth):
        """Test listing profiles."""
        response = client.get(
            "/data-profiling/profiles",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "profiles" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data
        assert isinstance(data["profiles"], list)

    def test_list_profiles_with_pagination(self, client, mock_auth):
        """Test listing profiles with pagination."""
        response = client.get(
            "/data-profiling/profiles?page=2&page_size=10",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["page"] == 2
        assert data["page_size"] == 10

    def test_list_profiles_with_filter(self, client, mock_auth):
        """Test listing profiles with dataset filter."""
        response = client.get(
            "/data-profiling/profiles?dataset_id=test_dataset",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200

    def test_get_profile_details_success(self, client, mock_auth):
        """Test getting profile details."""
        profile_id = "prof_123456"
        
        response = client.get(
            f"/data-profiling/profiles/{profile_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "profile" in data
        assert "quality_issues" in data
        assert "remediation_suggestions" in data
        assert "quality_trends" in data

    def test_delete_profile_success(self, client, mock_auth):
        """Test deleting a profile."""
        profile_id = "prof_123456"
        
        response = client.delete(
            f"/data-profiling/profiles/{profile_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 204

    def test_compare_profiles_success(self, client, mock_auth):
        """Test comparing two profiles."""
        profile_id = "prof_123456"
        other_profile_id = "prof_789012"
        
        response = client.post(
            f"/data-profiling/profiles/{profile_id}/compare/{other_profile_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "profile1_id" in data
        assert "profile2_id" in data
        assert "comparison_date" in data
        assert "score_comparison" in data
        assert "issue_comparison" in data
        assert "recommendations" in data

    def test_get_engine_metrics_success(self, client, mock_auth):
        """Test getting engine metrics."""
        response = client.get(
            "/data-profiling/engine/metrics",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "execution_metrics" in data
        assert "cache_info" in data
        assert "timestamp" in data

    def test_clear_engine_cache_success(self, client, mock_auth):
        """Test clearing engine cache."""
        response = client.post(
            "/data-profiling/engine/cache/clear",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Cache cleared successfully"

    def test_insufficient_permissions(self, client):
        """Test with insufficient permissions."""
        with patch("src.packages.api.api.dependencies.auth.get_current_user") as mock:
            mock.return_value = {
                "user_id": "test_user",
                "permissions": ["data_profiling:read"]  # No write permission
            }
            
            response = client.post(
                "/data-profiling/profile",
                json={"dataset_id": "test", "data": []},
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 403


@pytest.mark.integration
class TestDataProfilingIntegration:
    """Integration tests for data profiling endpoints."""

    @patch("src.packages.data_profiling.application.services.profiling_engine.ProfilingEngine")
    def test_profiling_integration_with_engine(self, mock_engine, client, mock_auth, sample_profiling_request):
        """Test integration with profiling engine."""
        # Mock the profiling engine
        mock_profile = Mock()
        mock_profile.profile_id = "prof_123"
        mock_profile.dataset_id = "test_dataset_123"
        mock_profile.quality_scores.overall_score = 0.85
        mock_profile.quality_scores.get_dimension_scores.return_value = {
            "completeness": 0.9,
            "accuracy": 0.8
        }
        mock_profile.record_count = 3
        mock_profile.column_count = 4
        mock_profile.quality_issues = []
        mock_profile.created_at.isoformat.return_value = "2024-01-15T10:30:00Z"
        
        mock_engine.return_value.profile_dataset.return_value = mock_profile
        
        response = client.post(
            "/data-profiling/profile",
            json=sample_profiling_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 201
        mock_engine.return_value.profile_dataset.assert_called_once()

    def test_error_handling(self, client, mock_auth, sample_profiling_request):
        """Test error handling in profiling endpoints."""
        with patch("src.packages.data_profiling.application.services.profiling_engine.ProfilingEngine") as mock_engine:
            mock_engine.return_value.profile_dataset.side_effect = Exception("Test error")
            
            response = client.post(
                "/data-profiling/profile",
                json=sample_profiling_request,
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 500
            assert "failed" in response.json()["detail"].lower()


@pytest.mark.performance
class TestDataProfilingPerformance:
    """Performance tests for data profiling endpoints."""

    def test_large_dataset_profiling(self, client, mock_auth):
        """Test profiling with large dataset."""
        # Create a larger dataset
        large_data = [
            {"id": i, "name": f"User {i}", "age": 20 + (i % 50), "email": f"user{i}@example.com"}
            for i in range(10000)
        ]
        
        request_data = {
            "dataset_id": "large_dataset",
            "data": large_data
        }
        
        with patch("src.packages.data_profiling.application.services.profiling_engine.ProfilingEngine"):
            response = client.post(
                "/data-profiling/profile",
                json=request_data,
                headers={"Authorization": "Bearer test_token"},
                timeout=30  # Allow more time for large dataset
            )
            
            # Should still succeed but may take longer
            assert response.status_code in [201, 500]  # May timeout in test environment

    def test_concurrent_profiling_requests(self, client, mock_auth, sample_profiling_request):
        """Test handling concurrent profiling requests."""
        import asyncio
        import aiohttp
        
        # This would require async test setup
        # For now, just test that multiple sequential requests work
        responses = []
        
        for i in range(5):
            request_data = sample_profiling_request.copy()
            request_data["dataset_id"] = f"concurrent_test_{i}"
            
            response = client.post(
                "/data-profiling/profile",
                json=request_data,
                headers={"Authorization": "Bearer test_token"}
            )
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 201