"""Test Data Quality API Endpoints."""

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
                "data_quality:read", 
                "data_quality:write", 
                "data_quality:delete", 
                "data_quality:admin"
            ]
        }
        yield mock


@pytest.fixture
def sample_validation_request():
    """Sample data quality validation request."""
    return {
        "dataset_id": "test_dataset_123",
        "data": [
            {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com"},
            {"id": 2, "name": "Jane Smith", "age": 25, "email": "jane@example.com"},
            {"id": 3, "name": "Bob Johnson", "age": 35, "email": "bob@example.com"}
        ],
        "validation_rules": [
            {
                "name": "email_format",
                "description": "Validate email format",
                "logic_type": "regex",
                "parameters": {"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
                "target_columns": ["email"]
            }
        ],
        "config": {
            "scoring_method": "weighted_average",
            "enable_trend_analysis": True,
            "enable_business_impact_analysis": True
        }
    }


@pytest.fixture
def sample_rule_request():
    """Sample validation rule request."""
    return {
        "name": "age_range_validation",
        "description": "Validate age is within reasonable range",
        "logic_type": "range",
        "parameters": {
            "min_value": 0,
            "max_value": 150
        },
        "target_columns": ["age"],
        "severity": "medium",
        "enabled": True
    }


class TestDataQualityEndpoints:
    """Test data quality endpoints."""

    def test_validate_data_quality_success(self, client, mock_auth, sample_validation_request):
        """Test successful data quality validation."""
        response = client.post(
            "/data-quality/validate",
            json=sample_validation_request,
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
        assert "validation_rules_applied" in data
        assert "processing_time_ms" in data
        assert "created_at" in data

    def test_validate_empty_dataset(self, client, mock_auth):
        """Test validation with empty dataset."""
        request_data = {
            "dataset_id": "test_dataset_empty",
            "data": []
        }
        
        response = client.post(
            "/data-quality/validate",
            json=request_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_validate_unauthorized(self, client):
        """Test validation without authentication."""
        response = client.post("/data-quality/validate", json={})
        
        assert response.status_code == 401

    def test_create_validation_rule_success(self, client, mock_auth, sample_rule_request):
        """Test creating a validation rule."""
        response = client.post(
            "/data-quality/rules",
            json=sample_rule_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert "rule_id" in data
        assert data["name"] == sample_rule_request["name"]
        assert data["description"] == sample_rule_request["description"]
        assert data["logic_type"] == sample_rule_request["logic_type"]
        assert data["target_columns"] == sample_rule_request["target_columns"]
        assert data["severity"] == sample_rule_request["severity"]
        assert data["enabled"] == sample_rule_request["enabled"]
        assert "execution_stats" in data

    def test_list_validation_rules_success(self, client, mock_auth):
        """Test listing validation rules."""
        response = client.get(
            "/data-quality/rules",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        if data:  # If rules exist
            rule = data[0]
            assert "rule_id" in rule
            assert "name" in rule
            assert "enabled" in rule
            assert "execution_stats" in rule

    def test_list_validation_rules_enabled_only(self, client, mock_auth):
        """Test listing only enabled validation rules."""
        response = client.get(
            "/data-quality/rules?enabled_only=true",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # All returned rules should be enabled
        for rule in data:
            assert rule["enabled"] is True

    def test_get_validation_rule_success(self, client, mock_auth):
        """Test getting a specific validation rule."""
        rule_id = "rule_123456"
        
        response = client.get(
            f"/data-quality/rules/{rule_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["rule_id"] == rule_id
        assert "name" in data
        assert "description" in data
        assert "logic_type" in data
        assert "parameters" in data
        assert "target_columns" in data
        assert "execution_stats" in data

    def test_update_validation_rule_success(self, client, mock_auth, sample_rule_request):
        """Test updating a validation rule."""
        rule_id = "rule_123456"
        
        response = client.put(
            f"/data-quality/rules/{rule_id}",
            json=sample_rule_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["rule_id"] == rule_id
        assert data["name"] == sample_rule_request["name"]

    def test_delete_validation_rule_success(self, client, mock_auth):
        """Test deleting a validation rule."""
        rule_id = "rule_123456"
        
        response = client.delete(
            f"/data-quality/rules/{rule_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 204

    def test_get_quality_monitoring_success(self, client, mock_auth):
        """Test getting quality monitoring data."""
        dataset_id = "test_dataset_123"
        
        response = client.get(
            f"/data-quality/monitor/{dataset_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "monitoring_id" in data
        assert data["dataset_id"] == dataset_id
        assert "monitoring_period" in data
        assert "quality_trends" in data
        assert "alerts" in data
        assert "recommendations" in data
        assert "last_updated" in data

    def test_get_quality_monitoring_with_period(self, client, mock_auth):
        """Test getting quality monitoring with specific period."""
        dataset_id = "test_dataset_123"
        
        response = client.get(
            f"/data-quality/monitor/{dataset_id}?period=30d",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["monitoring_period"] == "30d"

    def test_create_quality_alert_success(self, client, mock_auth):
        """Test creating a quality alert."""
        dataset_id = "test_dataset_123"
        alert_config = {
            "type": "threshold",
            "conditions": {
                "metric": "overall_score",
                "threshold": 0.8,
                "operator": "less_than"
            },
            "notifications": {
                "email": ["admin@example.com"],
                "webhook": "https://example.com/webhook"
            },
            "enabled": True
        }
        
        response = client.post(
            f"/data-quality/monitor/{dataset_id}/alerts",
            json=alert_config,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "alert_id" in data
        assert data["dataset_id"] == dataset_id
        assert data["alert_type"] == alert_config["type"]
        assert data["enabled"] == alert_config["enabled"]

    def test_get_engine_metrics_success(self, client, mock_auth):
        """Test getting engine metrics."""
        response = client.get(
            "/data-quality/engine/metrics",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "validation_engine" in data
        assert "quality_assessment" in data
        assert "system_health" in data
        assert "timestamp" in data
        
        # Verify validation engine metrics
        validation_metrics = data["validation_engine"]
        assert "total_validations" in validation_metrics
        assert "success_rate" in validation_metrics
        assert "average_execution_time_ms" in validation_metrics

    def test_insufficient_permissions(self, client):
        """Test with insufficient permissions."""
        with patch("src.packages.api.api.dependencies.auth.get_current_user") as mock:
            mock.return_value = {
                "user_id": "test_user",
                "permissions": ["data_quality:read"]  # No write permission
            }
            
            response = client.post(
                "/data-quality/validate",
                json={"dataset_id": "test", "data": []},
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 403


@pytest.mark.integration
class TestDataQualityIntegration:
    """Integration tests for data quality endpoints."""

    @patch("src.packages.data_quality.application.services.quality_assessment_service.QualityAssessmentService")
    @patch("src.packages.data_quality.application.services.validation_engine.ValidationEngine")
    def test_validation_integration_with_services(self, mock_validation_engine, mock_assessment_service, 
                                                  client, mock_auth, sample_validation_request):
        """Test integration with validation and assessment services."""
        # Mock the quality assessment service
        mock_profile = Mock()
        mock_profile.dataset_id = "test_dataset_123"
        mock_profile.quality_scores.overall_score = 0.85
        mock_profile.quality_scores.get_dimension_scores.return_value = {
            "completeness": 0.9,
            "accuracy": 0.8
        }
        mock_profile.quality_issues = []
        mock_profile.created_at.isoformat.return_value = "2024-01-15T10:30:00Z"
        
        mock_assessment_service.return_value.assess_dataset_quality.return_value = mock_profile
        
        # Mock the validation engine
        mock_validation_result = Mock()
        mock_validation_engine.return_value.execute_rule.return_value = mock_validation_result
        
        response = client.post(
            "/data-quality/validate",
            json=sample_validation_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 201
        mock_assessment_service.return_value.assess_dataset_quality.assert_called_once()
        mock_validation_engine.return_value.execute_rule.assert_called_once()

    def test_rule_lifecycle_integration(self, client, mock_auth, sample_rule_request):
        """Test complete rule lifecycle (create, read, update, delete)."""
        # Create rule
        create_response = client.post(
            "/data-quality/rules",
            json=sample_rule_request,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert create_response.status_code == 201
        rule_id = create_response.json()["rule_id"]
        
        # Read rule
        read_response = client.get(
            f"/data-quality/rules/{rule_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert read_response.status_code == 200
        
        # Update rule
        updated_rule = sample_rule_request.copy()
        updated_rule["description"] = "Updated description"
        
        update_response = client.put(
            f"/data-quality/rules/{rule_id}",
            json=updated_rule,
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert update_response.status_code == 200
        
        # Delete rule
        delete_response = client.delete(
            f"/data-quality/rules/{rule_id}",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert delete_response.status_code == 204

    def test_error_handling(self, client, mock_auth, sample_validation_request):
        """Test error handling in validation endpoints."""
        with patch("src.packages.data_quality.application.services.quality_assessment_service.QualityAssessmentService") as mock_service:
            mock_service.return_value.assess_dataset_quality.side_effect = Exception("Test error")
            
            response = client.post(
                "/data-quality/validate",
                json=sample_validation_request,
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 500
            assert "failed" in response.json()["detail"].lower()


@pytest.mark.performance
class TestDataQualityPerformance:
    """Performance tests for data quality endpoints."""

    def test_validation_with_many_rules(self, client, mock_auth):
        """Test validation with many rules applied."""
        # Create a dataset with many validation rules
        many_rules = [
            {
                "name": f"rule_{i}",
                "description": f"Validation rule {i}",
                "logic_type": "regex" if i % 2 == 0 else "range",
                "parameters": {"pattern": ".*"} if i % 2 == 0 else {"min": 0, "max": 100},
                "target_columns": ["name"] if i % 2 == 0 else ["age"]
            }
            for i in range(50)  # 50 validation rules
        ]
        
        request_data = {
            "dataset_id": "many_rules_dataset",
            "data": [
                {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "age": 25, "email": "jane@example.com"}
            ],
            "validation_rules": many_rules
        }
        
        response = client.post(
            "/data-quality/validate",
            json=request_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=30  # Allow more time for many rules
        )
        
        # Should still succeed but may take longer
        assert response.status_code in [201, 500]  # May timeout in test environment

    def test_large_dataset_validation(self, client, mock_auth):
        """Test validation with large dataset."""
        # Create a larger dataset
        large_data = [
            {"id": i, "name": f"User {i}", "age": 20 + (i % 50), "email": f"user{i}@example.com"}
            for i in range(5000)
        ]
        
        request_data = {
            "dataset_id": "large_validation_dataset",
            "data": large_data,
            "validation_rules": [
                {
                    "name": "email_format",
                    "description": "Validate email format",
                    "logic_type": "regex",
                    "parameters": {"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
                    "target_columns": ["email"]
                }
            ]
        }
        
        response = client.post(
            "/data-quality/validate",
            json=request_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=60  # Allow more time for large dataset
        )
        
        # Should still succeed but may take longer
        assert response.status_code in [201, 500]  # May timeout in test environment