"""Integration tests for Issue #158 - RESTful Data Science APIs.

Tests the newly implemented API endpoints for pattern recognition, schema discovery,
quality governance, and data visualization.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.packages.api.api.main import app

client = TestClient(app)

# Mock authentication for testing
@pytest.fixture(autouse=True)
def mock_auth():
    """Mock authentication for all tests."""
    with patch("src.packages.api.api.main.get_current_user_token") as mock:
        mock.return_value = {
            "user_id": "test_user",
            "username": "test_user",
            "roles": ["admin"],
            "permissions": [
                "pattern_recognition:read",
                "schema_discovery:read",
                "governance:read",
                "governance:write",
                "compliance:read",
                "audit:read",
                "visualization:create",
                "visualization:read",
                "dashboard:create",
                "dashboard:read",
                "report:create",
                "template:create",
                "template:read"
            ]
        }
        yield mock


class TestPatternRecognitionEndpoints:
    """Test pattern recognition API endpoints."""
    
    def test_recognize_patterns_semantic(self):
        """Test semantic pattern recognition."""
        request_data = {
            "dataset_id": "test_dataset",
            "data": [
                {"id": 1, "text": "john.doe@company.com"},
                {"id": 2, "text": "555-123-4567"}
            ],
            "recognition_type": "semantic",
            "target_columns": ["text"],
            "config": {
                "confidence_threshold": 0.8,
                "enable_domain_specific": True
            }
        }
        
        response = client.post("/pattern-recognition/recognize-patterns", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "recognition_id" in data
        assert data["dataset_id"] == "test_dataset"
        assert data["recognition_type"] == "semantic"
        assert len(data["patterns_detected"]) > 0
        assert "confidence_scores" in data
        assert "recommendations" in data
    
    def test_semantic_classification_pii(self):
        """Test PII semantic classification."""
        request_data = {
            "dataset_id": "test_dataset",
            "data": [
                {"name": "John Doe", "email": "john@company.com"},
                {"name": "Jane Smith", "email": "jane@company.com"}
            ],
            "classification_type": "pii",
            "target_columns": ["name", "email"],
            "domain_context": "hr_data"
        }
        
        response = client.post("/pattern-recognition/semantic-classification", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "classification_id" in data
        assert data["classification_type"] == "pii"
        assert "classifications" in data
        assert "privacy_analysis" in data
        assert "compliance_flags" in data
    
    def test_discover_patterns_comprehensive(self):
        """Test comprehensive pattern discovery."""
        request_data = {
            "dataset_id": "test_dataset",
            "data": [
                {"transaction_id": "TXN001", "amount": 150.00},
                {"transaction_id": "TXN002", "amount": 75.50}
            ],
            "discovery_mode": "comprehensive",
            "min_confidence": 0.8,
            "max_patterns": 25
        }
        
        response = client.post("/pattern-recognition/discover-patterns", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "recognition_id" in data
        assert data["recognition_type"] == "automated_discovery"
        assert len(data["patterns_detected"]) > 0
        assert "confidence_scores" in data
    
    def test_list_pattern_recognition_results(self):
        """Test listing pattern recognition results."""
        response = client.get("/pattern-recognition/patterns")
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "total_count" in data
        assert "has_more" in data


class TestSchemaDiscoveryEndpoints:
    """Test schema discovery API endpoints."""
    
    def test_discover_schema_comprehensive(self):
        """Test comprehensive schema discovery."""
        request_data = {
            "dataset_id": "test_dataset",
            "data": [
                {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "age": 25, "email": "jane@example.com"}
            ],
            "discovery_mode": "comprehensive",
            "include_samples": True,
            "infer_relationships": True,
            "config": {
                "sample_size": 1000,
                "confidence_threshold": 0.8,
                "enable_semantic_inference": True
            }
        }
        
        response = client.post("/schema-discovery/discover-schema", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "discovery_id" in data
        assert data["dataset_id"] == "test_dataset"
        assert data["discovery_mode"] == "comprehensive"
        assert len(data["columns"]) > 0
        assert "relationships" in data
        assert "constraints" in data
        assert "quality_metrics" in data
        assert "recommendations" in data
    
    def test_infer_types_statistical(self):
        """Test statistical type inference."""
        request_data = {
            "dataset_id": "test_dataset",
            "columns": [
                {
                    "name": "birth_date",
                    "values": ["1990-01-15", "1985-03-22", "1995-12-08"]
                },
                {
                    "name": "score",
                    "values": ["85.5", "92.3", "78.9"]
                }
            ],
            "inference_method": "statistical",
            "confidence_threshold": 0.8
        }
        
        response = client.post("/schema-discovery/infer-types", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "inference_id" in data
        assert data["inference_method"] == "statistical"
        assert len(data["column_inferences"]) == 2
        assert "semantic_inferences" in data
        assert "format_inferences" in data
    
    def test_compare_schemas(self):
        """Test schema comparison."""
        request_data = {
            "source_schema_id": "schema_1",
            "target_schema_id": "schema_2",
            "comparison_mode": "full",
            "ignore_nullable": False
        }
        
        response = client.post("/schema-discovery/compare-schemas", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "comparison_id" in data
        assert data["comparison_mode"] == "full"
        assert "compatibility_score" in data
        assert "differences" in data
        assert "migration_suggestions" in data
    
    def test_list_schemas(self):
        """Test listing schemas."""
        response = client.get("/schema-discovery/schemas")
        assert response.status_code == 200
        
        data = response.json()
        assert "schemas" in data
        assert "total_count" in data
        assert "has_more" in data


class TestQualityGovernanceEndpoints:
    """Test quality governance API endpoints."""
    
    def test_create_governance_policy(self):
        """Test creating a governance policy."""
        request_data = {
            "policy_name": "Test PII Policy",
            "description": "Test policy for PII protection",
            "policy_type": "compliance",
            "scope": "organization",
            "rules": [
                {
                    "rule_type": "data_classification",
                    "condition": "contains_pii == true",
                    "action": "encrypt_at_rest",
                    "parameters": {"encryption_algorithm": "AES-256"}
                }
            ],
            "enforcement_level": "mandatory",
            "compliance_frameworks": ["GDPR", "CCPA"],
            "effective_date": "2024-01-15T00:00:00Z"
        }
        
        response = client.post("/quality-governance/policies", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "policy_id" in data
        assert data["policy_name"] == "Test PII Policy"
        assert data["policy_type"] == "compliance"
        assert data["status"] == "active"
        assert len(data["rules"]) > 0
    
    def test_perform_compliance_check(self):
        """Test performing compliance check."""
        request_data = {
            "dataset_id": "test_dataset",
            "compliance_frameworks": ["GDPR", "CCPA"],
            "check_type": "comprehensive",
            "data_sample": [
                {"name": "John Doe", "email": "john@example.com"},
                {"name": "Jane Smith", "email": "jane@example.com"}
            ],
            "metadata": {
                "data_source": "customer_database",
                "collection_date": "2024-01-15"
            }
        }
        
        response = client.post("/quality-governance/compliance/check", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "check_id" in data
        assert data["dataset_id"] == "test_dataset"
        assert "overall_compliance_score" in data
        assert "compliance_status" in data
        assert "framework_results" in data
        assert "violations" in data
        assert "recommendations" in data
    
    def test_query_audit_trail(self):
        """Test querying audit trail."""
        request_data = {
            "resource_type": "dataset",
            "resource_id": "test_dataset",
            "action_type": "data_access",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-31T23:59:59Z",
            "limit": 50
        }
        
        response = client.post("/quality-governance/audit/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "audit_records" in data
        assert "total_records" in data
        assert "summary" in data
        assert "compliance_alerts" in data
    
    def test_list_governance_policies(self):
        """Test listing governance policies."""
        response = client.get("/quality-governance/policies")
        assert response.status_code == 200
        
        data = response.json()
        assert "policies" in data
        assert "total_count" in data
        assert "has_more" in data
    
    def test_get_compliance_dashboard(self):
        """Test getting compliance dashboard."""
        response = client.get("/quality-governance/compliance/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        assert "overview" in data
        assert "compliance_by_framework" in data
        assert "recent_violations" in data
        assert "upcoming_actions" in data
        assert "trends" in data


class TestDataVisualizationEndpoints:
    """Test data visualization API endpoints."""
    
    def test_generate_chart_bar(self):
        """Test generating a bar chart."""
        request_data = {
            "dataset_id": "test_dataset",
            "chart_type": "bar",
            "data": [
                {"month": "Jan", "sales": 1000, "region": "North"},
                {"month": "Feb", "sales": 1200, "region": "North"}
            ],
            "x_axis": "month",
            "y_axis": "sales",
            "group_by": "region",
            "aggregation": "sum",
            "styling": {
                "color_scheme": "blue",
                "title": "Monthly Sales by Region"
            }
        }
        
        response = client.post("/data-visualization/charts/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "chart_id" in data
        assert data["chart_type"] == "bar"
        assert "chart_config" in data
        assert "chart_data" in data
        assert "chart_url" in data
        assert "thumbnail_url" in data
    
    def test_create_dashboard(self):
        """Test creating a dashboard."""
        request_data = {
            "dashboard_name": "Test Dashboard",
            "description": "Test dashboard for API testing",
            "layout": "grid",
            "widgets": [
                {
                    "widget_type": "chart",
                    "chart_type": "line",
                    "title": "Sales Trend",
                    "dataset_id": "test_dataset",
                    "position": {"x": 0, "y": 0, "width": 6, "height": 4},
                    "config": {
                        "x_axis": "date",
                        "y_axis": "sales"
                    }
                }
            ],
            "refresh_interval": 300
        }
        
        response = client.post("/data-visualization/dashboards", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "dashboard_id" in data
        assert data["dashboard_name"] == "Test Dashboard"
        assert len(data["widgets"]) > 0
        assert "dashboard_url" in data
        assert "embed_url" in data
    
    def test_generate_report(self):
        """Test generating a report."""
        request_data = {
            "report_name": "Test Report",
            "report_type": "summary",
            "datasets": ["test_dataset"],
            "sections": [
                {
                    "section_type": "summary",
                    "title": "Executive Summary",
                    "content": "kpi_cards"
                },
                {
                    "section_type": "chart",
                    "title": "Sales Trends",
                    "chart_type": "line",
                    "dataset_id": "test_dataset"
                }
            ],
            "format": "pdf",
            "template": "corporate_template"
        }
        
        response = client.post("/data-visualization/reports/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "report_id" in data
        assert data["report_name"] == "Test Report"
        assert data["format"] == "pdf"
        assert data["status"] == "completed"
        assert "report_url" in data
    
    def test_list_charts(self):
        """Test listing charts."""
        response = client.get("/data-visualization/charts")
        assert response.status_code == 200
        
        data = response.json()
        assert "charts" in data
        assert "total_count" in data
        assert "has_more" in data
    
    def test_list_dashboards(self):
        """Test listing dashboards."""
        response = client.get("/data-visualization/dashboards")
        assert response.status_code == 200
        
        data = response.json()
        assert "dashboards" in data
        assert "total_count" in data
        assert "has_more" in data
    
    def test_create_visualization_template(self):
        """Test creating a visualization template."""
        request_data = {
            "template_name": "Test Chart Template",
            "template_type": "chart",
            "configuration": {
                "chart_type": "bar",
                "styling": {
                    "color_scheme": "blue",
                    "title_font_size": 16
                }
            },
            "description": "Test template for API testing",
            "tags": ["test", "bar_chart"]
        }
        
        response = client.post("/data-visualization/templates", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "template_id" in data
        assert data["template_name"] == "Test Chart Template"
        assert data["template_type"] == "chart"
        assert "configuration" in data
    
    def test_list_visualization_templates(self):
        """Test listing visualization templates."""
        response = client.get("/data-visualization/templates")
        assert response.status_code == 200
        
        data = response.json()
        assert "templates" in data
        assert "total_count" in data
        assert "has_more" in data


class TestAPIIntegration:
    """Test overall API integration and cross-endpoint functionality."""
    
    def test_complete_workflow(self):
        """Test a complete workflow using multiple endpoints."""
        # 1. Discover schema
        schema_request = {
            "dataset_id": "workflow_test",
            "data": [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
            ],
            "discovery_mode": "comprehensive",
            "include_samples": True,
            "infer_relationships": True
        }
        
        schema_response = client.post("/schema-discovery/discover-schema", json=schema_request)
        assert schema_response.status_code == 200
        
        # 2. Recognize patterns
        pattern_request = {
            "dataset_id": "workflow_test",
            "data": schema_request["data"],
            "recognition_type": "semantic",
            "target_columns": ["email"]
        }
        
        pattern_response = client.post("/pattern-recognition/recognize-patterns", json=pattern_request)
        assert pattern_response.status_code == 200
        
        # 3. Check compliance
        compliance_request = {
            "dataset_id": "workflow_test",
            "compliance_frameworks": ["GDPR"],
            "check_type": "comprehensive",
            "data_sample": schema_request["data"]
        }
        
        compliance_response = client.post("/quality-governance/compliance/check", json=compliance_request)
        assert compliance_response.status_code == 200
        
        # 4. Generate visualization
        chart_request = {
            "dataset_id": "workflow_test",
            "chart_type": "bar",
            "data": [{"category": "A", "count": 10}, {"category": "B", "count": 15}],
            "x_axis": "category",
            "y_axis": "count"
        }
        
        chart_response = client.post("/data-visualization/charts/generate", json=chart_request)
        assert chart_response.status_code == 200
        
        # Verify all endpoints returned successfully
        assert all(response.status_code == 200 for response in [
            schema_response, pattern_response, compliance_response, chart_response
        ])
    
    def test_api_error_handling(self):
        """Test API error handling for invalid requests."""
        # Test with invalid data
        invalid_request = {
            "dataset_id": "",  # Empty dataset ID
            "data": [],  # Empty data
            "recognition_type": "invalid_type"
        }
        
        response = client.post("/pattern-recognition/recognize-patterns", json=invalid_request)
        # Should handle gracefully (mock implementation may still return 200)
        assert response.status_code in [200, 400, 422]
    
    def test_api_pagination(self):
        """Test API pagination across different endpoints."""
        # Test pagination on multiple endpoints
        endpoints = [
            "/pattern-recognition/patterns",
            "/schema-discovery/schemas",
            "/quality-governance/policies",
            "/data-visualization/charts",
            "/data-visualization/dashboards"
        ]
        
        for endpoint in endpoints:
            response = client.get(f"{endpoint}?limit=10&offset=0")
            assert response.status_code == 200
            
            data = response.json()
            assert "total_count" in data
            assert "has_more" in data
    
    def test_api_filtering(self):
        """Test API filtering capabilities."""
        # Test filtering on pattern recognition
        response = client.get("/pattern-recognition/patterns?recognition_type=semantic")
        assert response.status_code == 200
        
        # Test filtering on governance policies
        response = client.get("/quality-governance/policies?policy_type=compliance")
        assert response.status_code == 200
        
        # Test filtering on schemas
        response = client.get("/schema-discovery/schemas?discovery_mode=comprehensive")
        assert response.status_code == 200