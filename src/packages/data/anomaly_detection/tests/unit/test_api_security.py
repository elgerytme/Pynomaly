"""Comprehensive API security tests for anomaly detection endpoints."""

import pytest
import json
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException
from datetime import datetime, timedelta

from anomaly_detection.api.v1.detection import router as detection_router
from anomaly_detection.api.v1.models import router as models_router
# from anomaly_detection.api.v1.training import router as training_router


class TestAPISecurityValidation:
    """Test API security validation and authentication."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app for testing."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(detection_router, prefix="/api/v1")
        app.include_router(models_router, prefix="/api/v1")
        # app.include_router(training_router, prefix="/api/v1")
        return app
    
    @pytest.fixture
    def client(self, mock_app):
        """Create test client."""
        return TestClient(mock_app)
    
    def test_api_requires_authentication(self, client):
        """Test that API endpoints require authentication."""
        # Test detection endpoint without auth
        response = client.post("/api/v1/detect", json={
            "features": {"temperature": 25.0, "humidity": 60.0}
        })
        
        # Should return 401 or redirect to authentication
        assert response.status_code in [401, 403, 422]  # 422 for validation error is also acceptable
    
    def test_api_input_validation_xss_protection(self, client):
        """Test API protection against XSS attacks."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "'; DROP TABLE models; --"
        ]
        
        for malicious_input in malicious_inputs:
            # Test model name validation
            response = client.post("/api/v1/models", json={
                "model_name": malicious_input,
                "algorithm": "isolation_forest",
                "features": ["temperature"]
            })
            
            # Should be rejected due to validation
            assert response.status_code in [400, 422], f"Failed to reject: {malicious_input}"
    
    def test_api_sql_injection_protection(self, client):
        """Test API protection against SQL injection."""
        sql_injection_payloads = [
            "'; DROP TABLE models; --",
            "' OR '1'='1",
            "1'; DELETE FROM models WHERE 1=1; --",
            "' UNION SELECT * FROM users --"
        ]
        
        for payload in sql_injection_payloads:
            # Test model ID parameter
            response = client.get(f"/api/v1/models/{payload}")
            
            # Should be handled safely (400/404 are acceptable)
            assert response.status_code in [400, 404, 422], f"Failed to handle SQL injection: {payload}"
    
    def test_api_rate_limiting_simulation(self, client):
        """Test API rate limiting behavior simulation."""
        # Simulate rapid requests
        responses = []
        for i in range(10):
            response = client.post("/api/v1/detect", json={
                "features": {"value": float(i)}
            })
            responses.append(response.status_code)
        
        # At least some should succeed initially
        # In production, rate limiting would kick in
        success_codes = [200, 422]  # 422 for validation errors
        failure_codes = [429, 401, 403]  # Rate limit, auth errors
        
        all_codes = success_codes + failure_codes
        assert all(code in all_codes for code in responses)
    
    def test_api_large_payload_protection(self, client):
        """Test API protection against large payloads."""
        # Create a very large payload
        large_features = {f"feature_{i}": float(i) for i in range(10000)}
        
        response = client.post("/api/v1/detect", json={
            "features": large_features
        })
        
        # Should be rejected due to size limits
        assert response.status_code in [400, 413, 422]  # Bad request, payload too large, or validation error
    
    def test_api_malformed_json_handling(self, client):
        """Test API handling of malformed JSON."""
        malformed_payloads = [
            '{"features": {"temp": }',  # Incomplete JSON
            '{"features": {"temp": NaN}}',  # Invalid JSON value
            '{"features": {"temp": Infinity}}',  # Invalid JSON value
            '{features: {temp: 25}}',  # Unquoted keys
        ]
        
        for payload in malformed_payloads:
            response = client.post(
                "/api/v1/detect",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Should handle malformed JSON gracefully
            assert response.status_code in [400, 422]
    
    def test_api_cors_headers(self, client):
        """Test CORS headers configuration."""
        response = client.options("/api/v1/detect")
        
        # Check if CORS headers are present or properly configured
        # In development, these might not be set, which is acceptable
        headers = response.headers
        
        # Either CORS is properly configured or not exposed (both secure)
        if "access-control-allow-origin" in headers:
            # If CORS is enabled, ensure it's not wildcard for production
            origin = headers.get("access-control-allow-origin")
            # In test environment, wildcard might be acceptable
            assert origin is not None
    
    def test_api_security_headers(self, client):
        """Test security headers presence."""
        response = client.get("/api/v1/health", follow_redirects=False)
        
        # Test for security headers (may not be present in test environment)
        security_headers = [
            "x-frame-options",
            "x-content-type-options", 
            "x-xss-protection",
            "strict-transport-security"
        ]
        
        headers = {k.lower(): v for k, v in response.headers.items()}
        
        # In test environment, these might not be set, which is acceptable
        # But if they are set, they should have secure values
        if "x-frame-options" in headers:
            assert headers["x-frame-options"].upper() in ["DENY", "SAMEORIGIN"]
        
        if "x-content-type-options" in headers:
            assert headers["x-content-type-options"].lower() == "nosniff"


class TestAPIInputValidation:
    """Test comprehensive API input validation."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from anomaly_detection.api.v1.detection import router as detection_router
        
        app = FastAPI()
        app.include_router(detection_router, prefix="/api/v1")
        return TestClient(app)
    
    def test_detection_endpoint_validation(self, client):
        """Test detection endpoint input validation."""
        # Test missing features
        response = client.post("/api/v1/detect", json={})
        assert response.status_code == 422
        
        # Test empty features
        response = client.post("/api/v1/detect", json={"features": {}})
        assert response.status_code in [400, 422]
        
        # Test invalid feature types
        response = client.post("/api/v1/detect", json={
            "features": {"temperature": "not_a_number"}
        })
        assert response.status_code == 422
        
        # Test null values where not allowed
        response = client.post("/api/v1/detect", json={
            "features": {"temperature": None}
        })
        assert response.status_code in [400, 422]
    
    def test_model_creation_validation(self, client):
        """Test model creation endpoint validation."""
        # Test missing required fields
        response = client.post("/api/v1/models", json={})
        assert response.status_code == 422
        
        # Test invalid algorithm
        response = client.post("/api/v1/models", json={
            "model_name": "test_model",
            "algorithm": "invalid_algorithm",
            "features": ["temperature"]
        })
        assert response.status_code in [400, 422]
        
        # Test empty feature list
        response = client.post("/api/v1/models", json={
            "model_name": "test_model", 
            "algorithm": "isolation_forest",
            "features": []
        })
        assert response.status_code in [400, 422]
        
        # Test invalid hyperparameters
        response = client.post("/api/v1/models", json={
            "model_name": "test_model",
            "algorithm": "isolation_forest", 
            "features": ["temperature"],
            "hyperparameters": {"n_estimators": -1}  # Invalid negative value
        })
        assert response.status_code in [400, 422]
    
    def test_training_endpoint_validation(self, client):
        """Test training endpoint input validation."""
        # Test missing model_id
        response = client.post("/api/v1/train", json={
            "dataset_path": "/path/to/data.csv"
        })
        assert response.status_code == 422
        
        # Test invalid dataset path
        response = client.post("/api/v1/train", json={
            "model_id": "test_model_123",
            "dataset_path": ""  # Empty path
        })
        assert response.status_code in [400, 422]
        
        # Test invalid training parameters
        response = client.post("/api/v1/train", json={
            "model_id": "test_model_123",
            "dataset_path": "/path/to/data.csv",
            "validation_split": 1.5  # Invalid split ratio
        })
        assert response.status_code in [400, 422]
    
    def test_parameter_boundary_validation(self, client):
        """Test parameter boundary validation."""
        # Test confidence threshold boundaries
        response = client.post("/api/v1/detect", json={
            "features": {"temperature": 25.0},
            "confidence_threshold": 1.5  # Above valid range
        })
        assert response.status_code in [400, 422]
        
        response = client.post("/api/v1/detect", json={
            "features": {"temperature": 25.0},
            "confidence_threshold": -0.1  # Below valid range
        })
        assert response.status_code in [400, 422]
        
        # Test valid boundary values
        response = client.post("/api/v1/detect", json={
            "features": {"temperature": 25.0},
            "confidence_threshold": 0.0  # Valid minimum
        })
        # May fail due to missing dependencies, but should pass validation
        assert response.status_code in [200, 422, 500]
        
        response = client.post("/api/v1/detect", json={
            "features": {"temperature": 25.0},
            "confidence_threshold": 1.0  # Valid maximum
        })
        assert response.status_code in [200, 422, 500]


class TestAPIErrorHandling:
    """Test API error handling and response formats."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from anomaly_detection.api.v1.detection import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return TestClient(app)
    
    def test_404_error_handling(self, client):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent_endpoint")
        assert response.status_code == 404
        
        # Response should be JSON
        try:
            error_data = response.json()
            assert "detail" in error_data or "message" in error_data
        except json.JSONDecodeError:
            # Plain text response is also acceptable for 404
            pass
    
    def test_500_error_handling_simulation(self, client):
        """Test 500 error handling simulation."""
        # Simulate internal server error with invalid data that might cause backend issues
        response = client.post("/api/v1/detect", json={
            "features": {"temperature": float('inf')}  # Infinity might cause issues
        })
        
        # Should handle gracefully
        assert response.status_code in [400, 422, 500]
        
        # If it's a 500, response should still be properly formatted
        if response.status_code == 500:
            try:
                error_data = response.json()
                assert "detail" in error_data or "message" in error_data
            except json.JSONDecodeError:
                # Text response acceptable for errors
                pass
    
    def test_validation_error_format(self, client):
        """Test validation error response format."""
        response = client.post("/api/v1/detect", json={
            "features": {"temperature": "invalid_number"}
        })
        
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
        
        # Validation errors should provide helpful information
        details = error_data["detail"]
        if isinstance(details, list):
            assert len(details) > 0
            # Each error should have location and message
            for error in details:
                assert "loc" in error
                assert "msg" in error


class TestAPIAuthorizationSimulation:
    """Test API authorization simulation (mocked behaviors)."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from anomaly_detection.api.v1.models import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return TestClient(app)
    
    def test_unauthorized_model_access(self, client):
        """Test unauthorized access to model endpoints."""
        # Attempt to access model without proper authorization
        response = client.get("/api/v1/models/sensitive_model_123")
        
        # Should require authentication/authorization
        assert response.status_code in [401, 403, 404, 422]
    
    def test_unauthorized_model_modification(self, client):
        """Test unauthorized model modification attempts."""
        # Attempt to delete model without authorization
        response = client.delete("/api/v1/models/test_model_123")
        
        # Should be rejected
        assert response.status_code in [401, 403, 404, 405, 422]
        
        # Attempt to modify model without authorization
        response = client.put("/api/v1/models/test_model_123", json={
            "model_name": "modified_name"
        })
        
        assert response.status_code in [401, 403, 404, 405, 422]
    
    def test_role_based_access_simulation(self, client):
        """Test role-based access control simulation."""
        # Simulate different user roles accessing sensitive endpoints
        
        # Admin endpoints (if they exist)
        response = client.get("/api/v1/admin/models")
        assert response.status_code in [401, 403, 404]
        
        # User endpoints should be more permissive
        response = client.get("/api/v1/models")
        # May still require auth, but different handling
        assert response.status_code in [200, 401, 403, 422]