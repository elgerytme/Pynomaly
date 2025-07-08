"""
Comprehensive Validation, Error, and Edge-Case Tests for API Endpoints

This test suite covers:
1. Submit malformed JSON, missing required fields, wrong enum values—assert 422 responses
2. Test boundary values (e.g., max string length, numeric limits)  
3. Trigger internal errors via monkeypatching service layer to raise exceptions; assert 5xx handling and error body
"""

import json
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from uuid import uuid4, UUID
from fastapi.testclient import TestClient
from pynomaly.presentation.api.app import create_app
from pynomaly.domain.exceptions import (
    DatasetError, 
    DetectorError, 
    ValidationError,
    AuthenticationError
)


class TestMalformedJSONValidation:
    """Test malformed JSON requests and assert 422 responses."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get auth headers for protected endpoints."""
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "admin123"
        })
        if login_response.status_code == 200:
            token = login_response.json().get("access_token")
            return {"Authorization": f"Bearer {token}"}
        return {}
    
    def test_malformed_json_auth_register(self, client):
        """Test malformed JSON in auth register endpoint."""
        # Send malformed JSON
        response = client.post(
            "/api/v1/auth/register",
            data='{"username": "test", "email": "test@', # Malformed JSON
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 422 for malformed JSON
        assert response.status_code == 422
    
    def test_missing_required_fields_auth_register(self, client):
        """Test missing required fields in auth register."""
        # Missing username field
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "password123"
                # Missing required 'username' field
            }
        )
        
        # Should return 422 for missing required fields
        assert response.status_code == 422
        
        # Check error details
        error_data = response.json()
        assert "detail" in error_data
        assert any("username" in str(error) for error in error_data["detail"])
    
    def test_missing_required_fields_auth_login(self, client):
        """Test missing required fields in auth login."""
        # Missing password field
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "admin"
                # Missing required 'password' field
            }
        )
        
        # Should return 422 for missing required fields
        assert response.status_code == 422
    
    def test_invalid_field_types_auth_register(self, client):
        """Test invalid field types in auth register."""
        # Invalid email format
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "invalid-email",  # Invalid email format
                "password": "password123"
            }
        )
        
        # Should return 422 for invalid field types
        assert response.status_code == 422
    
    def test_malformed_json_dataset_upload(self, client, auth_headers):
        """Test malformed JSON in dataset upload endpoint."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            temp_file = f.name
        
        try:
            # Send malformed JSON in data
            with open(temp_file, 'rb') as file:
                response = client.post(
                    "/api/v1/datasets/upload",
                    files={"file": ("test.csv", file, "text/csv")},
                    data='{"name": "test", "description": "test"', # Malformed JSON
                    headers=auth_headers
                )
            
            # Should return 422 for malformed JSON
            assert response.status_code == 422
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def test_invalid_enum_values(self, client, auth_headers):
        """Test invalid enum values in requests."""
        # Test with invalid algorithm type (assuming enum validation exists)
        response = client.post(
            "/api/v1/detectors/create",
            json={
                "name": "test_detector",
                "algorithm": "invalid_algorithm",  # Invalid enum value
                "parameters": {}
            },
            headers=auth_headers
        )
        
        # Should return 422 for invalid enum values
        assert response.status_code == 422
    
    def test_invalid_uuid_format(self, client, auth_headers):
        """Test invalid UUID format in path parameters."""
        # Invalid UUID format
        response = client.get(
            "/api/v1/datasets/invalid-uuid-format",
            headers=auth_headers
        )
        
        # Should return 422 for invalid UUID format
        assert response.status_code == 422
    
    def test_invalid_json_structure_complex(self, client, auth_headers):
        """Test complex invalid JSON structure."""
        # Nested malformed JSON
        response = client.post(
            "/api/v1/detectors/create",
            data='{"name": "test", "parameters": {"nested": {"invalid": json}}}',
            headers={"Content-Type": "application/json", **auth_headers}
        )
        
        # Should return 422 for malformed JSON
        assert response.status_code == 422


class TestBoundaryValues:
    """Test boundary values for string length, numeric limits, etc."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get auth headers for protected endpoints."""
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "admin123"
        })
        if login_response.status_code == 200:
            token = login_response.json().get("access_token")
            return {"Authorization": f"Bearer {token}"}
        return {}
    
    def test_max_string_length_username(self, client):
        """Test maximum string length for username."""
        # Very long username (assuming max 50 characters)
        long_username = "a" * 1000  # 1000 characters
        
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": long_username,
                "email": "test@example.com",
                "password": "password123"
            }
        )
        
        # Should return 422 for string too long
        assert response.status_code == 422
    
    def test_max_string_length_description(self, client, auth_headers):
        """Test maximum string length for description fields."""
        # Very long description
        long_description = "a" * 10000  # 10000 characters
        
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            temp_file = f.name
        
        try:
            with open(temp_file, 'rb') as file:
                response = client.post(
                    "/api/v1/datasets/upload",
                    files={"file": ("test.csv", file, "text/csv")},
                    data={
                        "name": "test_dataset",
                        "description": long_description
                    },
                    headers=auth_headers
                )
            
            # Should return 422 for string too long
            assert response.status_code == 422
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def test_empty_string_required_fields(self, client):
        """Test empty strings in required fields."""
        # Empty username
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "",  # Empty string
                "email": "test@example.com",
                "password": "password123"
            }
        )
        
        # Should return 422 for empty required field
        assert response.status_code == 422
    
    def test_min_password_length(self, client):
        """Test minimum password length."""
        # Too short password
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "123"  # Too short
            }
        )
        
        # Should return 422 for password too short
        assert response.status_code == 422
    
    def test_numeric_boundary_values(self, client, auth_headers):
        """Test numeric boundary values."""
        # Test with negative limit
        response = client.get(
            "/api/v1/datasets/",
            params={"limit": -1},  # Negative limit
            headers=auth_headers
        )
        
        # Should return 422 for invalid numeric value
        assert response.status_code == 422
    
    def test_numeric_upper_boundary(self, client, auth_headers):
        """Test numeric upper boundary values."""
        # Test with very large limit
        response = client.get(
            "/api/v1/datasets/",
            params={"limit": 999999},  # Very large limit
            headers=auth_headers
        )
        
        # Should return 422 for limit too high
        assert response.status_code == 422
    
    def test_file_size_boundary(self, client, auth_headers):
        """Test file size boundary values."""
        # Create a large CSV file (assuming max 10MB)
        large_content = "col1,col2\n" + "1,2\n" * 1000000  # Large file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(large_content)
            temp_file = f.name
        
        try:
            with open(temp_file, 'rb') as file:
                response = client.post(
                    "/api/v1/datasets/upload",
                    files={"file": ("large_test.csv", file, "text/csv")},
                    data={"name": "large_dataset"},
                    headers=auth_headers
                )
            
            # Should return 413 for file too large
            assert response.status_code == 413
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def test_zero_values(self, client, auth_headers):
        """Test zero values in numeric fields."""
        # Test with zero limit
        response = client.get(
            "/api/v1/datasets/",
            params={"limit": 0},  # Zero limit
            headers=auth_headers
        )
        
        # Should return 422 for zero limit
        assert response.status_code == 422


class TestInternalErrorHandling:
    """Test internal error handling via monkeypatching."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get auth headers for protected endpoints."""
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "admin123"
        })
        if login_response.status_code == 200:
            token = login_response.json().get("access_token")
            return {"Authorization": f"Bearer {token}"}
        return {}
    
    def test_database_connection_error(self, client, auth_headers):
        """Test database connection error handling."""
        # Monkeypatch the dataset repository to raise database error
        with patch('pynomaly.infrastructure.config.Container.dataset_repository') as mock_repo:
            mock_repo.return_value.find_all.side_effect = ConnectionError("Database connection failed")
            
            response = client.get("/api/v1/datasets/", headers=auth_headers)
            
            # Should return 500 for internal server error
            assert response.status_code == 500
            
            # Check error response structure
            error_data = response.json()
            assert "detail" in error_data
            assert "error" in error_data["detail"].lower()
    
    def test_service_layer_exception(self, client, auth_headers):
        """Test service layer exception handling."""
        # Monkeypatch the dataset repository to raise service exception
        with patch('pynomaly.infrastructure.config.Container.dataset_repository') as mock_repo:
            mock_repo.return_value.find_by_id.side_effect = RuntimeError("Service unavailable")
            
            test_dataset_id = str(uuid4())
            response = client.get(f"/api/v1/datasets/{test_dataset_id}", headers=auth_headers)
            
            # Should return 500 for internal server error
            assert response.status_code == 500
            
            # Check error response structure
            error_data = response.json()
            assert "detail" in error_data
    
    def test_auth_service_exception(self, client):
        """Test authentication service exception handling."""
        # Monkeypatch the auth service to raise exception
        with patch('pynomaly.infrastructure.auth.get_auth') as mock_auth:
            mock_auth.return_value = None  # Simulate auth service unavailable
            
            response = client.post("/api/v1/auth/login", data={
                "username": "admin",
                "password": "admin123"
            })
            
            # Should return 503 for service unavailable
            assert response.status_code == 503
            
            # Check error response structure
            error_data = response.json()
            assert "detail" in error_data
            assert "service not available" in error_data["detail"].lower()
    
    def test_file_processing_exception(self, client, auth_headers):
        """Test file processing exception handling."""
        # Create a malformed CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("malformed,csv,content\n")
            f.write("1,2\n")  # Inconsistent column count
            temp_file = f.name
        
        try:
            # Monkeypatch pandas to raise processing error
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = Exception("File processing failed")
                
                with open(temp_file, 'rb') as file:
                    response = client.post(
                        "/api/v1/datasets/upload",
                        files={"file": ("malformed.csv", file, "text/csv")},
                        data={"name": "test_dataset"},
                        headers=auth_headers
                    )
                
                # Should return 400 for bad request (file processing error)
                assert response.status_code == 400
                
                # Check error response structure
                error_data = response.json()
                assert "detail" in error_data
                assert "failed" in error_data["detail"].lower()
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def test_validation_error_handling(self, client, auth_headers):
        """Test validation error handling from domain layer."""
        # Monkeypatch the domain entity to raise validation error
        with patch('pynomaly.domain.entities.Dataset.__init__') as mock_init:
            mock_init.side_effect = ValidationError("Invalid dataset configuration")
            
            # Create a temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("col1,col2\n1,2\n3,4\n")
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as file:
                    response = client.post(
                        "/api/v1/datasets/upload",
                        files={"file": ("test.csv", file, "text/csv")},
                        data={"name": "test_dataset"},
                        headers=auth_headers
                    )
                
                # Should return 400 for validation error
                assert response.status_code == 400
                
                # Check error response structure
                error_data = response.json()
                assert "detail" in error_data
                assert "invalid" in error_data["detail"].lower()
            finally:
                # Cleanup
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def test_memory_error_handling(self, client, auth_headers):
        """Test memory error handling."""
        # Monkeypatch to raise memory error
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = MemoryError("Not enough memory")
            
            # Create a temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("col1,col2\n1,2\n3,4\n")
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as file:
                    response = client.post(
                        "/api/v1/datasets/upload",
                        files={"file": ("test.csv", file, "text/csv")},
                        data={"name": "test_dataset"},
                        headers=auth_headers
                    )
                
                # Should return 500 for internal server error
                assert response.status_code == 500
                
                # Check error response structure
                error_data = response.json()
                assert "detail" in error_data
            finally:
                # Cleanup
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def test_timeout_error_handling(self, client, auth_headers):
        """Test timeout error handling."""
        # Monkeypatch to raise timeout error
        with patch('pynomaly.infrastructure.config.Container.dataset_repository') as mock_repo:
            mock_repo.return_value.find_all.side_effect = TimeoutError("Request timeout")
            
            response = client.get("/api/v1/datasets/", headers=auth_headers)
            
            # Should return 500 for internal server error
            assert response.status_code == 500
            
            # Check error response structure
            error_data = response.json()
            assert "detail" in error_data
    
    def test_unexpected_exception_handling(self, client, auth_headers):
        """Test unexpected exception handling."""
        # Monkeypatch to raise unexpected exception
        with patch('pynomaly.infrastructure.config.Container.dataset_repository') as mock_repo:
            mock_repo.return_value.find_all.side_effect = Exception("Unexpected error")
            
            response = client.get("/api/v1/datasets/", headers=auth_headers)
            
            # Should return 500 for internal server error
            assert response.status_code == 500
            
            # Check error response structure
            error_data = response.json()
            assert "detail" in error_data


class TestErrorResponseStructure:
    """Test error response structure and consistency."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_422_error_structure(self, client):
        """Test 422 error response structure."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "",  # Empty required field
                "email": "test@example.com",
                "password": "password123"
            }
        )
        
        assert response.status_code == 422
        error_data = response.json()
        
        # Check standard error structure
        assert "detail" in error_data
        assert isinstance(error_data["detail"], list)
        
        # Check validation error details
        for error in error_data["detail"]:
            assert "loc" in error
            assert "msg" in error
            assert "type" in error
    
    def test_400_error_structure(self, client):
        """Test 400 error response structure."""
        # Create invalid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid file content")
            temp_file = f.name
        
        try:
            with open(temp_file, 'rb') as file:
                response = client.post(
                    "/api/v1/datasets/upload",
                    files={"file": ("test.txt", file, "text/plain")},
                    data={"name": "test_dataset"}
                )
            
            assert response.status_code == 400
            error_data = response.json()
            
            # Check standard error structure
            assert "detail" in error_data
            assert isinstance(error_data["detail"], str)
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def test_401_error_structure(self, client):
        """Test 401 error response structure."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 401
        error_data = response.json()
        
        # Check standard error structure
        assert "detail" in error_data
        assert isinstance(error_data["detail"], str)
    
    def test_404_error_structure(self, client):
        """Test 404 error response structure."""
        # Try to access non-existent endpoint
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
        error_data = response.json()
        
        # Check standard error structure
        assert "detail" in error_data
        assert isinstance(error_data["detail"], str)
    
    def test_405_error_structure(self, client):
        """Test 405 error response structure."""
        # Try unsupported method
        response = client.put("/api/v1/health/")
        
        assert response.status_code == 405
        error_data = response.json()
        
        # Check standard error structure
        assert "detail" in error_data
        assert isinstance(error_data["detail"], str)


class TestEdgeCaseScenarios:
    """Test edge case scenarios and combinations."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get auth headers for protected endpoints."""
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "admin123"
        })
        if login_response.status_code == 200:
            token = login_response.json().get("access_token")
            return {"Authorization": f"Bearer {token}"}
        return {}
    
    def test_concurrent_requests_with_errors(self, client, auth_headers):
        """Test concurrent requests that cause errors."""
        import threading
        import time
        
        results = []
        
        def make_request():
            # Make request that will cause error
            response = client.get(f"/api/v1/datasets/{uuid4()}", headers=auth_headers)
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All should return 404 (not found)
        assert all(status == 404 for status in results)
    
    def test_malformed_json_with_special_characters(self, client):
        """Test malformed JSON with special characters."""
        # JSON with special characters
        malformed_json = '{"username": "test", "email": "test@example.com", "password": "pass\\u0000word"}'
        
        response = client.post(
            "/api/v1/auth/register",
            data=malformed_json,
            headers={"Content-Type": "application/json"}
        )
        
        # Should handle special characters appropriately
        assert response.status_code in [400, 422]
    
    def test_empty_file_upload(self, client, auth_headers):
        """Test empty file upload."""
        # Create empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write nothing to file
            temp_file = f.name
        
        try:
            with open(temp_file, 'rb') as file:
                response = client.post(
                    "/api/v1/datasets/upload",
                    files={"file": ("empty.csv", file, "text/csv")},
                    data={"name": "empty_dataset"},
                    headers=auth_headers
                )
            
            # Should return 400 for empty file
            assert response.status_code == 400
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def test_unicode_in_requests(self, client):
        """Test unicode characters in requests."""
        # Unicode characters in username
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "用户名",  # Chinese characters
                "email": "test@example.com",
                "password": "password123"
            }
        )
        
        # Should handle unicode appropriately
        assert response.status_code in [200, 201, 400, 422]
    
    def test_sql_injection_attempts(self, client, auth_headers):
        """Test SQL injection attempts in parameters."""
        # SQL injection attempt in query parameter
        response = client.get(
            "/api/v1/datasets/",
            params={"limit": "1; DROP TABLE datasets;"},
            headers=auth_headers
        )
        
        # Should return 422 for invalid parameter type
        assert response.status_code == 422
    
    def test_xss_attempts(self, client):
        """Test XSS attempts in request data."""
        # XSS attempt in registration data
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "<script>alert('xss')</script>",
                "email": "test@example.com",
                "password": "password123"
            }
        )
        
        # Should handle XSS attempts appropriately
        assert response.status_code in [200, 201, 400, 422]
    
    def test_extremely_long_request_urls(self, client, auth_headers):
        """Test extremely long request URLs."""
        # Very long query parameter
        long_param = "a" * 10000
        
        response = client.get(
            f"/api/v1/datasets/?description={long_param}",
            headers=auth_headers
        )
        
        # Should handle long URLs appropriately
        assert response.status_code in [400, 414, 422]
    
    def test_multiple_errors_in_single_request(self, client):
        """Test multiple validation errors in single request."""
        # Multiple validation errors
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "",  # Empty username
                "email": "invalid-email",  # Invalid email
                "password": "123"  # Too short password
            }
        )
        
        assert response.status_code == 422
        error_data = response.json()
        
        # Should contain multiple error details
        assert "detail" in error_data
        assert len(error_data["detail"]) > 1
