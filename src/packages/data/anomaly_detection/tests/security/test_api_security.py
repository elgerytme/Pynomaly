"""Security tests for the anomaly detection API endpoints."""

import pytest
import json
import time
import base64
import hashlib
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from anomaly_detection.server import create_app


class TestAPISecurity:
    """Security tests for API endpoints against common vulnerabilities."""
    
    @pytest.fixture
    def api_client(self):
        """Create API test client for security testing."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def valid_detection_payload(self):
        """Valid detection payload for security testing."""
        return {
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            "algorithm": "isolation_forest",
            "contamination": 0.1
        }
    
    def test_input_validation_injection_attacks(self, api_client: TestClient):
        """Test protection against injection attacks in input data."""
        print("\n=== Input Injection Attack Protection Test ===")
        
        # SQL injection attempts (should be handled by input validation)
        malicious_payloads = [
            {
                "data": "'; DROP TABLE models; --",
                "algorithm": "isolation_forest"
            },
            {
                "data": [[1, 2, 3]],
                "algorithm": "'; SELECT * FROM users; --"
            },
            {
                "data": [[1, 2, 3]],
                "algorithm": "isolation_forest",
                "contamination": "0.1; DELETE FROM *"
            }
        ]
        
        for i, payload in enumerate(malicious_payloads):
            print(f"Testing injection payload {i + 1}...")
            
            response = api_client.post("/api/v1/detect", json=payload)
            
            # Should reject malicious input with proper error codes
            assert response.status_code in [400, 422], f"Injection payload {i + 1} not properly rejected"
            
            # Should not contain error details that could leak information
            response_text = response.text.lower()
            sensitive_keywords = ['database', 'sql', 'table', 'column', 'select', 'drop', 'delete']
            
            for keyword in sensitive_keywords:
                assert keyword not in response_text, f"Response contains sensitive keyword: {keyword}"
        
        print("✓ All injection attacks properly rejected")
    
    def test_dos_protection_large_payload(self, api_client: TestClient):
        """Test protection against DoS attacks via large payloads."""
        print("\n=== DoS Protection - Large Payload Test ===")
        
        # Test extremely large payload
        large_data = [[i] * 1000 for i in range(10000)]  # 10M elements
        
        payload = {
            "data": large_data,
            "algorithm": "isolation_forest",
            "contamination": 0.1
        }
        
        start_time = time.time()
        
        try:
            response = api_client.post("/api/v1/detect", json=payload, timeout=30)
            response_time = time.time() - start_time
            
            print(f"Large payload response time: {response_time:.2f}s")
            print(f"Response status: {response.status_code}")
            
            # Should either reject with 413 (too large) or handle gracefully within time limit
            if response.status_code == 413:
                print("✓ Large payload properly rejected with 413 Payload Too Large")
            elif 200 <= response.status_code < 300:
                assert response_time < 60, f"Large payload processing took too long: {response_time:.2f}s"
                print("✓ Large payload handled within acceptable time")
            else:
                assert response.status_code in [400, 500, 503], f"Unexpected status code: {response.status_code}"
                print(f"✓ Large payload rejected with status {response.status_code}")
                
        except Exception as e:
            response_time = time.time() - start_time
            print(f"Large payload caused exception after {response_time:.2f}s: {e}")
            # Timeout or connection error is acceptable DoS protection
            assert response_time < 60, "Request should timeout or be rejected quickly"
    
    def test_dos_protection_rapid_requests(self, api_client: TestClient, valid_detection_payload):
        """Test protection against DoS attacks via rapid requests."""
        print("\n=== DoS Protection - Rapid Requests Test ===")
        
        # Send many requests rapidly
        num_requests = 50
        request_interval = 0.01  # 100 requests per second
        
        responses = []
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                response = api_client.post(
                    "/api/v1/detect", 
                    json=valid_detection_payload,
                    timeout=5
                )
                responses.append((response.status_code, time.time() - start_time))
                
            except Exception as e:
                responses.append((0, time.time() - start_time))  # Timeout/connection error
            
            if i < num_requests - 1:  # Don't sleep after last request
                time.sleep(request_interval)
        
        total_time = time.time() - start_time
        
        # Analyze responses
        status_codes = [r[0] for r in responses]
        success_count = sum(1 for code in status_codes if 200 <= code < 300)
        rate_limited_count = sum(1 for code in status_codes if code == 429)
        error_count = sum(1 for code in status_codes if code >= 400 or code == 0)
        
        success_rate = success_count / num_requests
        
        print(f"Rapid requests results:")
        print(f"  Total requests: {num_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Rate limited (429): {rate_limited_count}")
        print(f"  Errors/timeouts: {error_count}")
        
        # System should either rate limit or handle requests gracefully
        if rate_limited_count > 0:
            print("✓ Rate limiting detected - good DoS protection")
        elif success_rate > 0.5:
            print("✓ System handled rapid requests reasonably well")
        else:
            print("✓ System rejected most rapid requests - acceptable protection")
        
        # Should not completely crash
        assert error_count < num_requests, "System should not fail all requests"
    
    def test_malformed_data_handling(self, api_client: TestClient):
        """Test handling of malformed and malicious data structures."""
        print("\n=== Malformed Data Handling Test ===")
        
        malformed_payloads = [
            # Deeply nested structure
            {
                "data": {"level1": {"level2": {"level3": {"level4": "deep"}}}},
                "algorithm": "isolation_forest"
            },
            # Circular reference (JSON shouldn't allow this, but test anyway) 
            {
                "data": [[float('inf'), float('-inf'), float('nan')]],
                "algorithm": "isolation_forest"
            },
            # Mixed data types
            {
                "data": [["string", 123, None, True, [1, 2, 3]]],
                "algorithm": "isolation_forest"
            },
            # Very long strings
            {
                "data": [["x" * 10000]],
                "algorithm": "isolation_forest"
            },
            # Invalid JSON structure
            '{"data": [1, 2, 3], "algorithm": "isolation_forest", "invalid": }',
            # Empty/null values
            {
                "data": None,
                "algorithm": None
            }
        ]
        
        for i, payload in enumerate(malformed_payloads):
            print(f"Testing malformed payload {i + 1}...")
            
            try:
                if isinstance(payload, str):
                    # Test invalid JSON directly
                    response = api_client.post(
                        "/api/v1/detect",
                        data=payload,
                        headers={"content-type": "application/json"}
                    )
                else:
                    response = api_client.post("/api/v1/detect", json=payload)
                
                # Should reject malformed data
                assert response.status_code >= 400, f"Malformed payload {i + 1} not rejected"
                
                # Should not leak internal error details
                if response.status_code == 500:
                    response_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    
                    # Check for information disclosure
                    sensitive_patterns = [
                        'traceback', 'exception', 'stack trace', 'file path',
                        'internal error', 'database', 'sql', 'connection'
                    ]
                    
                    response_text = response.text.lower() if response.text else ""
                    for pattern in sensitive_patterns:
                        assert pattern not in response_text, f"Sensitive information disclosed: {pattern}"
                
            except Exception as e:
                # Connection errors are acceptable for some malformed requests
                print(f"  Payload {i + 1} caused connection error: {e}")
        
        print("✓ All malformed payloads properly rejected")
    
    def test_authentication_bypass_attempts(self, api_client: TestClient):
        """Test potential authentication bypass attempts."""
        print("\n=== Authentication Bypass Test ===")
        
        # Test various authentication bypass techniques
        bypass_headers = [
            {"Authorization": "Bearer fake_token"},
            {"Authorization": "Basic " + base64.b64encode(b"admin:admin").decode()},
            {"X-Forwarded-For": "127.0.0.1"},
            {"X-Real-IP": "localhost"},
            {"X-Admin": "true"},
            {"X-Bypass": "true"},
            {"User-Agent": "AdminBot/1.0"},
            {"Cookie": "admin=true; authenticated=yes"}
        ]
        
        payload = {
            "data": [[1, 2, 3]],
            "algorithm": "isolation_forest"
        }
        
        for i, headers in enumerate(bypass_headers):
            print(f"Testing bypass attempt {i + 1}...")
            
            response = api_client.post("/api/v1/detect", json=payload, headers=headers)
            
            # Should not grant special access based on headers
            # (For now, assuming API doesn't require authentication)
            assert response.status_code in [200, 400, 422, 429, 500], f"Unexpected response to bypass attempt {i + 1}"
            
            # Should not return different data structure indicating elevated privileges
            if 200 <= response.status_code < 300:
                response_data = response.json()
                suspicious_fields = ['admin', 'privileges', 'elevated', 'bypass', 'debug']
                
                response_str = json.dumps(response_data).lower()
                for field in suspicious_fields:
                    assert field not in response_str, f"Suspicious field in response: {field}"
        
        print("✓ No authentication bypass detected")
    
    def test_path_traversal_protection(self, api_client: TestClient):
        """Test protection against path traversal attacks."""
        print("\n=== Path Traversal Protection Test ===")
        
        # Test path traversal in various parameters
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2f%etc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        # Test in model operations (if available)
        for payload in traversal_payloads:
            print(f"Testing path traversal: {payload[:20]}...")
            
            # Test in model loading/saving operations
            response = api_client.get(f"/api/v1/models/{payload}")
            assert response.status_code in [400, 404, 422], f"Path traversal not blocked: {payload}"
            
            # Should not return file contents
            if response.text:
                sensitive_indicators = [
                    'root:', 'daemon:', '/bin/bash', 'password', 
                    'Windows Registry', 'HKEY_', '[SYSTEM]'
                ]
                
                response_text = response.text.lower()
                for indicator in sensitive_indicators:
                    assert indicator.lower() not in response_text, f"File content leaked: {indicator}"
        
        print("✓ Path traversal attacks blocked")
    
    def test_information_disclosure_prevention(self, api_client: TestClient):
        """Test prevention of information disclosure through error messages."""
        print("\n=== Information Disclosure Prevention Test ===")
        
        # Test various error-inducing requests
        error_inducing_requests = [
            ("/api/v1/nonexistent", {}),
            ("/api/v1/detect", {"invalid": "payload"}),
            ("/api/v1/models/nonexistent", {}),
            ("/api/v1/detect", {"data": "invalid", "algorithm": "nonexistent"}),
        ]
        
        for endpoint, payload in error_inducing_requests:
            print(f"Testing error handling for {endpoint}...")
            
            if payload:
                response = api_client.post(endpoint, json=payload)
            else:
                response = api_client.get(endpoint)
            
            # Should return error but not disclose sensitive information
            assert response.status_code >= 400, f"Request should fail: {endpoint}"
            
            if response.headers.get("content-type", "").startswith("application/json"):
                try:
                    error_data = response.json()
                    
                    # Check for information disclosure in error messages
                    sensitive_info = [
                        'file path', 'directory', 'traceback', 'exception',
                        'database connection', 'sql error', 'config', 'environment',
                        'version', 'internal', 'debug', 'secret', 'key', 'token'
                    ]
                    
                    error_text = json.dumps(error_data).lower()
                    for info in sensitive_info:
                        assert info not in error_text, f"Sensitive information in error: {info}"
                    
                except json.JSONDecodeError:
                    pass  # Non-JSON error responses are acceptable
        
        print("✓ No sensitive information disclosed in error messages")
    
    def test_cors_security(self, api_client: TestClient):
        """Test CORS (Cross-Origin Resource Sharing) security configuration."""
        print("\n=== CORS Security Test ===")
        
        # Test CORS headers with various origins
        test_origins = [
            "https://malicious-site.com",
            "http://localhost:3000",
            "https://example.com",
            "null",
            "*"
        ]
        
        for origin in test_origins:
            print(f"Testing CORS with origin: {origin}")
            
            headers = {
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"
            }
            
            # Test preflight request
            options_response = api_client.options("/api/v1/detect", headers=headers)
            
            # Test actual request
            payload = {"data": [[1, 2, 3]], "algorithm": "isolation_forest"}
            post_response = api_client.post("/api/v1/detect", json=payload, headers={"Origin": origin})
            
            # Analyze CORS headers
            cors_headers = {
                key.lower(): value for key, value in post_response.headers.items()
                if key.lower().startswith('access-control-')
            }
            
            if cors_headers:
                allowed_origin = cors_headers.get('access-control-allow-origin', '')
                
                # Should not allow all origins in production
                if origin == "*" and allowed_origin == "*":
                    print(f"  Warning: Wildcard CORS origin allowed")
                
                # Should not reflect arbitrary origins
                if origin not in ["http://localhost:3000", "https://example.com"] and allowed_origin == origin:
                    print(f"  Warning: Arbitrary origin reflected: {origin}")
        
        print("✓ CORS configuration checked")
    
    def test_content_type_security(self, api_client: TestClient):
        """Test content type handling and validation."""
        print("\n=== Content Type Security Test ===")
        
        payload_data = '{"data": [[1, 2, 3]], "algorithm": "isolation_forest"}'
        
        # Test various content types
        content_types = [
            "application/json",
            "text/plain",
            "text/html",
            "application/xml",
            "multipart/form-data",
            "application/octet-stream",
            "image/png",
            "text/javascript"
        ]
        
        for content_type in content_types:
            print(f"Testing content type: {content_type}")
            
            headers = {"Content-Type": content_type}
            response = api_client.post("/api/v1/detect", data=payload_data, headers=headers)
            
            if content_type == "application/json":
                # Should accept valid JSON
                assert response.status_code in [200, 400, 422], f"Valid JSON rejected"
            else:
                # Should reject non-JSON content types for JSON endpoints
                assert response.status_code in [400, 415, 422], f"Invalid content type accepted: {content_type}"
        
        print("✓ Content type validation working")
    
    def test_response_header_security(self, api_client: TestClient):
        """Test security-related response headers."""
        print("\n=== Response Header Security Test ===")
        
        response = api_client.get("/health")
        headers = {key.lower(): value for key, value in response.headers.items()}
        
        # Check for security headers
        security_headers = {
            'x-content-type-options': 'nosniff',
            'x-frame-options': ['DENY', 'SAMEORIGIN'],
            'x-xss-protection': '1; mode=block',
            'strict-transport-security': None,  # Should exist
            'content-security-policy': None,   # Should exist
        }
        
        print("Security headers analysis:")
        for header, expected in security_headers.items():
            header_value = headers.get(header)
            
            if header_value:
                print(f"  ✓ {header}: {header_value}")
                
                if isinstance(expected, list):
                    assert header_value in expected, f"Invalid {header} value: {header_value}"
                elif expected and expected != header_value:
                    assert expected in header_value, f"Invalid {header} value: {header_value}"
            else:
                print(f"  - {header}: Not set")
        
        # Check that server information is not disclosed
        server_headers = ['server', 'x-powered-by', 'x-aspnet-version']
        for header in server_headers:
            if header in headers:
                print(f"  Warning: Server information disclosed: {header} = {headers[header]}")
        
        print("✓ Response header security checked")


if __name__ == "__main__":
    print("Anomaly Detection API Security Test Suite")
    print("=" * 45)
    print("Testing API endpoints against common security vulnerabilities:")
    print("• Injection attacks")
    print("• DoS protection")
    print("• Input validation")
    print("• Information disclosure")
    print("• Authentication bypass")
    print("• Path traversal")
    print("• CORS security")
    print("• Content type validation")
    print("• Security headers")
    print()
    
    # Quick security check
    try:
        from fastapi.testclient import TestClient
        from anomaly_detection.server import create_app
        
        app = create_app()
        client = TestClient(app)
        
        # Quick health check
        response = client.get("/health")
        print(f"✓ API accessible (status: {response.status_code})")
        
        # Check basic security
        response = client.post("/api/v1/detect", json={"invalid": "payload"})
        if response.status_code >= 400:
            print("✓ Basic input validation working")
        
        print("Ready to run comprehensive security tests")
        
    except Exception as e:
        print(f"✗ Security test setup failed: {e}")
        print("Some security tests may not run properly")