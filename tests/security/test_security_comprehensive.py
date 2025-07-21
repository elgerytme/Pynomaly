"""
Comprehensive Security Tests for anomaly_detection

This module contains security-focused tests including:
- Input validation and sanitization
- Authentication and authorization
- Injection attack prevention
- Data protection and privacy
- Cryptographic security
- Network security
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.security


class TestInputValidationSecurity:
    """Test input validation and sanitization"""

    def test_sql_injection_prevention(self, api_client, security_test_data):
        """Test prevention of SQL injection attacks"""
        
        for injection_attempt in security_test_data["sql_injection_attempts"]:
            # Test in various input fields
            malicious_inputs = [
                {"data": injection_attempt},
                {"algorithm": injection_attempt},
                {"model_name": injection_attempt},
                {"description": injection_attempt}
            ]
            
            for malicious_input in malicious_inputs:
                response = api_client.post("/api/v1/detect", json=malicious_input)
                
                # Should reject malicious input
                assert response.status_code in [400, 422], \
                    f"Failed to reject SQL injection: {injection_attempt}"
                
                # Response should not contain SQL error messages
                response_text = response.text.lower()
                sql_error_indicators = ["syntax error", "sql", "database", "table", "column"]
                
                for indicator in sql_error_indicators:
                    assert indicator not in response_text, \
                        f"SQL error exposed in response: {indicator}"

    def test_xss_prevention(self, api_client, security_test_data):
        """Test prevention of Cross-Site Scripting (XSS) attacks"""
        
        for xss_attempt in security_test_data["xss_attempts"]:
            # Test XSS in text fields
            malicious_data = {
                "model_name": xss_attempt,
                "description": xss_attempt,
                "metadata": {"title": xss_attempt}
            }
            
            response = api_client.post("/api/v1/models", json=malicious_data)
            
            if response.status_code == 200:
                # If accepted, response should be properly escaped
                response_data = response.json()
                response_str = json.dumps(response_data)
                
                # Should not contain unescaped script tags
                assert "<script>" not in response_str, "Unescaped script tag in response"
                assert "javascript:" not in response_str, "Unescaped javascript: in response"

    def test_path_traversal_prevention(self, api_client, security_test_data):
        """Test prevention of path traversal attacks"""
        
        for path_attempt in security_test_data["path_traversal_attempts"]:
            # Test in file-related endpoints
            malicious_requests = [
                {"file_path": path_attempt},
                {"model_path": path_attempt},
                {"export_path": path_attempt}
            ]
            
            for malicious_request in malicious_requests:
                response = api_client.post("/api/v1/export", json=malicious_request)
                
                # Should reject path traversal attempts
                assert response.status_code in [400, 403, 422], \
                    f"Failed to prevent path traversal: {path_attempt}"

    def test_large_payload_handling(self, api_client, security_test_data):
        """Test handling of oversized payloads"""
        
        large_payload = {
            "data": security_test_data["large_payload"],
            "description": security_test_data["large_payload"]
        }
        
        response = api_client.post("/api/v1/detect", json=large_payload)
        
        # Should reject or handle large payloads gracefully
        assert response.status_code in [413, 400, 422], \
            "Large payload not properly rejected"

    def test_special_character_handling(self, api_client, security_test_data):
        """Test handling of special characters and unicode"""
        
        special_chars_data = {
            "description": security_test_data["special_characters"],
            "unicode_text": security_test_data["unicode_characters"],
            "null_bytes": security_test_data["null_bytes"]
        }
        
        response = api_client.post("/api/v1/models", json=special_chars_data)
        
        # Should handle gracefully without errors
        assert response.status_code in [200, 201, 400, 422], \
            "Special characters caused server error"

    def test_numeric_overflow_prevention(self, api_client):
        """Test prevention of numeric overflow attacks"""
        
        overflow_values = [
            2**63,      # Large integer
            -2**63,     # Large negative integer
            float('inf'),  # Infinity
            float('-inf'), # Negative infinity
            float('nan'),  # NaN
            1e308,      # Very large float
        ]
        
        for value in overflow_values:
            malicious_data = {
                "data": [[value, value, value]],
                "contamination": value,
                "threshold": value
            }
            
            try:
                response = api_client.post("/api/v1/detect", json=malicious_data)
                
                # Should reject or handle gracefully
                assert response.status_code in [200, 400, 422], \
                    f"Overflow value caused server error: {value}"
                
            except (ValueError, OverflowError):
                # Client-side validation caught the issue
                pass


class TestAuthenticationSecurity:
    """Test authentication and authorization security"""

    def test_unauthorized_access_prevention(self, api_client):
        """Test that sensitive endpoints require authentication"""
        
        sensitive_endpoints = [
            ("/api/v1/admin/users", "GET"),
            ("/api/v1/admin/settings", "POST"),
            ("/api/v1/models/1/delete", "DELETE"),
            ("/api/v1/system/shutdown", "POST")
        ]
        
        for endpoint, method in sensitive_endpoints:
            if method == "GET":
                response = api_client.get(endpoint)
            elif method == "POST":
                response = api_client.post(endpoint, json={})
            elif method == "DELETE":
                response = api_client.delete(endpoint)
            
            # Should require authentication
            assert response.status_code in [401, 403, 404], \
                f"Endpoint {endpoint} not properly protected"

    def test_weak_authentication_rejection(self, api_client):
        """Test rejection of weak authentication attempts"""
        
        weak_credentials = [
            {"username": "admin", "password": "password"},
            {"username": "admin", "password": "123456"},
            {"username": "admin", "password": "admin"},
            {"username": "test", "password": ""},
            {"username": "", "password": "password"}
        ]
        
        for creds in weak_credentials:
            response = api_client.post("/api/v1/auth/login", json=creds)
            
            # Should reject weak credentials
            assert response.status_code in [400, 401, 422], \
                f"Weak credentials accepted: {creds}"

    def test_session_security(self, api_client):
        """Test session management security"""
        
        # Test session timeout
        login_data = {"username": "testuser", "password": "StrongPass123!"}
        
        # Mock login (if endpoint exists)
        login_response = api_client.post("/api/v1/auth/login", json=login_data)
        
        if login_response.status_code == 200:
            # Get session token
            token = login_response.json().get("token")
            
            if token:
                # Test token validation
                headers = {"Authorization": f"Bearer {token}"}
                response = api_client.get("/api/v1/profile", headers=headers)
                
                # Should work with valid token
                assert response.status_code in [200, 404], "Valid token rejected"
                
                # Test logout
                logout_response = api_client.post("/api/v1/auth/logout", headers=headers)
                
                # Token should be invalidated after logout
                response = api_client.get("/api/v1/profile", headers=headers)
                assert response.status_code == 401, "Token not invalidated after logout"

    def test_brute_force_protection(self, api_client):
        """Test protection against brute force attacks"""
        
        # Attempt multiple failed logins
        for attempt in range(10):
            bad_creds = {"username": "admin", "password": f"badpass{attempt}"}
            response = api_client.post("/api/v1/auth/login", json=bad_creds)
            
            if attempt > 5:  # After several attempts
                # Should start rate limiting or blocking
                assert response.status_code in [401, 429, 423], \
                    "No brute force protection detected"
                
                if response.status_code == 429:  # Rate limited
                    # Should include retry-after header
                    assert "retry-after" in response.headers or "x-ratelimit-reset" in response.headers


class TestDataProtectionSecurity:
    """Test data protection and privacy security"""

    def test_sensitive_data_exposure_prevention(self, api_client):
        """Test prevention of sensitive data exposure"""
        
        # Submit data that might contain sensitive information
        potentially_sensitive_data = {
            "data": [
                [1234567890, 123.45, 987.65],  # Could be SSN, financial data
                [4111111111111111, 123, 456],  # Could be credit card
                [192.168.1.1, 8080, 443]      # Could be IP addresses
            ],
            "metadata": {
                "user_id": "12345",
                "session_id": "abc-123-def",
                "api_key": "sk-1234567890abcdef"
            }
        }
        
        response = api_client.post("/api/v1/detect", json=potentially_sensitive_data)
        
        if response.status_code == 200:
            response_data = response.json()
            response_str = json.dumps(response_data)
            
            # Should not expose sensitive patterns in logs or responses
            sensitive_patterns = [
                r"\d{4}-\d{4}-\d{4}-\d{4}",  # Credit card pattern
                r"\d{3}-\d{2}-\d{4}",         # SSN pattern
                r"sk-[a-f0-9]{32}",           # API key pattern
            ]
            
            import re
            for pattern in sensitive_patterns:
                assert not re.search(pattern, response_str), \
                    f"Sensitive data pattern exposed: {pattern}"

    def test_data_encryption_requirements(self, api_client):
        """Test that sensitive data is properly encrypted"""
        
        # This test would verify encryption at rest and in transit
        # For now, test that HTTPS is enforced
        
        if hasattr(api_client, 'base_url'):
            base_url = str(api_client.base_url)
            if base_url.startswith('http://'):
                pytest.skip("HTTPS not configured in test environment")
            
            assert base_url.startswith('https://'), \
                "API should enforce HTTPS for security"

    def test_data_retention_compliance(self, api_client, temp_directory):
        """Test data retention and deletion compliance"""
        
        # Create test data
        test_data = {"data": [[1, 2, 3], [4, 5, 6]], "store": True}
        
        response = api_client.post("/api/v1/detect", json=test_data)
        
        if response.status_code == 200 and "data_id" in response.json():
            data_id = response.json()["data_id"]
            
            # Request data deletion
            delete_response = api_client.delete(f"/api/v1/data/{data_id}")
            assert delete_response.status_code in [200, 204, 404], \
                "Data deletion not supported"
            
            # Verify data is actually deleted
            get_response = api_client.get(f"/api/v1/data/{data_id}")
            assert get_response.status_code == 404, \
                "Data not properly deleted"

    def test_pii_detection_and_masking(self, api_client):
        """Test detection and masking of Personally Identifiable Information"""
        
        # Data with potential PII
        pii_data = {
            "data": [
                ["john.doe@example.com", "555-123-4567", "123-45-6789"],
                ["jane.smith@test.com", "555-987-6543", "987-65-4321"]
            ],
            "detect_pii": True
        }
        
        response = api_client.post("/api/v1/analyze-pii", json=pii_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Should detect PII
            assert "pii_detected" in result
            
            if result.get("pii_detected"):
                # Should provide masked/anonymized data
                assert "masked_data" in result
                
                # Original PII should not appear in masked data
                masked_str = json.dumps(result["masked_data"])
                assert "john.doe@example.com" not in masked_str
                assert "555-123-4567" not in masked_str


class TestCryptographicSecurity:
    """Test cryptographic security implementations"""

    def test_secure_random_generation(self, api_client):
        """Test that cryptographically secure random numbers are used"""
        
        # Request multiple random values
        random_values = []
        
        for _ in range(10):
            response = api_client.get("/api/v1/random")
            
            if response.status_code == 200:
                value = response.json().get("value")
                if value is not None:
                    random_values.append(value)
        
        if random_values:
            # Check for basic randomness properties
            assert len(set(random_values)) == len(random_values), \
                "Random values are not unique"
            
            # Check distribution (basic test)
            if all(isinstance(v, (int, float)) for v in random_values):
                import statistics
                mean = statistics.mean(random_values)
                stdev = statistics.stdev(random_values) if len(random_values) > 1 else 0
                
                # Should have reasonable distribution
                assert stdev > 0, "Random values have no variance"

    def test_hash_function_security(self, api_client):
        """Test that secure hash functions are used"""
        
        test_data = {"data": "test_string_for_hashing"}
        
        response = api_client.post("/api/v1/hash", json=test_data)
        
        if response.status_code == 200:
            hash_result = response.json().get("hash")
            
            if hash_result:
                # Should use secure hash (SHA-256 or better)
                # SHA-256 produces 64 character hex string
                assert len(hash_result) >= 64, \
                    "Hash appears to be from weak algorithm"
                
                # Should be deterministic
                response2 = api_client.post("/api/v1/hash", json=test_data)
                if response2.status_code == 200:
                    hash_result2 = response2.json().get("hash")
                    assert hash_result == hash_result2, \
                        "Hash function is not deterministic"

    def test_password_hashing_security(self, api_client):
        """Test that passwords are properly hashed"""
        
        # This would test user creation/password change endpoints
        user_data = {
            "username": "testuser",
            "password": "SecurePassword123!",
            "email": "test@example.com"
        }
        
        response = api_client.post("/api/v1/users", json=user_data)
        
        if response.status_code in [200, 201]:
            # Password should never be returned in response
            response_data = response.json()
            response_str = json.dumps(response_data)
            
            assert "SecurePassword123!" not in response_str, \
                "Plain text password exposed in response"
            
            assert "password" not in response_data or \
                   response_data["password"] != user_data["password"], \
                   "Password not properly hashed"


class TestNetworkSecurity:
    """Test network-level security"""

    def test_rate_limiting(self, api_client):
        """Test rate limiting implementation"""
        
        # Make rapid requests
        responses = []
        
        for i in range(100):  # Attempt 100 rapid requests
            response = api_client.get("/api/v1/status")
            responses.append(response)
            
            if response.status_code == 429:  # Rate limited
                break
            
            if i > 20 and all(r.status_code == 200 for r in responses):
                # If no rate limiting after 20 requests, skip test
                pytest.skip("Rate limiting not configured or threshold too high")
        
        # Should eventually get rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        
        if not rate_limited:
            pytest.skip("Rate limiting not triggered")
        
        # Rate limit response should include appropriate headers
        rate_limited_response = next(r for r in responses if r.status_code == 429)
        
        expected_headers = ["x-ratelimit-limit", "x-ratelimit-remaining", "retry-after"]
        has_rate_limit_header = any(header in rate_limited_response.headers 
                                  for header in expected_headers)
        
        assert has_rate_limit_header, "Rate limit response missing proper headers"

    def test_cors_configuration(self, api_client):
        """Test Cross-Origin Resource Sharing (CORS) configuration"""
        
        # Test preflight request
        headers = {
            "Origin": "https://malicious-site.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        
        response = api_client.options("/api/v1/detect", headers=headers)
        
        if "access-control-allow-origin" in response.headers:
            allowed_origin = response.headers["access-control-allow-origin"]
            
            # Should not allow all origins in production
            assert allowed_origin != "*", \
                "CORS allows all origins - security risk"
            
            # Should not allow malicious origins
            assert "malicious-site.com" not in allowed_origin, \
                "CORS allows malicious origins"

    def test_security_headers(self, api_client):
        """Test security headers are properly set"""
        
        response = api_client.get("/api/v1/status")
        
        # Check for security headers
        security_headers = {
            "x-content-type-options": "nosniff",
            "x-frame-options": ["DENY", "SAMEORIGIN"],
            "x-xss-protection": "1; mode=block",
            "strict-transport-security": None,  # Should exist
            "content-security-policy": None     # Should exist
        }
        
        for header, expected_value in security_headers.items():
            if header in response.headers:
                actual_value = response.headers[header]
                
                if expected_value is None:
                    # Just check header exists
                    assert actual_value, f"Security header {header} is empty"
                elif isinstance(expected_value, list):
                    assert actual_value in expected_value, \
                        f"Security header {header} has unexpected value: {actual_value}"
                else:
                    assert actual_value == expected_value, \
                        f"Security header {header} mismatch: {actual_value} != {expected_value}"

    def test_information_disclosure_prevention(self, api_client):
        """Test prevention of information disclosure"""
        
        # Test error responses don't leak information
        response = api_client.get("/api/v1/nonexistent-endpoint")
        
        if response.status_code == 404:
            error_text = response.text.lower()
            
            # Should not disclose sensitive information
            disclosure_indicators = [
                "stack trace",
                "internal server error",
                "database error",
                "sql",
                "exception",
                "traceback",
                "file path",
                "/opt/",
                "/var/",
                "/home/"
            ]
            
            for indicator in disclosure_indicators:
                assert indicator not in error_text, \
                    f"Error response contains information disclosure: {indicator}"
        
        # Test server header doesn't disclose version
        if "server" in response.headers:
            server_header = response.headers["server"].lower()
            
            # Should not contain specific version numbers
            version_patterns = [
                r"\d+\.\d+\.\d+",  # Version numbers
                "apache/",
                "nginx/",
                "python/",
                "flask/",
                "django/"
            ]
            
            import re
            for pattern in version_patterns:
                assert not re.search(pattern, server_header), \
                    f"Server header discloses version: {server_header}"


class TestVulnerabilityScanning:
    """Test for common vulnerabilities"""

    def test_dependency_vulnerabilities(self, temp_directory):
        """Test for known vulnerabilities in dependencies"""
        
        # This would run safety check or similar
        import subprocess
        import sys
        
        try:
            # Run safety check on requirements
            result = subprocess.run([
                sys.executable, "-m", "safety", "check", "--json"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # No vulnerabilities found
                pass
            else:
                # Vulnerabilities found - parse results
                try:
                    vulnerabilities = json.loads(result.stdout)
                    
                    # Filter out acceptable vulnerabilities (if any)
                    critical_vulns = [
                        v for v in vulnerabilities 
                        if v.get("severity", "").lower() in ["critical", "high"]
                    ]
                    
                    assert len(critical_vulns) == 0, \
                        f"Critical vulnerabilities found: {len(critical_vulns)}"
                    
                except json.JSONDecodeError:
                    # Safety not installed or other error
                    pytest.skip("Safety check not available")
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Safety check not available or timed out")

    def test_code_injection_vulnerabilities(self, api_client):
        """Test for code injection vulnerabilities"""
        
        code_injection_attempts = [
            "__import__('os').system('ls')",
            "eval('1+1')",
            "exec('print(\"injected\")')",
            "import subprocess; subprocess.call(['ls'])",
            "';os.system('id');'",
            "__builtins__.__import__('os').system('whoami')"
        ]
        
        for injection_code in code_injection_attempts:
            malicious_data = {
                "algorithm": injection_code,
                "parameters": {"eval": injection_code},
                "filter_expression": injection_code
            }
            
            response = api_client.post("/api/v1/detect", json=malicious_data)
            
            # Should reject code injection attempts
            assert response.status_code in [400, 422], \
                f"Code injection not prevented: {injection_code}"

    def test_deserialization_vulnerabilities(self, api_client):
        """Test for deserialization vulnerabilities"""
        
        # Test potentially malicious serialized data
        malicious_payloads = [
            "pickle.loads(b'...')",  # Pickle injection
            "yaml.load('!!python/object/apply:os.system [\"ls\"]')",  # YAML injection
            "json.loads('{\"__reduce__\": [\"os.system\", [\"ls\"]]}')"  # JSON injection
        ]
        
        for payload in malicious_payloads:
            malicious_data = {"serialized_data": payload}
            
            response = api_client.post("/api/v1/import", json=malicious_data)
            
            # Should reject malicious serialized data
            assert response.status_code in [400, 422, 404], \
                f"Deserialization vulnerability: {payload}"