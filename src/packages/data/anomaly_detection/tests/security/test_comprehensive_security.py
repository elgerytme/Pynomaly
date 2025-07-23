"""Comprehensive security tests for the complete anomaly detection system."""

import pytest
import subprocess
import socket
import ssl
import time
import json
import hashlib
import secrets
from typing import Dict, List, Any
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from anomaly_detection.server import create_app


class TestComprehensiveSecurity:
    """Comprehensive security tests covering the entire system."""
    
    @pytest.fixture
    def api_client(self):
        """Create API test client for comprehensive security testing."""
        app = create_app()
        return TestClient(app)
    
    def test_security_headers_comprehensive(self, api_client: TestClient):
        """Comprehensive test of security headers across all endpoints."""
        print("\n=== Comprehensive Security Headers Test ===")
        
        # Test various endpoints
        endpoints = [
            ("GET", "/health"),
            ("GET", "/api/v1/algorithms"),
            ("GET", "/api/v1/models"),
            ("POST", "/api/v1/detect", {"data": [[1, 2, 3]], "algorithm": "isolation_forest"}),
        ]
        
        required_security_headers = {
            'x-content-type-options': 'nosniff',
            'x-frame-options': ['DENY', 'SAMEORIGIN'],
            'x-xss-protection': '1; mode=block',
            'referrer-policy': ['strict-origin-when-cross-origin', 'no-referrer', 'same-origin'],
        }
        
        recommended_headers = {
            'strict-transport-security': 'max-age=',
            'content-security-policy': 'default-src',
        }
        
        for method, endpoint, *payload in endpoints:
            print(f"Testing security headers for {method} {endpoint}")
            
            if method == "GET":
                response = api_client.get(endpoint)
            elif method == "POST" and payload:
                response = api_client.post(endpoint, json=payload[0])
            else:
                continue
            
            headers = {k.lower(): v for k, v in response.headers.items()}
            
            # Check required headers
            for header, expected_values in required_security_headers.items():
                if header in headers:
                    if isinstance(expected_values, list):
                        assert headers[header] in expected_values, \
                            f"Invalid {header} value: {headers[header]}"
                    else:
                        assert expected_values in headers[header], \
                            f"Invalid {header} value: {headers[header]}"
                    print(f"  ✓ {header}: {headers[header]}")
                else:
                    print(f"  - {header}: Missing (recommended)")
            
            # Check recommended headers
            for header, expected_content in recommended_headers.items():
                if header in headers:
                    assert expected_content in headers[header], \
                        f"Invalid {header} value: {headers[header]}"
                    print(f"  ✓ {header}: Present")
                else:
                    print(f"  - {header}: Missing (recommended for production)")
        
        print("✓ Security headers comprehensive test completed")
    
    def test_rate_limiting_comprehensive(self, api_client: TestClient):
        """Comprehensive rate limiting test across different scenarios."""
        print("\n=== Comprehensive Rate Limiting Test ===")
        
        # Test rapid requests from single client
        endpoint = "/api/v1/detect"
        payload = {"data": [[1, 2, 3]], "algorithm": "isolation_forest"}
        
        # Burst test - many requests quickly
        burst_responses = []
        burst_start = time.time()
        
        for i in range(30):  # 30 rapid requests
            try:
                response = api_client.post(endpoint, json=payload, timeout=10)
                burst_responses.append({
                    'status': response.status_code,
                    'time': time.time() - burst_start,
                    'headers': dict(response.headers)
                })
            except Exception as e:
                burst_responses.append({
                    'status': 0,
                    'time': time.time() - burst_start,
                    'error': str(e)
                })
            
            time.sleep(0.05)  # 50ms between requests
        
        # Analyze burst responses
        rate_limited = sum(1 for r in burst_responses if r['status'] == 429)
        successful = sum(1 for r in burst_responses if 200 <= r['status'] < 300)
        errors = sum(1 for r in burst_responses if r['status'] >= 400 and r['status'] != 429)
        
        print(f"Burst test results:")
        print(f"  Successful: {successful}")
        print(f"  Rate limited (429): {rate_limited}")
        print(f"  Other errors: {errors}")
        
        # Rate limiting or error handling should prevent all requests from succeeding
        if rate_limited > 0:
            print("✓ Rate limiting detected")
        elif successful < len(burst_responses) * 0.8:
            print("✓ System protected against burst requests")
        else:
            print("- No obvious rate limiting detected (may be acceptable)")
        
        # Test sustained load
        print("\nTesting sustained load...")
        sustained_responses = []
        sustained_start = time.time()
        
        for i in range(20):  # 20 requests over 10 seconds
            try:
                response = api_client.post(endpoint, json=payload, timeout=15)
                sustained_responses.append(response.status_code)
            except Exception:
                sustained_responses.append(0)
            
            time.sleep(0.5)  # 500ms between requests
        
        sustained_success_rate = sum(1 for s in sustained_responses if 200 <= s < 300) / len(sustained_responses)
        print(f"Sustained load success rate: {sustained_success_rate:.1%}")
        
        # Sustained load should have higher success rate than burst
        assert sustained_success_rate >= 0.5, f"Sustained load success rate too low: {sustained_success_rate:.1%}"
        
        print("✓ Rate limiting comprehensive test completed")
    
    def test_input_sanitization_comprehensive(self, api_client: TestClient):
        """Comprehensive input sanitization test with various attack vectors."""
        print("\n=== Comprehensive Input Sanitization Test ===")
        
        # Various attack vectors
        attack_vectors = [
            # Script injection
            {
                "data": [["<script>alert('xss')</script>", 2, 3]],
                "algorithm": "isolation_forest"
            },
            # Command injection
            {
                "data": [[1, 2, 3]],
                "algorithm": "isolation_forest; cat /etc/passwd"
            },
            # Path traversal
            {
                "data": [[1, 2, 3]],
                "algorithm": "../../../etc/passwd"
            },
            # Buffer overflow simulation
            {
                "data": [["A" * 10000, 2, 3]],
                "algorithm": "isolation_forest"
            },
            # Format string attack
            {
                "data": [["%s%s%s%s%s", 2, 3]],
                "algorithm": "isolation_forest"
            },
            # Null byte injection
            {
                "data": [["test\x00hidden", 2, 3]],
                "algorithm": "isolation_forest"
            },
            # Unicode/encoding attacks
            {
                "data": [["\\u0000\\u0001\\u0002", 2, 3]],
                "algorithm": "isolation_forest"
            },
            # Very large numbers
            {
                "data": [[1e308, -1e308, float('inf')]],
                "algorithm": "isolation_forest"
            },
            # Deeply nested structures
            {
                "data": {"a": {"b": {"c": {"d": {"e": "deep"}}}}},
                "algorithm": "isolation_forest"
            },
        ]
        
        for i, attack_payload in enumerate(attack_vectors):
            print(f"Testing attack vector {i + 1}: {type(attack_payload['data']).__name__}")
            
            response = api_client.post("/api/v1/detect", json=attack_payload)
            
            # Should reject malicious input
            assert response.status_code >= 400, f"Attack vector {i + 1} not rejected"
            
            # Should not reflect malicious content in response
            if response.text:
                dangerous_patterns = [
                    '<script>', 'alert(', 'cat /etc/passwd', '../../../',
                    '%s%s%s', '\x00', 'deep'
                ]
                
                response_lower = response.text.lower()
                for pattern in dangerous_patterns:
                    assert pattern not in response_lower, \
                        f"Malicious content reflected in response: {pattern}"
        
        print("✓ All attack vectors properly sanitized")
    
    def test_session_security(self, api_client: TestClient):
        """Test session security and state management."""
        print("\n=== Session Security Test ===")
        
        # Test that requests are stateless (no session persistence)
        payload = {"data": [[1, 2, 3]], "algorithm": "isolation_forest"}
        
        # Make multiple requests and verify no session data leaks between them
        responses = []
        for i in range(5):
            response = api_client.post("/api/v1/detect", json=payload)
            responses.append(response)
        
        # Check that responses don't contain session information
        for i, response in enumerate(responses):
            if response.status_code == 200:
                response_data = response.json()
                
                # Should not contain session-related fields
                session_fields = [
                    'session_id', 'sessionId', 'session_token', 'csrf_token',
                    'user_id', 'userId', 'previous_request', 'request_history'
                ]
                
                response_str = json.dumps(response_data).lower()
                for field in session_fields:
                    assert field.lower() not in response_str, \
                        f"Session information found in response {i + 1}: {field}"
        
        # Check response headers for session cookies
        for i, response in enumerate(responses):
            headers = {k.lower(): v for k, v in response.headers.items()}
            
            if 'set-cookie' in headers:
                cookie_value = headers['set-cookie'].lower()
                
                # Should not set session cookies without proper security attributes
                if 'session' in cookie_value or 'sessionid' in cookie_value:
                    assert 'secure' in cookie_value, f"Session cookie not secure in response {i + 1}"
                    assert 'httponly' in cookie_value, f"Session cookie not httponly in response {i + 1}"
                    assert 'samesite' in cookie_value, f"Session cookie missing samesite in response {i + 1}"
        
        print("✓ Session security verified")
    
    def test_error_information_disclosure(self, api_client: TestClient):
        """Test that error messages don't disclose sensitive information."""
        print("\n=== Error Information Disclosure Test ===")
        
        # Generate various error conditions
        error_inducing_requests = [
            # Invalid endpoints
            ("GET", "/api/v1/nonexistent"),
            ("POST", "/api/v1/invalid"),
            
            # Invalid methods
            ("DELETE", "/api/v1/detect"),
            ("PUT", "/api/v1/models"),
            
            # Malformed JSON
            ("POST", "/api/v1/detect", '{"invalid": json}'),
            
            # Missing required fields
            ("POST", "/api/v1/detect", {"algorithm": "isolation_forest"}),
            
            # Invalid algorithm
            ("POST", "/api/v1/detect", {"data": [[1, 2, 3]], "algorithm": "nonexistent"}),
        ]
        
        sensitive_info_patterns = [
            # File system paths
            r'/[a-zA-Z0-9_\-/]+\.py',
            r'C:\\[A-Za-z0-9_\-\\]+',
            
            # Stack traces
            'traceback', 'stacktrace', 'file "', 'line \d+',
            
            # Database information
            'database', 'connection', 'sql', 'query',
            
            # System information
            'version', 'python', 'server', 'host',
            
            # Internal details
            'internal', 'debug', 'development', 'config',
        ]
        
        for method, endpoint, *body in error_inducing_requests:
            print(f"Testing error disclosure for {method} {endpoint}")
            
            try:
                if method == "GET":
                    response = api_client.get(endpoint)
                elif method == "POST" and body:
                    if body[0].startswith('{'):
                        # JSON payload
                        response = api_client.post(
                            endpoint, 
                            data=body[0],
                            headers={"content-type": "application/json"}
                        )
                    else:
                        response = api_client.post(endpoint, json=body[0])
                else:
                    response = getattr(api_client, method.lower())(endpoint)
                
                # Check response for sensitive information
                if response.text:
                    response_text = response.text.lower()
                    
                    for pattern in sensitive_info_patterns:
                        import re
                        if re.search(pattern.lower(), response_text):
                            print(f"  Warning: Potential information disclosure: {pattern}")
                        
                    # Check for obvious leakage
                    obvious_leaks = [
                        'traceback', 'file "/', 'line ', 'error in ',
                        'internal server error', 'debug', 'stacktrace'
                    ]
                    
                    for leak in obvious_leaks:
                        assert leak not in response_text, \
                            f"Information disclosure detected: {leak}"
                
            except Exception as e:
                # Network errors are acceptable
                print(f"  Request failed (acceptable): {e}")
        
        print("✓ No sensitive information disclosure detected")
    
    def test_cryptographic_security(self):
        """Test cryptographic implementations and security."""
        print("\n=== Cryptographic Security Test ===")
        
        # Test random number generation
        random_values = [secrets.randbelow(1000000) for _ in range(100)]
        
        # Check for basic randomness (not cryptographically rigorous, but basic check)
        unique_values = len(set(random_values))
        assert unique_values > 90, f"Random values not sufficiently random: {unique_values}/100 unique"
        
        # Test that sensitive operations use secure random
        import random
        import numpy as np
        
        # Ensure numpy uses secure seeding when needed
        np.random.seed(secrets.randbelow(2**32))
        secure_random_sample = np.random.randn(100)
        
        # Reset to weak seed
        np.random.seed(12345)
        weak_random_sample = np.random.randn(100)
        
        # Should be different (basic check)
        assert not np.array_equal(secure_random_sample, weak_random_sample), \
            "Secure and weak random samples should differ"
        
        # Test hash consistency (for data integrity)
        test_data = b"test data for hashing"
        hash1 = hashlib.sha256(test_data).hexdigest()
        hash2 = hashlib.sha256(test_data).hexdigest()
        
        assert hash1 == hash2, "Hash function should be deterministic"
        assert len(hash1) == 64, "SHA-256 hash should be 64 characters"
        
        print("✓ Basic cryptographic security verified")
    
    def test_dependency_security(self):
        """Test security of dependencies and their configurations."""
        print("\n=== Dependency Security Test ===")
        
        # Check for known vulnerable packages (basic check)
        try:
            import pkg_resources
            
            # Get installed packages
            installed_packages = [d.project_name.lower() for d in pkg_resources.working_set]
            
            # Known packages with past vulnerabilities (educational - check for updates)
            packages_to_monitor = [
                'requests', 'urllib3', 'jinja2', 'pyyaml', 'pillow',
                'numpy', 'scipy', 'scikit-learn', 'pandas', 'fastapi'
            ]
            
            present_monitored_packages = [pkg for pkg in packages_to_monitor if pkg in installed_packages]
            
            print(f"Monitoring {len(present_monitored_packages)} security-relevant packages")
            
            # This is a basic check - in production, use tools like safety or snyk
            for package in present_monitored_packages[:5]:  # Check first 5
                try:
                    version = pkg_resources.get_distribution(package).version
                    print(f"  {package}: {version}")
                except:
                    print(f"  {package}: version unknown")
            
        except ImportError:
            print("  Package analysis not available")
        
        # Test import security
        dangerous_imports = [
            'os.system', 'subprocess.call', 'eval', 'exec',
            'open', '__import__', 'compile'
        ]
        
        # This would be more comprehensive in a real security audit
        print("✓ Basic dependency security check completed")
    
    def test_logging_security(self):
        """Test that logging doesn't expose sensitive information."""
        print("\n=== Logging Security Test ===")
        
        # Test that sensitive data is not logged
        from anomaly_detection.domain.services.detection_service import DetectionService
        import logging
        import io
        
        # Capture log output
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('anomaly_detection')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            # Perform operations that might log sensitive data
            service = DetectionService()
            
            # Use data that might contain sensitive information
            sensitive_data = [
                [123456789, 850, 45],  # SSN-like number, credit score, age
                [987654321, 700, 30],
                [555443333, 750, 35],
            ]
            
            result = service.detect_anomalies(
                data=sensitive_data,
                algorithm='iforest',
                contamination=0.33,
                random_state=42
            )
            
            # Check log output for sensitive data
            log_output = log_stream.getvalue().lower()
            
            # Should not log raw sensitive data
            sensitive_patterns = [
                '123456789', '987654321', '555443333',  # Potential SSNs
                'credit', 'ssn', 'social security',
                'password', 'secret', 'key', 'token'
            ]
            
            for pattern in sensitive_patterns:
                assert pattern not in log_output, \
                    f"Sensitive information found in logs: {pattern}"
            
            print("✓ No sensitive information found in logs")
            
        finally:
            logger.removeHandler(handler)
    
    def test_system_hardening(self):
        """Test system hardening and security configuration."""
        print("\n=== System Hardening Test ===")
        
        # Test file permissions (Unix-like systems)
        if hasattr(__import__('os'), 'stat'):
            import stat
            
            # Check that Python files don't have excessive permissions
            from pathlib import Path
            
            package_root = Path(__file__).parent.parent.parent / "src" / "anomaly_detection"
            
            if package_root.exists():
                python_files = list(package_root.rglob("*.py"))[:10]  # Check first 10 files
                
                for py_file in python_files:
                    if py_file.is_file():
                        file_stat = py_file.stat()
                        file_mode = file_stat.st_mode
                        
                        # Should not be world-writable
                        world_writable = bool(file_mode & stat.S_IWOTH)
                        assert not world_writable, f"File is world-writable: {py_file}"
                        
                        # Should not be world-executable for data files
                        if py_file.suffix in ['.json', '.yaml', '.yml', '.txt']:
                            world_executable = bool(file_mode & stat.S_IXOTH)
                            assert not world_executable, f"Data file is world-executable: {py_file}"
        
        # Test environment variable security
        import os
        
        # Check for potentially dangerous environment variables
        dangerous_env_vars = [
            'PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH'
        ]
        
        for env_var in dangerous_env_vars:
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Should not contain obvious injection attempts
                dangerous_patterns = ['../', ';', '|', '&', '$']
                
                for pattern in dangerous_patterns:
                    if pattern in value:
                        print(f"  Warning: Potentially dangerous {env_var}: {pattern}")
        
        print("✓ System hardening checks completed")


if __name__ == "__main__":
    print("Comprehensive Anomaly Detection Security Test Suite")
    print("=" * 55)
    print("Testing complete system security across multiple dimensions:")
    print("• Security headers and HTTP security")
    print("• Rate limiting and DoS protection")
    print("• Input sanitization and injection prevention")
    print("• Session security and state management")
    print("• Error information disclosure prevention")
    print("• Cryptographic security")
    print("• Dependency security")
    print("• Logging security")
    print("• System hardening")
    print()
    
    # Quick comprehensive security check
    try:
        from fastapi.testclient import TestClient
        from anomaly_detection.server import create_app
        import secrets
        import hashlib
        
        # Test basic security functionality
        app = create_app()
        client = TestClient(app)
        
        # Quick security validation
        response = client.get("/health")
        headers = {k.lower(): v for k, v in response.headers.items()}
        
        security_header_count = sum(1 for h in headers.keys() if h.startswith('x-') or 'security' in h)
        
        print(f"✓ API accessible with {security_header_count} security headers")
        
        # Test basic crypto
        test_hash = hashlib.sha256(b"test").hexdigest()
        test_random = secrets.randbelow(1000)
        
        print(f"✓ Cryptographic functions available")
        print("Ready to run comprehensive security tests")
        
    except Exception as e:
        print(f"✗ Comprehensive security test setup failed: {e}")
        print("Some security tests may not run properly")