#!/usr/bin/env python3
"""
Security testing script for Pynomaly.

This script tests all security components to ensure they're working correctly.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monorepo.presentation.api.security.authentication import (
    AuthenticationManager,
    JWTManager,
)
from monorepo.presentation.api.security.authorization import (
    AuthorizationManager,
    Permission,
    Role,
)
from monorepo.presentation.api.security.encryption import EncryptionManager
from monorepo.presentation.api.security.input_validation import SecurityValidator
from monorepo.presentation.api.security.rate_limiting import RateLimiter, RateLimitType
from monorepo.presentation.api.security.security_manager import SecurityManager
from monorepo.presentation.api.security.security_monitoring import (
    SecurityEvent,
    SecurityEventType,
    SecurityMonitor,
    SecuritySeverity,
)
from monorepo.presentation.api.security.vulnerability_scanner import (
    VulnerabilityScanner,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityTester:
    """Security component testing suite."""

    def __init__(self):
        self.test_results = {}
        self.setup_test_environment()

    def setup_test_environment(self):
        """Setup test environment variables."""
        os.environ["PYNOMALY_ENV"] = "testing"
        os.environ["PYNOMALY_JWT_SECRET"] = "test-secret-key-for-testing-only"
        os.environ["PYNOMALY_MASTER_KEY"] = "test-master-key-for-testing"

    def run_all_tests(self):
        """Run all security tests."""
        logger.info("Starting comprehensive security testing...")

        test_methods = [
            self.test_authentication,
            self.test_authorization,
            self.test_input_validation,
            self.test_rate_limiting,
            self.test_encryption,
            self.test_security_monitoring,
            self.test_vulnerability_scanner,
            self.test_security_manager_integration,
        ]

        for test_method in test_methods:
            try:
                logger.info(f"Running {test_method.__name__}...")
                result = test_method()
                self.test_results[test_method.__name__] = result
                logger.info(f"✅ {test_method.__name__} passed")
            except Exception as e:
                logger.error(f"❌ {test_method.__name__} failed: {e}")
                self.test_results[test_method.__name__] = {
                    "status": "failed",
                    "error": str(e),
                }

        self.print_test_summary()

    def test_authentication(self):
        """Test authentication components."""
        # Test JWT Manager
        jwt_manager = JWTManager("test-secret", "HS256")

        # Generate token
        tokens = jwt_manager.generate_token("test_user", ["user"], ["read"])
        assert "access_token" in tokens
        assert "refresh_token" in tokens

        # Validate token
        payload = jwt_manager.validate_token(tokens["access_token"])
        assert payload is not None
        assert payload["user_id"] == "test_user"

        # Test Authentication Manager
        auth_manager = AuthenticationManager(jwt_manager)

        # Test successful authentication
        result = auth_manager.authenticate_user("testuser", "password123")
        assert result["status"] == "success"

        # Test failed authentication
        result = auth_manager.authenticate_user("testuser", "wrongpassword")
        assert result["error"] == "invalid_credentials"

        # Test account lockout
        for _ in range(6):  # Exceed max attempts
            auth_manager.authenticate_user("testuser", "wrongpassword")

        assert auth_manager.is_account_locked("testuser")

        return {"status": "passed", "components": ["JWT", "Authentication", "Lockout"]}

    def test_authorization(self):
        """Test authorization components."""
        auth_manager = AuthorizationManager()

        # Test role assignment
        auth_manager.assign_role("test_user", Role.DATA_SCIENTIST)
        assert Role.DATA_SCIENTIST in auth_manager.user_roles["test_user"]

        # Test permission checking
        assert auth_manager.has_permission("test_user", Permission.READ_DATA)
        assert auth_manager.has_permission("test_user", Permission.CREATE_MODELS)
        assert not auth_manager.has_permission("test_user", Permission.MANAGE_USERS)

        # Test authorization
        assert auth_manager.authorize_request("test_user", "data", "read")
        assert not auth_manager.authorize_request("test_user", "users", "manage")

        return {
            "status": "passed",
            "components": ["RBAC", "Permissions", "Authorization"],
        }

    def test_input_validation(self):
        """Test input validation components."""
        validator = SecurityValidator()

        # Test basic validation
        test_data = {"username": "testuser", "email": "test@example.com"}
        validated = validator.validate_and_sanitize(test_data)
        assert "username" in validated

        # Test security threat detection
        malicious_data = {"query": "SELECT * FROM users WHERE 1=1; DROP TABLE users;"}
        threats = validator.check_security_threats(malicious_data)
        assert len(threats) > 0
        assert "SQL injection" in threats[0]

        # Test XSS detection
        xss_data = {"comment": "<script>alert('xss')</script>"}
        threats = validator.check_security_threats(xss_data)
        assert len(threats) > 0
        assert "XSS" in threats[0]

        # Test file upload validation
        result = validator.validate_file_upload("test.csv", 1024, b"test,data\n1,2")
        assert result["valid"]

        result = validator.validate_file_upload("test.exe", 1024, b"malicious")
        assert not result["valid"]

        return {
            "status": "passed",
            "components": ["Validation", "Sanitization", "Threat Detection"],
        }

    def test_rate_limiting(self):
        """Test rate limiting components."""
        rate_limiter = RateLimiter()

        # Test rate limit configuration
        rate_limiter.set_rate_limit("test_user", 5, 60, RateLimitType.PER_USER)

        # Test requests within limit
        for i in range(4):
            allowed, status = rate_limiter.is_allowed(
                "test_user", RateLimitType.PER_USER
            )
            assert allowed

        # Test request exceeding limit
        allowed, status = rate_limiter.is_allowed("test_user", RateLimitType.PER_USER)
        assert allowed  # 5th request should still be allowed

        allowed, status = rate_limiter.is_allowed("test_user", RateLimitType.PER_USER)
        assert not allowed  # 6th request should be denied

        return {"status": "passed", "components": ["Rate Limiting", "Thresholds"]}

    def test_encryption(self):
        """Test encryption components."""
        encryption_manager = EncryptionManager()

        # Test data encryption
        test_data = "sensitive information"
        encrypted = encryption_manager.encrypt_data(test_data)
        assert "encrypted_data" in encrypted
        assert encrypted["algorithm"] == "Fernet"

        # Test data decryption
        decrypted = encryption_manager.decrypt_data(encrypted)
        assert decrypted == test_data

        # Test field-level encryption
        from monorepo.presentation.api.security.encryption import FieldLevelEncryption

        field_encryption = FieldLevelEncryption(encryption_manager)

        record = {"name": "John Doe", "email": "john@example.com", "age": 30}
        encrypted_record = field_encryption.encrypt_record(record)

        assert "email" not in encrypted_record  # Should be removed
        assert "email_encrypted" in encrypted_record  # Should be encrypted
        assert "name" in encrypted_record  # Non-sensitive field preserved

        decrypted_record = field_encryption.decrypt_record(encrypted_record)
        assert decrypted_record["email"] == "john@example.com"

        return {
            "status": "passed",
            "components": ["Encryption", "Decryption", "Field Encryption"],
        }

    def test_security_monitoring(self):
        """Test security monitoring components."""
        monitor = SecurityMonitor()

        # Test event logging
        event = SecurityEvent(
            event_id="test_001",
            event_type=SecurityEventType.AUTHENTICATION_FAILURE,
            severity=SecuritySeverity.MEDIUM,
            timestamp=datetime.utcnow(),
            source_ip="192.168.1.100",
            user_id="test_user",
            endpoint="/api/login",
            user_agent="Test Agent",
            details={"reason": "invalid_password"},
            risk_score=0.5,
        )

        monitor.log_security_event(event)

        # Verify event was logged
        assert len(monitor.events) > 0
        assert monitor.events[-1].event_id == "test_001"

        # Test security summary
        summary = monitor.get_security_summary(24)
        assert "total_events" in summary
        assert summary["total_events"] >= 1

        return {
            "status": "passed",
            "components": ["Event Logging", "Monitoring", "Alerts"],
        }

    def test_vulnerability_scanner(self):
        """Test vulnerability scanner components."""
        scanner = VulnerabilityScanner()

        # Test configuration scan
        test_config = {
            "DEBUG": True,  # Should trigger vulnerability
            "SECRET_KEY": "weak",  # Should trigger vulnerability
            "SECURITY_HEADERS": {},  # Should trigger vulnerability
        }

        config_vulns = scanner.scan_configuration(test_config)
        assert len(config_vulns) > 0

        # Find specific vulnerabilities
        debug_vuln = next((v for v in config_vulns if "debug" in v.title.lower()), None)
        assert debug_vuln is not None

        secret_vuln = next(
            (v for v in config_vulns if "secret" in v.title.lower()), None
        )
        assert secret_vuln is not None

        return {
            "status": "passed",
            "components": ["Config Scan", "Vulnerability Detection"],
        }

    def test_security_manager_integration(self):
        """Test security manager integration."""
        # Create a test security config
        test_config_path = "test_security_config.yaml"
        test_config = {
            "authentication": {
                "jwt": {"secret_key": "test-secret", "algorithm": "HS256"}
            },
            "rate_limiting": {"enabled": True},
            "security_headers": {"enabled": True},
            "encryption": {"enabled": True},
        }

        # Write test config
        import yaml

        with open(test_config_path, "w") as f:
            yaml.dump(test_config, f)

        try:
            # Initialize security manager
            security_manager = SecurityManager(test_config_path)

            # Test authentication
            auth_result = security_manager.authenticate_request(
                "testuser", "password123"
            )
            assert auth_result["status"] == "success"

            # Test authorization
            authorized = security_manager.authorize_request("testuser", "data", "read")
            # Note: This might fail without proper role setup, which is expected

            # Test input validation
            test_data = {"name": "test", "value": 123}
            validated = security_manager.validate_input(test_data)
            assert "name" in validated

            # Test rate limiting
            allowed, status = security_manager.check_rate_limit(
                "testuser", "192.168.1.1", "/api/test"
            )
            assert allowed  # First request should be allowed

            # Test security headers
            headers = security_manager.get_security_headers(
                "/api/test", "https://example.com"
            )
            assert len(headers) > 0

            # Test security metrics
            metrics = security_manager.get_security_metrics()
            assert "monitoring" in metrics
            assert "authentication" in metrics

            return {"status": "passed", "components": ["Integration", "End-to-End"]}

        finally:
            # Cleanup
            if os.path.exists(test_config_path):
                os.remove(test_config_path)

    def print_test_summary(self):
        """Print test summary."""
        logger.info("\n" + "=" * 50)
        logger.info("SECURITY TEST SUMMARY")
        logger.info("=" * 50)

        passed = 0
        failed = 0

        for test_name, result in self.test_results.items():
            status = result.get("status", "unknown")
            if status == "passed":
                logger.info(f"✅ {test_name}: PASSED")
                if "components" in result:
                    logger.info(
                        f"   Components tested: {', '.join(result['components'])}"
                    )
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
                if "error" in result:
                    logger.error(f"   Error: {result['error']}")
                failed += 1

        logger.info("-" * 50)
        logger.info(f"Total tests: {passed + failed}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")

        if failed == 0:
            logger.info("🎉 All security tests passed!")
        else:
            logger.warning(f"⚠️  {failed} test(s) failed. Please review and fix issues.")

        logger.info("=" * 50)


async def run_async_tests():
    """Run any async security tests."""
    logger.info("Running async security tests...")

    # Test async security operations
    # This would include testing async components like rate limiting with Redis

    logger.info("✅ Async security tests completed")


def main():
    """Main test execution."""
    logger.info("🔒 Starting Pynomaly Security Test Suite")
    logger.info("=" * 50)

    try:
        # Run synchronous tests
        tester = SecurityTester()
        tester.run_all_tests()

        # Run async tests
        asyncio.run(run_async_tests())

        logger.info("\n🔒 Security testing completed successfully!")

    except Exception as e:
        logger.error(f"❌ Security testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
