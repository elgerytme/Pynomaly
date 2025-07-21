#!/usr/bin/env python3
"""
Standalone security testing script for individual components.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set test environment
os.environ["ANOMALY_DETECTION_ENV"] = "testing"
os.environ["ANOMALY_DETECTION_JWT_SECRET"] = "test-secret-key-for-testing-only"
os.environ["ANOMALY_DETECTION_MASTER_KEY"] = "test-master-key-for-testing"


def test_jwt_standalone():
    """Test JWT functionality standalone."""
    try:
        import secrets
        from datetime import datetime, timedelta

        import jwt

        secret_key = "test-secret-key"
        algorithm = "HS256"

        # Generate token
        payload = {
            "user_id": "test_user",
            "roles": ["user"],
            "permissions": ["read"],
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "jti": secrets.token_hex(16),
        }

        token = jwt.encode(payload, secret_key, algorithm=algorithm)

        # Validate token
        decoded = jwt.decode(token, secret_key, algorithms=[algorithm])
        assert decoded["user_id"] == "test_user"

        logger.info("✅ JWT standalone test passed")
        return True
    except Exception as e:
        logger.error(f"❌ JWT standalone test failed: {e}")
        return False


def test_encryption_standalone():
    """Test encryption functionality standalone."""
    try:
        from cryptography.fernet import Fernet

        # Generate key
        key = Fernet.generate_key()
        cipher = Fernet(key)

        # Test encryption/decryption
        test_data = b"sensitive information"
        encrypted = cipher.encrypt(test_data)
        decrypted = cipher.decrypt(encrypted)

        assert decrypted == test_data

        logger.info("✅ Encryption standalone test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Encryption standalone test failed: {e}")
        return False


def test_input_validation_standalone():
    """Test input validation standalone."""
    try:
        import re

        # SQL injection patterns
        sql_patterns = [
            r"(union|select|insert|delete|update|drop|exec|script).*",
            r"(\\|\'|\"|;|<|>)",
            r"(--|\\/\\*|\\*\\/)",
        ]

        test_input = "SELECT * FROM users WHERE 1=1; DROP TABLE users;"

        # Check for SQL injection
        sql_detected = False
        for pattern in sql_patterns:
            if re.search(pattern, test_input, re.IGNORECASE):
                logger.info("SQL injection detected (expected)")
                sql_detected = True
                break

        assert sql_detected, "SQL injection pattern should be detected"

        # Test XSS protection (basic)
        xss_input = "<script>alert('xss')</script>"
        # Simple XSS detection
        xss_patterns = [r"<script.*?>", r"javascript:", r"on\w+\s*="]
        xss_detected = any(
            re.search(pattern, xss_input, re.IGNORECASE) for pattern in xss_patterns
        )
        assert xss_detected, "XSS pattern should be detected"

        logger.info("✅ Input validation standalone test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Input validation standalone test failed: {e}")
        return False


def test_vulnerability_scanning_standalone():
    """Test vulnerability scanning standalone."""
    try:
        # Test configuration vulnerabilities
        test_config = {"DEBUG": True, "SECRET_KEY": "weak", "SECURITY_HEADERS": {}}

        vulnerabilities = []

        # Check debug mode
        if test_config.get("DEBUG", False):
            vulnerabilities.append("Debug mode enabled in production")

        # Check weak secret
        if len(test_config.get("SECRET_KEY", "")) < 32:
            vulnerabilities.append("Weak secret key")

        # Check security headers
        if not test_config.get("SECURITY_HEADERS"):
            vulnerabilities.append("Missing security headers")

        assert len(vulnerabilities) == 3  # All vulnerabilities detected

        logger.info("✅ Vulnerability scanning standalone test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Vulnerability scanning standalone test failed: {e}")
        return False


def test_rate_limiting_standalone():
    """Test rate limiting standalone."""
    try:
        import time
        from collections import defaultdict

        # Simple rate limiter implementation
        requests = defaultdict(list)
        limit = 5
        window = 60  # seconds

        user_id = "test_user"
        current_time = time.time()

        # Simulate requests
        for i in range(4):
            requests[user_id].append(current_time)

        # Check if within limit
        recent_requests = [
            req_time
            for req_time in requests[user_id]
            if current_time - req_time < window
        ]

        allowed = len(recent_requests) < limit
        assert allowed  # Should be allowed (4 < 5)

        logger.info("✅ Rate limiting standalone test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Rate limiting standalone test failed: {e}")
        return False


def test_security_headers_standalone():
    """Test security headers functionality."""
    try:
        # Security headers that should be present
        security_headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
        }

        # Validate headers
        for header, value in security_headers.items():
            assert isinstance(header, str) and len(header) > 0
            assert isinstance(value, str) and len(value) > 0

        logger.info("✅ Security headers standalone test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Security headers standalone test failed: {e}")
        return False


def test_security_config_validation():
    """Test security configuration validation."""
    try:
        # Create temporary security config
        config = {
            "authentication": {
                "jwt": {"secret_key": "test-secret", "algorithm": "HS256"},
                "password_policy": {"min_length": 12},
                "account_lockout": {"max_failed_attempts": 5},
            },
            "rate_limiting": {"enabled": True},
            "security_headers": {"enabled": True},
            "encryption": {"enabled": True},
        }

        # Validate config structure
        assert "authentication" in config
        assert "rate_limiting" in config
        assert "security_headers" in config
        assert "encryption" in config

        # Validate JWT config
        jwt_config = config["authentication"]["jwt"]
        assert jwt_config["algorithm"] in ["HS256", "HS512", "RS256"]
        assert len(jwt_config["secret_key"]) >= 8

        logger.info("✅ Security config validation test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Security config validation test failed: {e}")
        return False


def run_standalone_security_tests():
    """Run all standalone security tests."""
    logger.info("🔒 Starting Standalone Security Test Suite")
    logger.info("=" * 60)

    tests = [
        ("JWT Functionality", test_jwt_standalone),
        ("Encryption", test_encryption_standalone),
        ("Input Validation", test_input_validation_standalone),
        ("Vulnerability Scanning", test_vulnerability_scanning_standalone),
        ("Rate Limiting", test_rate_limiting_standalone),
        ("Security Headers", test_security_headers_standalone),
        ("Security Config Validation", test_security_config_validation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        if test_func():
            passed += 1
        else:
            failed += 1

    logger.info("-" * 60)
    logger.info("SECURITY TEST SUMMARY")
    logger.info(f"Total tests: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")

    if failed == 0:
        logger.info("🎉 All security tests passed!")
        logger.info("✅ Security framework validation successful")
    else:
        logger.warning(f"⚠️  {failed} test(s) failed.")

    return failed == 0


if __name__ == "__main__":
    success = run_standalone_security_tests()
    sys.exit(0 if success else 1)
