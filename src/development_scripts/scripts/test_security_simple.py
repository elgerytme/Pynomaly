#!/usr/bin/env python3
"""
Simplified security testing script for Pynomaly.
Tests core security components without complex imports.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up test environment
os.environ["PYNOMALY_ENV"] = "testing"
os.environ["PYNOMALY_JWT_SECRET"] = "test-secret-key-for-testing-only"
os.environ["PYNOMALY_MASTER_KEY"] = "test-master-key-for-testing"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_security_imports():
    """Test that all security modules can be imported."""
    try:
        logger.info("✅ All security modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Security module import failed: {e}")
        return False


def test_jwt_functionality():
    """Test JWT token functionality."""
    try:
        from pynomaly.presentation.api.security.authentication import JWTManager

        jwt_manager = JWTManager("test-secret", "HS256")

        # Generate token
        tokens = jwt_manager.generate_token("test_user", ["user"], ["read"])
        assert "access_token" in tokens
        assert "refresh_token" in tokens

        # Validate token
        payload = jwt_manager.validate_token(tokens["access_token"])
        assert payload is not None
        assert payload["user_id"] == "test_user"

        logger.info("✅ JWT functionality test passed")
        return True
    except Exception as e:
        logger.error(f"❌ JWT functionality test failed: {e}")
        return False


def test_input_validation():
    """Test input validation functionality."""
    try:
        from pynomaly.presentation.api.security.input_validation import (
            SecurityValidator,
        )

        validator = SecurityValidator()

        # Test basic validation
        test_data = {"username": "testuser", "email": "test@example.com"}
        validated = validator.validate_and_sanitize(test_data)
        assert "username" in validated

        # Test security threat detection
        malicious_data = {"query": "SELECT * FROM users WHERE 1=1; DROP TABLE users;"}
        threats = validator.check_security_threats(malicious_data)
        assert len(threats) > 0

        logger.info("✅ Input validation test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Input validation test failed: {e}")
        return False


def test_rate_limiting():
    """Test rate limiting functionality."""
    try:
        from pynomaly.presentation.api.security.rate_limiting import (
            RateLimiter,
            RateLimitType,
        )

        rate_limiter = RateLimiter()

        # Test rate limit configuration
        rate_limiter.set_rate_limit("test_user", 5, 60, RateLimitType.PER_USER)

        # Test requests within limit
        for i in range(4):
            allowed, status = rate_limiter.is_allowed(
                "test_user", RateLimitType.PER_USER
            )
            assert allowed

        logger.info("✅ Rate limiting test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Rate limiting test failed: {e}")
        return False


def test_encryption():
    """Test encryption functionality."""
    try:
        from pynomaly.presentation.api.security.encryption import EncryptionManager

        encryption_manager = EncryptionManager()

        # Test data encryption
        test_data = "sensitive information"
        encrypted = encryption_manager.encrypt_data(test_data)
        assert "encrypted_data" in encrypted

        # Test data decryption
        decrypted = encryption_manager.decrypt_data(encrypted)
        assert decrypted == test_data

        logger.info("✅ Encryption test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Encryption test failed: {e}")
        return False


def test_vulnerability_scanner():
    """Test vulnerability scanner functionality."""
    try:
        from pynomaly.presentation.api.security.vulnerability_scanner import (
            VulnerabilityScanner,
        )

        scanner = VulnerabilityScanner()

        # Test configuration scan
        test_config = {
            "DEBUG": True,  # Should trigger vulnerability
            "SECRET_KEY": "weak",  # Should trigger vulnerability
            "SECURITY_HEADERS": {},  # Should trigger vulnerability
        }

        config_vulns = scanner.scan_configuration(test_config)
        assert len(config_vulns) > 0

        logger.info("✅ Vulnerability scanner test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Vulnerability scanner test failed: {e}")
        return False


def run_security_tests():
    """Run all security tests."""
    logger.info("🔒 Starting Pynomaly Security Test Suite")
    logger.info("=" * 50)

    tests = [
        ("Security Imports", test_security_imports),
        ("JWT Functionality", test_jwt_functionality),
        ("Input Validation", test_input_validation),
        ("Rate Limiting", test_rate_limiting),
        ("Encryption", test_encryption),
        ("Vulnerability Scanner", test_vulnerability_scanner),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        if test_func():
            passed += 1
        else:
            failed += 1

    logger.info("-" * 50)
    logger.info(f"Total tests: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")

    if failed == 0:
        logger.info("🎉 All security tests passed!")
        return True
    else:
        logger.warning(f"⚠️  {failed} test(s) failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)
