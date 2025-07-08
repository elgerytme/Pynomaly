"""Simple security and authentication integration tests."""

import pytest
from fastapi.testclient import TestClient
from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app


class TestSecurityIntegrationSimple:
    """Simple security integration tests."""

    @pytest.fixture
    def client_with_auth_enabled(self):
        """Create test client with authentication enabled."""
        container = create_container()
        # Override settings to enable auth for testing
        from pynomaly.infrastructure.config import Settings

        settings = Settings(auth_enabled=True, debug=True)
        container.config.override(settings)

        app = create_app(container)
        return TestClient(app)

    @pytest.fixture
    def client_with_auth_disabled(self):
        """Create test client with authentication disabled."""
        container = create_container()
        # Keep auth disabled for baseline testing
        from pynomaly.infrastructure.config import Settings

        settings = Settings(auth_enabled=False, debug=True)
        container.config.override(settings)

        app = create_app(container)
        return TestClient(app)

    def test_health_endpoint_works_without_auth(self, client_with_auth_disabled):
        """Test that health endpoint works when auth is disabled."""
        response = client_with_auth_disabled.get("/api/health/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "checks" in data

    def test_auth_system_can_be_configured(self, client_with_auth_enabled):
        """Test that authentication system can be properly configured."""
        # This test verifies the app can start with auth enabled
        # The actual auth logic testing would require more setup
        response = client_with_auth_enabled.get("/api/health/")
        # Health endpoint should work regardless of auth status
        assert response.status_code == 200

    def test_security_exceptions_are_importable(self):
        """Test that security exception classes can be imported correctly."""
        from pynomaly.domain.exceptions import AuthenticationError, AuthorizationError

        # Test AuthenticationError
        auth_error = AuthenticationError("Test auth error", username="testuser")
        assert str(auth_error) == "Test auth error | Details: username=testuser"
        assert auth_error.details["username"] == "testuser"

        # Test AuthorizationError
        authz_error = AuthorizationError(
            "Access denied", user_id="123", required_permission="read"
        )
        assert "Access denied" in str(authz_error)
        assert authz_error.details["user_id"] == "123"
        assert authz_error.details["required_permission"] == "read"

    def test_security_headers_configuration(self):
        """Test that security headers can be configured."""
        try:
            from pynomaly.infrastructure.security import (
                create_development_headers,
                create_production_headers,
            )

            dev_headers = create_development_headers()
            prod_headers = create_production_headers()

            # Both should return some configuration
            assert dev_headers is not None
            assert prod_headers is not None

        except ImportError:
            pytest.skip("Security headers module not available")

    def test_input_sanitization_available(self):
        """Test that input sanitization components are available."""
        try:
            from pynomaly.infrastructure.security import (
                InputSanitizer,
                SanitizationConfig,
            )

            config = SanitizationConfig(
                level="moderate", max_length=1000, allow_html=False
            )
            sanitizer = InputSanitizer(config)

            # Test basic sanitization
            clean_input = sanitizer.sanitize("Hello, World!")
            assert clean_input == "Hello, World!"

        except ImportError:
            pytest.skip("Input sanitization module not available")

    def test_encryption_services_available(self):
        """Test that encryption services are available."""
        try:
            from pynomaly.infrastructure.security import (
                EncryptionConfig,
                EncryptionService,
            )

            config = EncryptionConfig(
                algorithm="fernet", key_length=32, enable_key_rotation=False
            )
            encryption_service = EncryptionService(config)

            # Basic service instantiation test
            assert encryption_service is not None

        except ImportError:
            pytest.skip("Encryption services module not available")


if __name__ == "__main__":
    print("✅ Security integration test module loaded successfully")
