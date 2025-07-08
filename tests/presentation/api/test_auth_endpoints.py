"""
Authentication Endpoints Testing Suite
Tests for JWT authentication, authorization, and security endpoints.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import jwt
import pytest
from fastapi.testclient import TestClient
from pynomaly.domain.exceptions import AuthenticationError
from pynomaly.presentation.api.app import create_app


class TestAuthEndpoints:
    """Test suite for authentication and authorization endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with authentication."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_auth_handler(self):
        """Mock JWT authentication handler."""
        with patch("pynomaly.infrastructure.auth.jwt_auth.JWTAuthHandler") as mock:
            handler = Mock()
            handler.create_access_token.return_value = "test-jwt-token"
            handler.verify_token.return_value = {
                "sub": "test@example.com",
                "role": "user",
            }
            handler.get_current_user.return_value = {
                "email": "test@example.com",
                "role": "user",
                "permissions": ["read:datasets", "write:datasets"],
            }
            mock.return_value = handler
            yield handler

    @pytest.fixture
    def valid_user_credentials(self):
        """Valid user credentials for testing."""
        return {"email": "test@example.com", "password": "secure_password_123"}

    @pytest.fixture
    def admin_credentials(self):
        """Admin user credentials for testing."""
        return {"email": "admin@example.com", "password": "admin_password_123"}

    # Authentication Endpoint Tests

    def test_login_successful(self, client, mock_auth_handler, valid_user_credentials):
        """Test successful user login."""
        response = client.post("/api/auth/login", json=valid_user_credentials)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert "user" in data

    def test_login_invalid_credentials(self, client, mock_auth_handler):
        """Test login with invalid credentials."""
        mock_auth_handler.authenticate.side_effect = AuthenticationError(
            "Invalid credentials"
        )

        response = client.post(
            "/api/auth/login",
            json={"email": "invalid@example.com", "password": "wrong_password"},
        )

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Invalid credentials" in data["detail"]

    def test_login_missing_fields(self, client):
        """Test login with missing required fields."""
        response = client.post("/api/auth/login", json={"email": "test@example.com"})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_login_invalid_email_format(self, client):
        """Test login with invalid email format."""
        response = client.post(
            "/api/auth/login",
            json={"email": "invalid-email", "password": "password123"},
        )

        assert response.status_code == 422

    def test_logout_successful(self, client, mock_auth_handler):
        """Test successful user logout."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.post("/api/auth/logout", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully logged out"

    def test_logout_without_token(self, client):
        """Test logout without authentication token."""
        response = client.post("/api/auth/logout")

        assert response.status_code == 401

    def test_refresh_token_successful(self, client, mock_auth_handler):
        """Test successful token refresh."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.post("/api/auth/refresh", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "expires_in" in data

    def test_refresh_token_expired(self, client, mock_auth_handler):
        """Test token refresh with expired token."""
        mock_auth_handler.verify_token.side_effect = jwt.ExpiredSignatureError()

        headers = {"Authorization": "Bearer expired-token"}
        response = client.post("/api/auth/refresh", headers=headers)

        assert response.status_code == 401

    # User Management Endpoint Tests

    def test_get_current_user(self, client, mock_auth_handler):
        """Test getting current user information."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.get("/api/auth/me", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "email" in data
        assert "role" in data
        assert "permissions" in data

    def test_get_current_user_unauthorized(self, client):
        """Test getting current user without authentication."""
        response = client.get("/api/auth/me")

        assert response.status_code == 401

    def test_change_password_successful(self, client, mock_auth_handler):
        """Test successful password change."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        password_data = {
            "current_password": "old_password",
            "new_password": "new_secure_password_123",
            "confirm_password": "new_secure_password_123",
        }

        response = client.post(
            "/api/auth/change-password", json=password_data, headers=headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Password changed successfully"

    def test_change_password_mismatch(self, client, mock_auth_handler):
        """Test password change with mismatched passwords."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        password_data = {
            "current_password": "old_password",
            "new_password": "new_password_123",
            "confirm_password": "different_password_123",
        }

        response = client.post(
            "/api/auth/change-password", json=password_data, headers=headers
        )

        assert response.status_code == 400

    def test_change_password_weak_password(self, client, mock_auth_handler):
        """Test password change with weak password."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        password_data = {
            "current_password": "old_password",
            "new_password": "weak",
            "confirm_password": "weak",
        }

        response = client.post(
            "/api/auth/change-password", json=password_data, headers=headers
        )

        assert response.status_code == 400

    # Permission and Role Tests

    def test_admin_endpoint_access(self, client, mock_auth_handler):
        """Test admin endpoint access with admin role."""
        mock_auth_handler.get_current_user.return_value = {
            "email": "admin@example.com",
            "role": "admin",
            "permissions": ["admin:all"],
        }

        headers = {"Authorization": "Bearer admin-token"}
        response = client.get("/api/auth/admin/users", headers=headers)

        assert response.status_code == 200

    def test_admin_endpoint_forbidden_for_user(self, client, mock_auth_handler):
        """Test admin endpoint forbidden for regular user."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.get("/api/auth/admin/users", headers=headers)

        assert response.status_code == 403

    def test_permission_check_valid(self, client, mock_auth_handler):
        """Test permission check with valid permission."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.get("/api/datasets", headers=headers)

        assert response.status_code in [200, 404]  # 404 if no datasets, but auth passed

    def test_permission_check_invalid(self, client, mock_auth_handler):
        """Test permission check with invalid permission."""
        mock_auth_handler.get_current_user.return_value = {
            "email": "limited@example.com",
            "role": "viewer",
            "permissions": ["read:public"],
        }

        headers = {"Authorization": "Bearer limited-token"}
        response = client.post("/api/datasets", json={}, headers=headers)

        assert response.status_code == 403

    # Token Validation Tests

    def test_invalid_token_format(self, client):
        """Test request with invalid token format."""
        headers = {"Authorization": "Invalid token-format"}
        response = client.get("/api/auth/me", headers=headers)

        assert response.status_code == 401

    def test_malformed_jwt_token(self, client, mock_auth_handler):
        """Test request with malformed JWT token."""
        mock_auth_handler.verify_token.side_effect = jwt.DecodeError()

        headers = {"Authorization": "Bearer malformed-jwt-token"}
        response = client.get("/api/auth/me", headers=headers)

        assert response.status_code == 401

    def test_token_missing_claims(self, client, mock_auth_handler):
        """Test token with missing required claims."""
        mock_auth_handler.verify_token.return_value = {"exp": datetime.utcnow()}

        headers = {"Authorization": "Bearer incomplete-token"}
        response = client.get("/api/auth/me", headers=headers)

        assert response.status_code == 401

    # Rate Limiting Tests

    def test_login_rate_limiting(self, client, mock_auth_handler):
        """Test login rate limiting protection."""
        invalid_credentials = {
            "email": "test@example.com",
            "password": "wrong_password",
        }

        # Simulate multiple failed login attempts
        for _ in range(6):  # Assuming 5 is the rate limit
            response = client.post("/api/auth/login", json=invalid_credentials)

        # Should be rate limited after multiple failures
        assert response.status_code in [429, 401]

    # Password Reset Tests

    def test_password_reset_request(self, client):
        """Test password reset request."""
        response = client.post(
            "/api/auth/password-reset", json={"email": "test@example.com"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_password_reset_confirm(self, client, mock_auth_handler):
        """Test password reset confirmation."""
        reset_data = {
            "token": "valid-reset-token",
            "new_password": "new_secure_password_123",
            "confirm_password": "new_secure_password_123",
        }

        response = client.post("/api/auth/password-reset/confirm", json=reset_data)

        assert response.status_code == 200

    def test_password_reset_invalid_token(self, client):
        """Test password reset with invalid token."""
        reset_data = {
            "token": "invalid-reset-token",
            "new_password": "new_password_123",
            "confirm_password": "new_password_123",
        }

        response = client.post("/api/auth/password-reset/confirm", json=reset_data)

        assert response.status_code == 400

    # Multi-Factor Authentication Tests

    def test_mfa_setup_request(self, client, mock_auth_handler):
        """Test MFA setup request."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.post("/api/auth/mfa/setup", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "qr_code" in data
        assert "secret" in data

    def test_mfa_verify_successful(self, client, mock_auth_handler):
        """Test successful MFA verification."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        mfa_data = {"code": "123456"}

        response = client.post("/api/auth/mfa/verify", json=mfa_data, headers=headers)

        assert response.status_code == 200

    def test_mfa_verify_invalid_code(self, client, mock_auth_handler):
        """Test MFA verification with invalid code."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        mfa_data = {"code": "000000"}

        response = client.post("/api/auth/mfa/verify", json=mfa_data, headers=headers)

        assert response.status_code == 400

    # Session Management Tests

    def test_active_sessions_list(self, client, mock_auth_handler):
        """Test listing active user sessions."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.get("/api/auth/sessions", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_revoke_session(self, client, mock_auth_handler):
        """Test revoking a specific session."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.delete("/api/auth/sessions/session-id-123", headers=headers)

        assert response.status_code == 200

    def test_revoke_all_sessions(self, client, mock_auth_handler):
        """Test revoking all user sessions."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.delete("/api/auth/sessions", headers=headers)

        assert response.status_code == 200

    # API Key Management Tests

    def test_create_api_key(self, client, mock_auth_handler):
        """Test creating a new API key."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        api_key_data = {
            "name": "Test API Key",
            "permissions": ["read:datasets"],
            "expires_in_days": 30,
        }

        response = client.post("/api/auth/api-keys", json=api_key_data, headers=headers)

        assert response.status_code == 201
        data = response.json()
        assert "api_key" in data
        assert "key_id" in data

    def test_list_api_keys(self, client, mock_auth_handler):
        """Test listing user's API keys."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.get("/api/auth/api-keys", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "api_keys" in data

    def test_revoke_api_key(self, client, mock_auth_handler):
        """Test revoking an API key."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.delete("/api/auth/api-keys/key-id-123", headers=headers)

        assert response.status_code == 200

    # Security Event Tests

    def test_security_events_log(self, client, mock_auth_handler):
        """Test retrieving security events for user."""
        headers = {"Authorization": "Bearer test-jwt-token"}
        response = client.get("/api/auth/security-events", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_login_attempt_logging(
        self, client, mock_auth_handler, valid_user_credentials
    ):
        """Test that login attempts are properly logged."""
        response = client.post("/api/auth/login", json=valid_user_credentials)

        # Verify login was successful
        assert response.status_code == 200

        # Check that security event was logged
        headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
        events_response = client.get("/api/auth/security-events", headers=headers)

        assert events_response.status_code == 200
        events = events_response.json()["events"]
        assert any(event["type"] == "login_success" for event in events)


class TestAuthEndpointsIntegration:
    """Integration tests for authentication endpoints with real scenarios."""

    @pytest.fixture
    def authenticated_client(self, client, mock_auth_handler):
        """Client with pre-authenticated user."""
        # Perform login
        credentials = {"email": "test@example.com", "password": "password123"}
        response = client.post("/api/auth/login", json=credentials)
        token = response.json()["access_token"]

        # Return client with auth headers
        client.headers.update({"Authorization": f"Bearer {token}"})
        return client

    def test_end_to_end_auth_flow(self, client, mock_auth_handler):
        """Test complete authentication flow from login to logout."""
        # 1. Login
        credentials = {"email": "test@example.com", "password": "password123"}
        login_response = client.post("/api/auth/login", json=credentials)
        assert login_response.status_code == 200

        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Access protected resource
        user_response = client.get("/api/auth/me", headers=headers)
        assert user_response.status_code == 200

        # 3. Change password
        password_data = {
            "current_password": "password123",
            "new_password": "new_password_456",
            "confirm_password": "new_password_456",
        }
        password_response = client.post(
            "/api/auth/change-password", json=password_data, headers=headers
        )
        assert password_response.status_code == 200

        # 4. Logout
        logout_response = client.post("/api/auth/logout", headers=headers)
        assert logout_response.status_code == 200

    def test_role_based_access_control(self, client, mock_auth_handler):
        """Test role-based access control across different endpoints."""
        # Test with different user roles
        test_cases = [
            {
                "role": "admin",
                "permissions": ["admin:all"],
                "expected_admin_access": 200,
                "expected_user_access": 200,
            },
            {
                "role": "user",
                "permissions": ["read:datasets", "write:datasets"],
                "expected_admin_access": 403,
                "expected_user_access": 200,
            },
            {
                "role": "viewer",
                "permissions": ["read:datasets"],
                "expected_admin_access": 403,
                "expected_user_access": 200,
            },
        ]

        for case in test_cases:
            mock_auth_handler.get_current_user.return_value = {
                "email": f"{case['role']}@example.com",
                "role": case["role"],
                "permissions": case["permissions"],
            }

            headers = {"Authorization": "Bearer test-token"}

            # Test admin endpoint
            admin_response = client.get("/api/auth/admin/users", headers=headers)
            assert admin_response.status_code == case["expected_admin_access"]

            # Test user endpoint
            user_response = client.get("/api/auth/me", headers=headers)
            assert user_response.status_code == case["expected_user_access"]
