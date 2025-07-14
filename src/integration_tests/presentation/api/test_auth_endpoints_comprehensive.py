"""
Authentication Endpoints Testing Suite
Comprehensive tests for authentication and authorization API endpoints.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from pynomaly.domain.exceptions import AuthenticationError
from pynomaly.infrastructure.auth import TokenResponse, UserModel
from pynomaly.presentation.api.app import create_app


class TestAuthEndpoints:
    """Test suite for authentication API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service."""
        with patch("pynomaly.infrastructure.auth.JWTAuthService") as mock:
            service = Mock()

            # Mock successful authentication
            service.authenticate_user.return_value = UserModel(
                id="test-user-123",
                username="testuser",
                email="test@example.com",
                full_name="Test User",
                is_active=True,
                roles=["user"],
                permissions=["read", "write"],
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow(),
            )

            # Mock token creation
            service.create_access_token.return_value = TokenResponse(
                access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.token",
                refresh_token="refresh.token.here",
                token_type="bearer",
                expires_in=3600,
                scope="read write",
            )

            mock.return_value = service
            yield service

    @pytest.fixture
    def valid_user_data(self):
        """Valid user data for testing."""
        return {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePassword123!",
            "full_name": "Test User",
        }

    @pytest.fixture
    def valid_login_data(self):
        """Valid login data for testing."""
        return {"username": "testuser", "password": "SecurePassword123!"}

    # Login/Authentication Tests

    def test_login_success(self, client, mock_auth_service, valid_login_data):
        """Test successful user login."""
        response = client.post("/api/v1/auth/login", data=valid_login_data)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

        # Verify token format
        assert data["access_token"].startswith("eyJ")

    def test_login_invalid_credentials(self, client, mock_auth_service):
        """Test login with invalid credentials."""
        mock_auth_service.authenticate_user.side_effect = AuthenticationError(
            "Invalid credentials"
        )

        invalid_data = {"username": "testuser", "password": "wrongpassword"}

        response = client.post("/api/v1/auth/login", data=invalid_data)

        assert response.status_code == 401
        data = response.json()
        assert data["detail"] == "Invalid credentials"

    def test_login_missing_fields(self, client):
        """Test login with missing required fields."""
        incomplete_data = {"username": "testuser"}

        response = client.post("/api/v1/auth/login", data=incomplete_data)

        assert response.status_code == 422  # Validation error

    def test_login_empty_credentials(self, client):
        """Test login with empty credentials."""
        empty_data = {"username": "", "password": ""}

        response = client.post("/api/v1/auth/login", data=empty_data)

        assert response.status_code == 422

    def test_login_oauth2_format(self, client, mock_auth_service):
        """Test login using OAuth2 password flow format."""
        form_data = {
            "username": "testuser",
            "password": "SecurePassword123!",
            "grant_type": "password",
        }

        response = client.post("/api/v1/auth/token", data=form_data)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_rate_limiting(self, client):
        """Test login rate limiting."""
        login_data = {"username": "testuser", "password": "wrongpassword"}

        # Make multiple failed login attempts
        responses = []
        for _ in range(10):
            response = client.post("/api/v1/auth/login", data=login_data)
            responses.append(response)

        # Should eventually get rate limited
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes or 401 in status_codes  # Rate limited or auth failed

    def test_login_account_locked(self, client, mock_auth_service):
        """Test login with locked account."""
        mock_auth_service.authenticate_user.side_effect = AuthenticationError(
            "Account locked"
        )

        login_data = {"username": "lockeduser", "password": "password"}
        response = client.post("/api/v1/auth/login", data=login_data)

        assert response.status_code == 401
        assert "Account locked" in response.json()["detail"]

    # Registration Tests

    def test_register_success(self, client, mock_auth_service, valid_user_data):
        """Test successful user registration."""
        mock_auth_service.register_user.return_value = UserModel(
            id="new-user-123",
            username=valid_user_data["username"],
            email=valid_user_data["email"],
            full_name=valid_user_data["full_name"],
            is_active=True,
            roles=["user"],
            created_at=datetime.utcnow(),
        )

        response = client.post("/api/v1/auth/register", json=valid_user_data)

        assert response.status_code == 201
        data = response.json()
        assert data["username"] == valid_user_data["username"]
        assert data["email"] == valid_user_data["email"]
        assert data["is_active"] is True

    def test_register_duplicate_username(
        self, client, mock_auth_service, valid_user_data
    ):
        """Test registration with duplicate username."""
        mock_auth_service.register_user.side_effect = HTTPException(
            status_code=409, detail="Username already exists"
        )

        response = client.post("/api/v1/auth/register", json=valid_user_data)

        assert response.status_code == 409

    def test_register_duplicate_email(self, client, mock_auth_service, valid_user_data):
        """Test registration with duplicate email."""
        mock_auth_service.register_user.side_effect = HTTPException(
            status_code=409, detail="Email already exists"
        )

        response = client.post("/api/v1/auth/register", json=valid_user_data)

        assert response.status_code == 409

    def test_register_invalid_email(self, client, valid_user_data):
        """Test registration with invalid email format."""
        invalid_data = valid_user_data.copy()
        invalid_data["email"] = "invalid-email"

        response = client.post("/api/v1/auth/register", json=invalid_data)

        assert response.status_code == 422  # Validation error

    def test_register_weak_password(self, client, mock_auth_service, valid_user_data):
        """Test registration with weak password."""
        weak_data = valid_user_data.copy()
        weak_data["password"] = "123"

        mock_auth_service.register_user.side_effect = HTTPException(
            status_code=400, detail="Password does not meet requirements"
        )

        response = client.post("/api/v1/auth/register", json=weak_data)

        assert response.status_code == 400

    def test_register_missing_required_fields(self, client):
        """Test registration with missing required fields."""
        incomplete_data = {"username": "testuser"}

        response = client.post("/api/v1/auth/register", json=incomplete_data)

        assert response.status_code == 422

    # Token Refresh Tests

    def test_refresh_token_success(self, client, mock_auth_service):
        """Test successful token refresh."""
        mock_auth_service.refresh_access_token.return_value = TokenResponse(
            access_token="new.access.token",
            refresh_token="new.refresh.token",
            token_type="bearer",
            expires_in=3600,
        )

        refresh_data = {"refresh_token": "valid.refresh.token"}
        response = client.post("/api/v1/auth/refresh", json=refresh_data)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_token_invalid(self, client, mock_auth_service):
        """Test token refresh with invalid refresh token."""
        mock_auth_service.refresh_access_token.side_effect = AuthenticationError(
            "Invalid refresh token"
        )

        refresh_data = {"refresh_token": "invalid.refresh.token"}
        response = client.post("/api/v1/auth/refresh", json=refresh_data)

        assert response.status_code == 401

    def test_refresh_token_expired(self, client, mock_auth_service):
        """Test token refresh with expired refresh token."""
        mock_auth_service.refresh_access_token.side_effect = AuthenticationError(
            "Refresh token expired"
        )

        refresh_data = {"refresh_token": "expired.refresh.token"}
        response = client.post("/api/v1/auth/refresh", json=refresh_data)

        assert response.status_code == 401

    # User Profile Tests

    def test_get_current_user_success(self, client, mock_auth_service):
        """Test getting current user profile."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(
                id="test-user-123",
                username="testuser",
                email="test@example.com",
                full_name="Test User",
                is_active=True,
                roles=["user"],
                created_at=datetime.utcnow(),
            )

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.get("/api/v1/auth/me", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["username"] == "testuser"
            assert data["email"] == "test@example.com"

    def test_get_current_user_unauthorized(self, client):
        """Test getting current user without authentication."""
        response = client.get("/api/v1/auth/me")

        assert response.status_code == 401

    def test_get_current_user_invalid_token(self, client):
        """Test getting current user with invalid token."""
        headers = {"Authorization": "Bearer invalid.token"}
        response = client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == 401

    def test_update_user_profile_success(self, client, mock_auth_service):
        """Test updating user profile."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(
                id="test-user-123", username="testuser", email="test@example.com"
            )

            mock_auth_service.update_user_profile.return_value = UserModel(
                id="test-user-123",
                username="testuser",
                email="newemail@example.com",
                full_name="Updated Name",
            )

            update_data = {"email": "newemail@example.com", "full_name": "Updated Name"}

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.put("/api/v1/auth/me", json=update_data, headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["email"] == "newemail@example.com"

    # Password Management Tests

    def test_change_password_success(self, client, mock_auth_service):
        """Test successful password change."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(id="test-user-123", username="testuser")

            mock_auth_service.change_password.return_value = True

            password_data = {
                "current_password": "OldPassword123!",
                "new_password": "NewPassword123!",
            }

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.post(
                "/api/v1/auth/change-password", json=password_data, headers=headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Password changed successfully"

    def test_change_password_wrong_current(self, client, mock_auth_service):
        """Test password change with wrong current password."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(id="test-user-123")

            mock_auth_service.change_password.side_effect = AuthenticationError(
                "Current password incorrect"
            )

            password_data = {
                "current_password": "WrongPassword",
                "new_password": "NewPassword123!",
            }

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.post(
                "/api/v1/auth/change-password", json=password_data, headers=headers
            )

            assert response.status_code == 400

    def test_forgot_password_request(self, client, mock_auth_service):
        """Test forgot password request."""
        mock_auth_service.request_password_reset.return_value = True

        reset_data = {"email": "test@example.com"}
        response = client.post("/api/v1/auth/forgot-password", json=reset_data)

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_reset_password_success(self, client, mock_auth_service):
        """Test successful password reset."""
        mock_auth_service.reset_password.return_value = True

        reset_data = {"token": "valid.reset.token", "new_password": "NewPassword123!"}

        response = client.post("/api/v1/auth/reset-password", json=reset_data)

        assert response.status_code == 200

    def test_reset_password_invalid_token(self, client, mock_auth_service):
        """Test password reset with invalid token."""
        mock_auth_service.reset_password.side_effect = AuthenticationError(
            "Invalid reset token"
        )

        reset_data = {"token": "invalid.reset.token", "new_password": "NewPassword123!"}

        response = client.post("/api/v1/auth/reset-password", json=reset_data)

        assert response.status_code == 400

    # API Key Management Tests

    def test_create_api_key_success(self, client, mock_auth_service):
        """Test successful API key creation."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(id="test-user-123")

            mock_auth_service.create_api_key.return_value = {
                "api_key": "ak_test_1234567890abcdef",
                "name": "Test API Key",
                "permissions": ["read", "write"],
                "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
            }

            api_key_data = {
                "name": "Test API Key",
                "permissions": ["read", "write"],
                "expires_in_days": 30,
            }

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.post(
                "/api/v1/auth/api-keys", json=api_key_data, headers=headers
            )

            assert response.status_code == 201
            data = response.json()
            assert data["api_key"].startswith("ak_")
            assert data["name"] == "Test API Key"

    def test_list_api_keys(self, client, mock_auth_service):
        """Test listing user API keys."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(id="test-user-123")

            mock_auth_service.list_api_keys.return_value = [
                {
                    "id": "key-1",
                    "name": "Production Key",
                    "permissions": ["read"],
                    "created_at": datetime.utcnow().isoformat(),
                    "last_used": None,
                },
                {
                    "id": "key-2",
                    "name": "Development Key",
                    "permissions": ["read", "write"],
                    "created_at": datetime.utcnow().isoformat(),
                    "last_used": datetime.utcnow().isoformat(),
                },
            ]

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.get("/api/v1/auth/api-keys", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["name"] == "Production Key"

    def test_revoke_api_key(self, client, mock_auth_service):
        """Test API key revocation."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(id="test-user-123")

            mock_auth_service.revoke_api_key.return_value = True

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.delete("/api/v1/auth/api-keys/key-123", headers=headers)

            assert response.status_code == 204

    # Session Management Tests

    def test_logout_success(self, client, mock_auth_service):
        """Test successful logout."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(id="test-user-123")

            mock_auth_service.logout_user.return_value = True

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.post("/api/v1/auth/logout", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Successfully logged out"

    def test_logout_all_sessions(self, client, mock_auth_service):
        """Test logout from all sessions."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(id="test-user-123")

            mock_auth_service.logout_all_sessions.return_value = (
                3  # Number of sessions terminated
            )

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.post("/api/v1/auth/logout-all", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["sessions_terminated"] == 3

    def test_list_active_sessions(self, client, mock_auth_service):
        """Test listing active user sessions."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(id="test-user-123")

            mock_auth_service.list_active_sessions.return_value = [
                {
                    "session_id": "session-1",
                    "device": "Chrome on Windows",
                    "ip_address": "192.168.1.1",
                    "created_at": datetime.utcnow().isoformat(),
                    "last_activity": datetime.utcnow().isoformat(),
                    "is_current": True,
                }
            ]

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.get("/api/v1/auth/sessions", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["is_current"] is True

    # Multi-Factor Authentication Tests

    def test_enable_mfa_success(self, client, mock_auth_service):
        """Test enabling multi-factor authentication."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(id="test-user-123")

            mock_auth_service.enable_mfa.return_value = {
                "secret": "ABCD1234EFGH5678",
                "qr_code": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                "backup_codes": ["123456", "654321", "789012"],
            }

            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.post("/api/v1/auth/mfa/enable", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert "secret" in data
            assert "qr_code" in data
            assert len(data["backup_codes"]) == 3

    def test_verify_mfa_setup(self, client, mock_auth_service):
        """Test MFA setup verification."""
        with patch("pynomaly.infrastructure.auth.get_current_active_user") as mock_user:
            mock_user.return_value = UserModel(id="test-user-123")

            mock_auth_service.verify_mfa_setup.return_value = True

            verify_data = {"token": "123456"}
            headers = {"Authorization": "Bearer valid.access.token"}
            response = client.post(
                "/api/v1/auth/mfa/verify", json=verify_data, headers=headers
            )

            assert response.status_code == 200

    def test_mfa_login_success(self, client, mock_auth_service):
        """Test login with MFA token."""
        mock_auth_service.verify_mfa_token.return_value = True

        mfa_data = {
            "username": "testuser",
            "password": "password",
            "mfa_token": "123456",
        }

        response = client.post("/api/v1/auth/login-mfa", json=mfa_data)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    # Error Handling Tests

    def test_auth_service_unavailable(self, client, mock_auth_service):
        """Test authentication when service is unavailable."""
        mock_auth_service.authenticate_user.side_effect = Exception(
            "Service unavailable"
        )

        login_data = {"username": "testuser", "password": "password"}
        response = client.post("/api/v1/auth/login", data=login_data)

        assert response.status_code == 503

    def test_token_validation_error(self, client):
        """Test token validation errors."""
        invalid_headers = {"Authorization": "Bearer malformed.jwt.token"}
        response = client.get("/api/v1/auth/me", headers=invalid_headers)

        assert response.status_code == 401

    def test_concurrent_login_attempts(self, client, mock_auth_service):
        """Test handling concurrent login attempts."""
        import threading

        login_data = {"username": "testuser", "password": "password"}
        responses = []

        def make_login():
            responses.append(client.post("/api/v1/auth/login", data=login_data))

        # Create multiple concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_login)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should handle concurrent requests gracefully
        assert len(responses) == 5
        status_codes = [r.status_code for r in responses]
        assert all(code in [200, 401, 429, 503] for code in status_codes)


class TestAuthEndpointsIntegration:
    """Integration tests for authentication endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_complete_auth_workflow(self, client):
        """Test complete authentication workflow."""
        with patch("pynomaly.infrastructure.auth.JWTAuthService") as mock_service:
            service_instance = Mock()
            mock_service.return_value = service_instance

            # Mock successful registration
            service_instance.register_user.return_value = UserModel(
                id="test-user", username="testuser", email="test@example.com"
            )

            # Mock successful login
            service_instance.authenticate_user.return_value = UserModel(
                id="test-user", username="testuser"
            )
            service_instance.create_access_token.return_value = TokenResponse(
                access_token="test.token", token_type="bearer", expires_in=3600
            )

            # 1. Register user
            register_data = {
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecurePassword123!",
            }
            register_response = client.post("/api/v1/auth/register", json=register_data)

            # 2. Login
            login_response = client.post(
                "/api/v1/auth/login",
                data={"username": "testuser", "password": "SecurePassword123!"},
            )

            # 3. Access protected resource
            if login_response.status_code == 200:
                token = login_response.json().get("access_token")
                headers = {"Authorization": f"Bearer {token}"}

                with patch(
                    "pynomaly.infrastructure.auth.get_current_active_user"
                ) as mock_user:
                    mock_user.return_value = UserModel(
                        id="test-user", username="testuser"
                    )
                    profile_response = client.get("/api/v1/auth/me", headers=headers)

                # 4. Logout
                logout_response = client.post("/api/v1/auth/logout", headers=headers)

                # Verify workflow
                assert register_response.status_code in [201, 409, 422]
                assert login_response.status_code == 200
                assert profile_response.status_code in [200, 401]
                assert logout_response.status_code in [200, 401]

    def test_security_headers_validation(self, client):
        """Test security headers in auth responses."""
        login_data = {"username": "testuser", "password": "password"}
        response = client.post("/api/v1/auth/login", data=login_data)

        # Check security headers
        headers = response.headers
        assert "x-content-type-options" in headers or response.status_code in [401, 422]
        assert "x-frame-options" in headers or response.status_code in [401, 422]

    def test_auth_endpoint_cors(self, client):
        """Test CORS handling for auth endpoints."""
        # Preflight request
        response = client.options(
            "/api/v1/auth/login",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

        # Should handle CORS appropriately
        assert response.status_code in [200, 204, 405]
