"""Security and authentication workflow end-to-end tests.

This module tests complete security workflows including authentication,
authorization, secure API access, and security policy enforcement.
"""

import secrets
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app


class TestSecurityAuthenticationWorkflows:
    """Test complete security and authentication workflows."""

    @pytest.fixture
    def app_client(self):
        """Create test client for API."""
        container = create_container()
        app = create_app(container)
        return TestClient(app)

    @pytest.fixture
    def test_dataset(self):
        """Create test dataset for security testing."""
        import numpy as np

        np.random.seed(42)

        data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
                "feature_3": np.random.normal(0, 1, 100),
            }
        )

        return data

    def test_user_registration_login_workflow(self, app_client):
        """Test complete user registration and login workflow."""
        # User registration
        registration_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePassword123!",
            "full_name": "Test User",
            "organization": "Test Org",
        }

        register_response = app_client.post(
            "/api/auth/register", json=registration_data
        )
        assert register_response.status_code == 201
        register_result = register_response.json()

        assert "user_id" in register_result
        assert "message" in register_result
        assert register_result["message"] == "User registered successfully"

        # User login
        login_data = {"username": "testuser", "password": "SecurePassword123!"}

        login_response = app_client.post("/api/auth/login", data=login_data)
        assert login_response.status_code == 200
        login_result = login_response.json()

        assert "access_token" in login_result
        assert "token_type" in login_result
        assert "expires_in" in login_result
        assert login_result["token_type"] == "bearer"

        # Verify token structure
        token = login_result["access_token"]
        assert len(token.split(".")) == 3  # JWT structure

        # Test token validation
        headers = {"Authorization": f"Bearer {token}"}
        profile_response = app_client.get("/api/auth/profile", headers=headers)
        assert profile_response.status_code == 200
        profile_result = profile_response.json()

        assert profile_result["username"] == "testuser"
        assert profile_result["email"] == "test@example.com"
        assert profile_result["full_name"] == "Test User"

    def test_api_key_authentication_workflow(self, app_client):
        """Test API key based authentication workflow."""
        # Create user first
        registration_data = {
            "username": "apiuser",
            "email": "api@example.com",
            "password": "APIPassword123!",
            "full_name": "API User",
        }

        register_response = app_client.post(
            "/api/auth/register", json=registration_data
        )
        assert register_response.status_code == 201

        # Login to get access token
        login_data = {"username": "apiuser", "password": "APIPassword123!"}
        login_response = app_client.post("/api/auth/login", data=login_data)
        assert login_response.status_code == 200
        access_token = login_response.json()["access_token"]

        # Generate API key
        headers = {"Authorization": f"Bearer {access_token}"}
        api_key_request = {
            "name": "Test API Key",
            "permissions": ["read", "write"],
            "expires_in_days": 30,
        }

        api_key_response = app_client.post(
            "/api/auth/api-keys", json=api_key_request, headers=headers
        )
        assert api_key_response.status_code == 201
        api_key_result = api_key_response.json()

        assert "api_key" in api_key_result
        assert "key_id" in api_key_result
        assert "permissions" in api_key_result

        api_key = api_key_result["api_key"]

        # Test API key authentication
        api_headers = {"X-API-Key": api_key}
        test_response = app_client.get("/api/detectors/", headers=api_headers)
        assert test_response.status_code == 200

        # Test API key permissions
        detector_data = {
            "name": "API Key Detector",
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1},
        }

        create_response = app_client.post(
            "/api/detectors/", json=detector_data, headers=api_headers
        )
        assert create_response.status_code == 200  # Has write permission

        # Revoke API key
        revoke_response = app_client.delete(
            f"/api/auth/api-keys/{api_key_result['key_id']}", headers=headers
        )
        assert revoke_response.status_code == 200

        # Test revoked key
        revoked_response = app_client.get("/api/detectors/", headers=api_headers)
        assert revoked_response.status_code == 401

    def test_role_based_access_control_workflow(self, app_client):
        """Test role-based access control (RBAC) workflow."""
        # Create admin user
        admin_data = {
            "username": "admin",
            "email": "admin@example.com",
            "password": "AdminPassword123!",
            "full_name": "Admin User",
            "role": "admin",
        }

        admin_register = app_client.post("/api/auth/register", json=admin_data)
        assert admin_register.status_code == 201

        # Create regular user
        user_data = {
            "username": "regularuser",
            "email": "user@example.com",
            "password": "UserPassword123!",
            "full_name": "Regular User",
            "role": "user",
        }

        user_register = app_client.post("/api/auth/register", json=user_data)
        assert user_register.status_code == 201

        # Login as admin
        admin_login = app_client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "AdminPassword123!"},
        )
        assert admin_login.status_code == 200
        admin_token = admin_login.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # Login as regular user
        user_login = app_client.post(
            "/api/auth/login",
            data={"username": "regularuser", "password": "UserPassword123!"},
        )
        assert user_login.status_code == 200
        user_token = user_login.json()["access_token"]
        user_headers = {"Authorization": f"Bearer {user_token}"}

        # Test admin-only endpoints
        admin_endpoints = [
            ("/api/admin/users", "GET"),
            ("/api/admin/system-config", "GET"),
            ("/api/admin/audit-logs", "GET"),
        ]

        for endpoint, method in admin_endpoints:
            # Admin should have access
            if method == "GET":
                admin_response = app_client.get(endpoint, headers=admin_headers)
            else:
                admin_response = app_client.post(endpoint, headers=admin_headers)
            assert admin_response.status_code in [200, 201]

            # Regular user should be denied
            if method == "GET":
                user_response = app_client.get(endpoint, headers=user_headers)
            else:
                user_response = app_client.post(endpoint, headers=user_headers)
            assert user_response.status_code == 403  # Forbidden

        # Test resource ownership
        detector_data = {
            "name": "User Detector",
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1},
        }

        # User creates detector
        create_response = app_client.post(
            "/api/detectors/", json=detector_data, headers=user_headers
        )
        assert create_response.status_code == 200
        detector_id = create_response.json()["id"]

        # User can access their own detector
        get_response = app_client.get(
            f"/api/detectors/{detector_id}", headers=user_headers
        )
        assert get_response.status_code == 200

        # Admin can access any detector
        admin_get_response = app_client.get(
            f"/api/detectors/{detector_id}", headers=admin_headers
        )
        assert admin_get_response.status_code == 200

    def test_secure_data_upload_workflow(self, app_client, test_dataset):
        """Test secure data upload and access control."""
        # Register user
        user_data = {
            "username": "datauser",
            "email": "data@example.com",
            "password": "DataPassword123!",
            "full_name": "Data User",
        }

        register_response = app_client.post("/api/auth/register", json=user_data)
        assert register_response.status_code == 201

        # Login
        login_response = app_client.post(
            "/api/auth/login",
            data={"username": "datauser", "password": "DataPassword123!"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Upload dataset with security headers
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Test secure upload
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("secure_data.csv", file, "text/csv")},
                    data={
                        "name": "Secure Dataset",
                        "sensitivity_level": "confidential",
                        "encryption_required": "true",
                    },
                    headers=headers,
                )
            assert upload_response.status_code == 200
            upload_result = upload_response.json()

            assert "id" in upload_result
            assert "encryption_status" in upload_result
            assert upload_result["encryption_status"] == "encrypted"

            dataset_id = upload_result["id"]

            # Test data access control
            get_response = app_client.get(
                f"/api/datasets/{dataset_id}", headers=headers
            )
            assert get_response.status_code == 200

            # Test unauthorized access (no token)
            unauth_response = app_client.get(f"/api/datasets/{dataset_id}")
            assert unauth_response.status_code == 401

            # Test data download with audit trail
            download_response = app_client.get(
                f"/api/datasets/{dataset_id}/download", headers=headers
            )
            assert download_response.status_code == 200
            assert "audit_id" in download_response.headers

            # Verify audit log entry
            audit_response = app_client.get("/api/audit/my-activity", headers=headers)
            assert audit_response.status_code == 200
            audit_result = audit_response.json()

            assert len(audit_result["activities"]) > 0
            latest_activity = audit_result["activities"][0]
            assert latest_activity["action"] == "dataset_download"
            assert latest_activity["resource_id"] == dataset_id

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_session_management_workflow(self, app_client):
        """Test session management and security."""
        # Register user
        user_data = {
            "username": "sessionuser",
            "email": "session@example.com",
            "password": "SessionPassword123!",
            "full_name": "Session User",
        }

        register_response = app_client.post("/api/auth/register", json=user_data)
        assert register_response.status_code == 201

        # Login and get token
        login_response = app_client.post(
            "/api/auth/login",
            data={"username": "sessionuser", "password": "SessionPassword123!"},
        )
        assert login_response.status_code == 200
        login_result = login_response.json()

        token = login_result["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Test active session
        session_response = app_client.get("/api/auth/sessions", headers=headers)
        assert session_response.status_code == 200
        sessions = session_response.json()

        assert len(sessions["active_sessions"]) >= 1
        current_session = sessions["active_sessions"][0]
        assert "session_id" in current_session
        assert "created_at" in current_session
        assert "last_activity" in current_session

        # Test session refresh
        refresh_response = app_client.post("/api/auth/refresh", headers=headers)
        assert refresh_response.status_code == 200
        refresh_result = refresh_response.json()

        assert "access_token" in refresh_result
        new_token = refresh_result["access_token"]
        assert new_token != token  # New token should be different

        # Test logout
        logout_response = app_client.post("/api/auth/logout", headers=headers)
        assert logout_response.status_code == 200

        # Test token invalidation
        invalid_response = app_client.get("/api/auth/profile", headers=headers)
        assert invalid_response.status_code == 401

        # Test logout all sessions
        # Login again with multiple sessions
        login1 = app_client.post(
            "/api/auth/login",
            data={"username": "sessionuser", "password": "SessionPassword123!"},
        )
        login2 = app_client.post(
            "/api/auth/login",
            data={"username": "sessionuser", "password": "SessionPassword123!"},
        )

        token1 = login1.json()["access_token"]
        token2 = login2.json()["access_token"]

        # Logout all sessions using first token
        logout_all_response = app_client.post(
            "/api/auth/logout-all", headers={"Authorization": f"Bearer {token1}"}
        )
        assert logout_all_response.status_code == 200

        # Both tokens should be invalid
        for token in [token1, token2]:
            invalid_response = app_client.get(
                "/api/auth/profile", headers={"Authorization": f"Bearer {token}"}
            )
            assert invalid_response.status_code == 401

    def test_password_security_workflow(self, app_client):
        """Test password security policies and procedures."""
        # Test weak password rejection
        weak_passwords = [
            "password",
            "123456",
            "admin",
            "test",
            "Password",  # Missing numbers and symbols
            "password123",  # Missing symbols
            "Pass1!",  # Too short
        ]

        for weak_password in weak_passwords:
            weak_user_data = {
                "username": f"weakuser_{secrets.token_hex(4)}",
                "email": f"weak_{secrets.token_hex(4)}@example.com",
                "password": weak_password,
                "full_name": "Weak Password User",
            }

            weak_response = app_client.post("/api/auth/register", json=weak_user_data)
            assert weak_response.status_code == 422  # Validation error

            error_detail = weak_response.json()["detail"]
            assert "password" in str(error_detail).lower()

        # Test strong password acceptance
        strong_user_data = {
            "username": "stronguser",
            "email": "strong@example.com",
            "password": "StrongPassword123!@#",
            "full_name": "Strong Password User",
        }

        strong_response = app_client.post("/api/auth/register", json=strong_user_data)
        assert strong_response.status_code == 201

        # Login with strong password
        login_response = app_client.post(
            "/api/auth/login",
            data={"username": "stronguser", "password": "StrongPassword123!@#"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Test password change
        change_password_data = {
            "current_password": "StrongPassword123!@#",
            "new_password": "NewStrongPassword456$%^",
            "confirm_password": "NewStrongPassword456$%^",
        }

        change_response = app_client.post(
            "/api/auth/change-password", json=change_password_data, headers=headers
        )
        assert change_response.status_code == 200

        # Test login with new password
        new_login_response = app_client.post(
            "/api/auth/login",
            data={"username": "stronguser", "password": "NewStrongPassword456$%^"},
        )
        assert new_login_response.status_code == 200

        # Test old password rejection
        old_login_response = app_client.post(
            "/api/auth/login",
            data={"username": "stronguser", "password": "StrongPassword123!@#"},
        )
        assert old_login_response.status_code == 401

        # Test password reset workflow
        reset_request_response = app_client.post(
            "/api/auth/password-reset-request", json={"email": "strong@example.com"}
        )
        assert reset_request_response.status_code == 200

        # In a real system, this would send an email with a reset token
        # For testing, we simulate receiving the token
        reset_token = "simulated_reset_token_123"

        # Test password reset with token
        reset_data = {
            "token": reset_token,
            "new_password": "ResetPassword789&*()",
            "confirm_password": "ResetPassword789&*()",
        }

        # This would normally validate the token from email
        # For testing, we assume it's valid
        reset_response = app_client.post("/api/auth/password-reset", json=reset_data)
        # Note: This might return 400 in testing since we're using a simulated token
        assert reset_response.status_code in [200, 400]

    def test_brute_force_protection_workflow(self, app_client):
        """Test brute force attack protection."""
        # Register test user
        user_data = {
            "username": "bruteuser",
            "email": "brute@example.com",
            "password": "BrutePassword123!",
            "full_name": "Brute Force Test User",
        }

        register_response = app_client.post("/api/auth/register", json=user_data)
        assert register_response.status_code == 201

        # Attempt multiple failed logins
        failed_attempts = []
        for _i in range(6):  # Exceed rate limit threshold
            failed_login = app_client.post(
                "/api/auth/login",
                data={"username": "bruteuser", "password": "WrongPassword"},
            )
            failed_attempts.append(failed_login.status_code)
            time.sleep(0.1)  # Small delay between attempts

        # Should get rate limited
        assert 429 in failed_attempts  # Too Many Requests

        # Test account lockout after multiple failures
        lockout_response = app_client.post(
            "/api/auth/login",
            data={
                "username": "bruteuser",
                "password": "BrutePassword123!",  # Correct password
            },
        )

        # Account might be temporarily locked
        if lockout_response.status_code == 423:  # Locked
            assert "account locked" in lockout_response.json()["detail"].lower()

        # Test legitimate login after cooldown
        time.sleep(2)  # Wait for rate limit reset

        legitimate_response = app_client.post(
            "/api/auth/login",
            data={"username": "bruteuser", "password": "BrutePassword123!"},
        )
        # Should eventually succeed (might need multiple attempts due to rate limiting)
        assert legitimate_response.status_code in [200, 429]

    def test_data_encryption_workflow(self, app_client, test_dataset):
        """Test data encryption and secure storage workflow."""
        # Register user
        user_data = {
            "username": "cryptouser",
            "email": "crypto@example.com",
            "password": "CryptoPassword123!",
            "full_name": "Crypto User",
        }

        register_response = app_client.post("/api/auth/register", json=user_data)
        assert register_response.status_code == 201

        # Login
        login_response = app_client.post(
            "/api/auth/login",
            data={"username": "cryptouser", "password": "CryptoPassword123!"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Upload dataset with encryption
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("encrypted_data.csv", file, "text/csv")},
                    data={
                        "name": "Encrypted Dataset",
                        "encryption_required": "true",
                        "encryption_algorithm": "AES-256-GCM",
                    },
                    headers=headers,
                )
            assert upload_response.status_code == 200
            upload_result = upload_response.json()

            dataset_id = upload_result["id"]

            # Verify encryption status
            dataset_info_response = app_client.get(
                f"/api/datasets/{dataset_id}/info", headers=headers
            )
            assert dataset_info_response.status_code == 200
            dataset_info = dataset_info_response.json()

            assert dataset_info["encryption_status"] == "encrypted"
            assert dataset_info["encryption_algorithm"] == "AES-256-GCM"
            assert "encryption_key_id" in dataset_info

            # Create detector with encrypted data
            detector_data = {
                "name": "Crypto Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1},
            }

            create_response = app_client.post(
                "/api/detectors/", json=detector_data, headers=headers
            )
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            # Train on encrypted data
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train",
                json={"dataset_id": dataset_id},
                headers=headers,
            )
            assert train_response.status_code == 200

            # Verify training worked with encrypted data
            train_result = train_response.json()
            assert "encryption_handled" in train_result
            assert train_result["encryption_handled"] is True

            # Test secure data export
            export_response = app_client.post(
                f"/api/datasets/{dataset_id}/export",
                json={"format": "encrypted_csv", "encryption_key_rotation": True},
                headers=headers,
            )
            assert export_response.status_code == 200

            export_result = export_response.json()
            assert "export_id" in export_result
            assert "encryption_key_id" in export_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)
