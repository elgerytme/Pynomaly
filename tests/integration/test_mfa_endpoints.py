#!/usr/bin/env python3
"""
Integration tests for MFA endpoints.
Tests the complete MFA flow including TOTP, SMS, email, and backup codes.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from pynomaly.application.dto.mfa_dto import MFAMethodType
from pynomaly.presentation.api.app import create_app


class TestMFAEndpoints:
    """Test MFA endpoint functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock application settings."""
        settings = Mock()
        settings.redis_url = "redis://localhost:6379"
        settings.get_cors_config.return_value = {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
        settings.secret_key = "test-secret-key"
        settings.app.name = "Test Pynomaly"
        settings.app.version = "1.0.0"
        settings.docs_enabled = True
        settings.cache_enabled = False
        settings.auth_enabled = False
        settings.monitoring.metrics_enabled = False
        settings.monitoring.tracing_enabled = False
        settings.monitoring.prometheus_enabled = False
        return settings

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        user = Mock()
        user.id = "user123"
        user.email = "test@example.com"
        user.username = "testuser"
        return user

    @pytest.fixture
    def mock_admin_user(self):
        """Mock authenticated admin user."""
        admin = Mock()
        admin.id = "admin123"
        admin.email = "admin@example.com"
        admin.username = "admin"
        admin.is_admin = True
        return admin

    @pytest.fixture
    def mock_mfa_service(self):
        """Mock MFA service."""
        service = Mock()
        service.generate_totp_secret.return_value = "TESTSECRET123"
        service.create_totp_setup_response.return_value = Mock(
            secret="TESTSECRET123",
            qr_code_url="data:image/png;base64,test",
            manual_entry_key="TESTSECRET123",
            backup_codes=["12345678", "87654321"]
        )
        service.confirm_totp_setup.return_value = True
        service.send_sms_code.return_value = True
        service.verify_sms_code.return_value = True
        service.send_email_code.return_value = True
        service.verify_email_code.return_value = True
        service.get_mfa_methods.return_value = []
        service.get_backup_codes_count.return_value = 5
        service.generate_backup_codes.return_value = ["12345678", "87654321"]
        service.verify_backup_code.return_value = True
        service.disable_mfa_method.return_value = True
        service.remember_device.return_value = "device123"
        service.get_trusted_devices.return_value = []
        service.revoke_trusted_device.return_value = True
        service.get_mfa_statistics.return_value = Mock(
            total_users=1000,
            mfa_enabled_users=650,
            mfa_adoption_rate=65.0,
            method_usage={"totp": 400, "sms": 200},
            recent_authentications=1200
        )
        return service

    @pytest.fixture
    def mock_auth_service(self):
        """Mock auth service."""
        service = Mock()
        service.create_access_token.return_value = "access_token_123"
        service.create_refresh_token.return_value = "refresh_token_123"
        return service

    @pytest.fixture
    def app_with_mfa(self, mock_settings, mock_user, mock_admin_user, mock_mfa_service, mock_auth_service):
        """Create FastAPI app with MFA endpoints."""
        with patch("pynomaly.infrastructure.config.create_container") as mock_container:
            container = Mock()
            container.config.return_value = mock_settings
            mock_container.return_value = container

            app = create_app(container)

            # Mock dependencies
            app.dependency_overrides = {}

            with patch("pynomaly.presentation.api.endpoints.mfa.get_current_user") as mock_get_user:
                mock_get_user.return_value = mock_user

                with patch("pynomaly.presentation.api.endpoints.mfa.get_current_admin_user") as mock_get_admin:
                    mock_get_admin.return_value = mock_admin_user

                    with patch("pynomaly.presentation.api.endpoints.mfa.get_mfa_service") as mock_get_mfa:
                        mock_get_mfa.return_value = mock_mfa_service

                        with patch("pynomaly.presentation.api.endpoints.mfa.get_auth") as mock_get_auth:
                            mock_get_auth.return_value = mock_auth_service

                            yield app

    def test_setup_totp_success(self, app_with_mfa):
        """Test successful TOTP setup."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/totp/setup",
            json={"app_name": "Pynomaly", "issuer": "Pynomaly Security"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["secret"] == "TESTSECRET123"
        assert data["manual_entry_key"] == "TESTSECRET123"
        assert "backup_codes" in data

    def test_setup_totp_missing_dependency(self, app_with_mfa, mock_mfa_service):
        """Test TOTP setup when pyotp is not available."""
        mock_mfa_service.generate_totp_secret.side_effect = RuntimeError("TOTP functionality requires pyotp library")

        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/totp/setup",
            json={"app_name": "Pynomaly", "issuer": "Pynomaly Security"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 500
        assert "Failed to set up TOTP authentication" in response.json()["detail"]

    def test_verify_totp_setup_success(self, app_with_mfa):
        """Test successful TOTP verification."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/totp/verify",
            json={"totp_code": "123456"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "TOTP authentication enabled successfully"

    def test_verify_totp_setup_invalid_code(self, app_with_mfa, mock_mfa_service):
        """Test TOTP verification with invalid code."""
        mock_mfa_service.confirm_totp_setup.return_value = False

        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/totp/verify",
            json={"totp_code": "123456"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 400
        assert "Invalid TOTP code" in response.json()["detail"]

    def test_setup_sms_success(self, app_with_mfa):
        """Test successful SMS setup."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/sms/setup",
            json={"phone_number": "+1234567890"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "SMS verification code sent successfully"

    def test_setup_sms_failure(self, app_with_mfa, mock_mfa_service):
        """Test SMS setup failure."""
        mock_mfa_service.send_sms_code.return_value = False

        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/sms/setup",
            json={"phone_number": "+1234567890"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 500
        assert "Failed to send SMS verification code" in response.json()["detail"]

    def test_verify_sms_success(self, app_with_mfa):
        """Test successful SMS verification."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/sms/verify",
            json={"sms_code": "123456"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "SMS authentication enabled successfully"

    def test_verify_sms_invalid_code(self, app_with_mfa, mock_mfa_service):
        """Test SMS verification with invalid code."""
        mock_mfa_service.verify_sms_code.return_value = False

        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/sms/verify",
            json={"sms_code": "123456"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 400
        assert "Invalid SMS code" in response.json()["detail"]

    def test_setup_email_success(self, app_with_mfa):
        """Test successful email setup."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/email/setup",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Email verification code sent successfully"

    def test_verify_email_success(self, app_with_mfa):
        """Test successful email verification."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/email/verify",
            json={"email_code": "123456"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Email authentication enabled successfully"

    def test_get_mfa_status(self, app_with_mfa, mock_mfa_service):
        """Test getting MFA status."""
        mock_mfa_service.get_mfa_methods.return_value = [
            Mock(
                id="totp_user123",
                method_type=MFAMethodType.TOTP,
                status=Mock(value="active"),
                display_name="Authenticator App",
                created_at="2024-01-01T00:00:00Z",
                is_primary=True
            )
        ]

        client = TestClient(app_with_mfa)

        response = client.get(
            "/api/v1/mfa/status",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mfa_enabled"] is True
        assert len(data["active_methods"]) == 1
        assert data["backup_codes_available"] is True

    def test_enable_mfa_totp(self, app_with_mfa):
        """Test enabling TOTP MFA."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/enable",
            json={
                "method_type": "totp",
                "verification_code": "123456",
                "set_as_primary": True
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "totp MFA enabled successfully"

    def test_enable_mfa_invalid_code(self, app_with_mfa, mock_mfa_service):
        """Test enabling MFA with invalid code."""
        mock_mfa_service.confirm_totp_setup.return_value = False

        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/enable",
            json={
                "method_type": "totp",
                "verification_code": "123456",
                "set_as_primary": True
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 400
        assert "Invalid verification code" in response.json()["detail"]

    def test_disable_mfa_totp(self, app_with_mfa, mock_mfa_service):
        """Test disabling TOTP MFA."""
        mock_mfa_service.verify_totp_code.return_value = True

        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/disable",
            json={
                "method_id": "totp_user123",
                "verification_code": "123456"
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "totp MFA disabled successfully"

    def test_disable_mfa_invalid_method_id(self, app_with_mfa):
        """Test disabling MFA with invalid method ID."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/disable",
            json={
                "method_id": "invalid_method_id",
                "verification_code": "123456"
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 400
        assert "Invalid method ID" in response.json()["detail"]

    def test_verify_mfa_login_success(self, app_with_mfa, mock_mfa_service):
        """Test successful MFA login verification."""
        mock_mfa_service.verify_totp_code.return_value = True

        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/verify",
            json={
                "method_type": "totp",
                "verification_code": "123456",
                "remember_device": True
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "access_token_123"
        assert data["refresh_token"] == "refresh_token_123"
        assert data["token_type"] == "bearer"
        assert data["device_remembered"] is True

    def test_verify_mfa_login_invalid_code(self, app_with_mfa, mock_mfa_service):
        """Test MFA login verification with invalid code."""
        mock_mfa_service.verify_totp_code.return_value = False

        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/verify",
            json={
                "method_type": "totp",
                "verification_code": "123456",
                "remember_device": False
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 400
        assert "Invalid MFA code" in response.json()["detail"]

    def test_get_backup_codes(self, app_with_mfa):
        """Test getting backup codes."""
        client = TestClient(app_with_mfa)

        response = client.get(
            "/api/v1/mfa/backup-codes",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["backup_codes"] == []  # API doesn't expose actual codes
        assert data["codes_remaining"] == 5

    def test_regenerate_backup_codes(self, app_with_mfa):
        """Test regenerating backup codes."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/backup-codes/regenerate",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["backup_codes"] == ["12345678", "87654321"]
        assert data["codes_remaining"] == 2

    def test_recover_with_backup_code(self, app_with_mfa):
        """Test account recovery with backup code."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/recovery",
            json={"backup_code": "12345678"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Account recovery successful"
        assert data["access_token"] == "access_token_123"
        assert data["refresh_token"] == "refresh_token_123"

    def test_recover_invalid_backup_code(self, app_with_mfa, mock_mfa_service):
        """Test account recovery with invalid backup code."""
        mock_mfa_service.verify_backup_code.return_value = False

        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/recovery",
            json={"backup_code": "12345678"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 400
        assert "Invalid backup code" in response.json()["detail"]

    def test_get_trusted_devices(self, app_with_mfa):
        """Test getting trusted devices."""
        client = TestClient(app_with_mfa)

        response = client.get(
            "/api/v1/mfa/trusted-devices",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["devices"] == []
        assert data["total_devices"] == 0

    def test_revoke_trusted_device(self, app_with_mfa):
        """Test revoking trusted device."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/trusted-devices/revoke",
            json={"device_id": "device123"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Trusted device revoked successfully"

    def test_revoke_trusted_device_not_found(self, app_with_mfa, mock_mfa_service):
        """Test revoking trusted device when not found."""
        mock_mfa_service.revoke_trusted_device.return_value = False

        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/trusted-devices/revoke",
            json={"device_id": "device123"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 404
        assert "Trusted device not found" in response.json()["detail"]

    def test_get_mfa_settings_admin(self, app_with_mfa):
        """Test getting MFA settings as admin."""
        client = TestClient(app_with_mfa)

        response = client.get(
            "/api/v1/mfa/admin/settings",
            headers={"Authorization": "Bearer admin_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["enforce_mfa"] is False
        assert "totp" in data["allowed_methods"]
        assert data["backup_codes_enabled"] is True

    def test_get_mfa_statistics_admin(self, app_with_mfa):
        """Test getting MFA statistics as admin."""
        client = TestClient(app_with_mfa)

        response = client.get(
            "/api/v1/mfa/admin/statistics",
            headers={"Authorization": "Bearer admin_token"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_users"] == 1000
        assert data["mfa_enabled_users"] == 650
        assert data["mfa_adoption_rate"] == 65.0
        assert "totp" in data["method_usage"]

    def test_invalid_totp_code_format(self, app_with_mfa):
        """Test TOTP verification with invalid code format."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/totp/verify",
            json={"totp_code": "12345"},  # Too short
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 422  # Validation error

    def test_invalid_sms_code_format(self, app_with_mfa):
        """Test SMS verification with invalid code format."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/sms/verify",
            json={"sms_code": "12345a"},  # Contains non-digit
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 422  # Validation error

    def test_invalid_phone_number_format(self, app_with_mfa):
        """Test SMS setup with invalid phone number format."""
        client = TestClient(app_with_mfa)

        response = client.post(
            "/api/v1/mfa/sms/setup",
            json={"phone_number": "invalid-phone"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 422  # Validation error

    def test_unauthorized_access(self, app_with_mfa):
        """Test unauthorized access to MFA endpoints."""
        client = TestClient(app_with_mfa)

        # Remove the authorization header
        response = client.get("/api/v1/mfa/status")

        assert response.status_code == 401  # Unauthorized

    def test_service_error_handling(self, app_with_mfa, mock_mfa_service):
        """Test error handling when MFA service fails."""
        mock_mfa_service.get_mfa_methods.side_effect = Exception("Service error")

        client = TestClient(app_with_mfa)

        response = client.get(
            "/api/v1/mfa/status",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 500
        assert "Failed to retrieve MFA status" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
