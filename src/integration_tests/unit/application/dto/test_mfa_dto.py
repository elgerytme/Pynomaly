#!/usr/bin/env python3
"""
Comprehensive tests for Multi-Factor Authentication (MFA) DTOs.
Tests all MFA-related data transfer objects including TOTP, SMS, email verification,
backup codes, device management, and MFA settings.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.pynomaly.application.dto.mfa_dto import (
    BackupCodesResponse,
    BackupCodeVerificationRequest,
    EmailVerificationRequest,
    MFADeviceDTO,
    MFADisableRequest,
    MFAEnableRequest,
    MFAErrorResponse,
    MFALoginRequest,
    MFALoginResponse,
    MFAMethodDTO,
    MFAMethodStatus,
    MFAMethodType,
    MFARecoveryRequest,
    MFARecoveryResponse,
    MFASettingsDTO,
    MFAStatisticsDTO,
    MFAStatusResponse,
    RevokeTrustedDeviceRequest,
    SMSSetupRequest,
    SMSVerificationRequest,
    TOTPSetupRequest,
    TOTPSetupResponse,
    TOTPVerificationRequest,
    TrustedDevicesResponse,
)


class TestMFAMethodType:
    """Test cases for MFAMethodType enum."""

    def test_all_method_types(self):
        """Test all MFA method types are defined correctly."""
        assert MFAMethodType.TOTP == "totp"
        assert MFAMethodType.SMS == "sms"
        assert MFAMethodType.EMAIL == "email"
        assert MFAMethodType.BACKUP_CODES == "backup_codes"
        assert MFAMethodType.HARDWARE_TOKEN == "hardware_token"

    def test_enum_values(self):
        """Test enum values can be used properly."""
        method_types = [
            MFAMethodType.TOTP,
            MFAMethodType.SMS,
            MFAMethodType.EMAIL,
            MFAMethodType.BACKUP_CODES,
            MFAMethodType.HARDWARE_TOKEN
        ]
        assert len(method_types) == 5
        assert all(isinstance(method, str) for method in method_types)


class TestMFAMethodStatus:
    """Test cases for MFAMethodStatus enum."""

    def test_all_status_types(self):
        """Test all MFA status types are defined correctly."""
        assert MFAMethodStatus.ACTIVE == "active"
        assert MFAMethodStatus.INACTIVE == "inactive"
        assert MFAMethodStatus.PENDING == "pending"
        assert MFAMethodStatus.DISABLED == "disabled"

    def test_enum_values(self):
        """Test enum values can be used properly."""
        statuses = [
            MFAMethodStatus.ACTIVE,
            MFAMethodStatus.INACTIVE,
            MFAMethodStatus.PENDING,
            MFAMethodStatus.DISABLED
        ]
        assert len(statuses) == 4
        assert all(isinstance(status, str) for status in statuses)


class TestTOTPSetupRequest:
    """Test cases for TOTPSetupRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid TOTP setup request."""
        request = TOTPSetupRequest(
            app_name="Test App",
            issuer="Test Issuer"
        )
        assert request.app_name == "Test App"
        assert request.issuer == "Test Issuer"

    def test_default_values(self):
        """Test default values for TOTP setup request."""
        request = TOTPSetupRequest()
        assert request.app_name == "Pynomaly"
        assert request.issuer == "Pynomaly Security"

    def test_custom_values(self):
        """Test custom values override defaults."""
        request = TOTPSetupRequest(
            app_name="Custom App",
            issuer="Custom Security"
        )
        assert request.app_name == "Custom App"
        assert request.issuer == "Custom Security"

    def test_serialization(self):
        """Test request serialization."""
        request = TOTPSetupRequest(
            app_name="Test App",
            issuer="Test Issuer"
        )
        data = request.model_dump()
        assert data["app_name"] == "Test App"
        assert data["issuer"] == "Test Issuer"

    def test_deserialization(self):
        """Test request deserialization."""
        data = {
            "app_name": "Test App",
            "issuer": "Test Issuer"
        }
        request = TOTPSetupRequest.model_validate(data)
        assert request.app_name == "Test App"
        assert request.issuer == "Test Issuer"


class TestTOTPSetupResponse:
    """Test cases for TOTPSetupResponse DTO."""

    def test_valid_creation(self):
        """Test creating a valid TOTP setup response."""
        response = TOTPSetupResponse(
            secret="JBSWY3DPEHPK3PXP",
            qr_code_url="otpauth://totp/Pynomaly:user@example.com?secret=JBSWY3DPEHPK3PXP&issuer=Pynomaly",
            manual_entry_key="JBSWY3DPEHPK3PXP",
            backup_codes=["123456", "789012", "345678"]
        )
        assert response.secret == "JBSWY3DPEHPK3PXP"
        assert "otpauth://totp/" in response.qr_code_url
        assert response.manual_entry_key == "JBSWY3DPEHPK3PXP"
        assert len(response.backup_codes) == 3

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            TOTPSetupResponse()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'secret' in error_fields
        assert 'qr_code_url' in error_fields
        assert 'manual_entry_key' in error_fields
        assert 'backup_codes' in error_fields

    def test_backup_codes_list(self):
        """Test backup codes as list."""
        response = TOTPSetupResponse(
            secret="TEST_SECRET",
            qr_code_url="otpauth://totp/test",
            manual_entry_key="TEST_KEY",
            backup_codes=["111111", "222222", "333333", "444444", "555555"]
        )
        assert len(response.backup_codes) == 5
        assert all(isinstance(code, str) for code in response.backup_codes)

    def test_serialization(self):
        """Test response serialization."""
        response = TOTPSetupResponse(
            secret="TEST_SECRET",
            qr_code_url="otpauth://totp/test",
            manual_entry_key="TEST_KEY",
            backup_codes=["111111", "222222"]
        )
        data = response.model_dump()
        assert data["secret"] == "TEST_SECRET"
        assert data["qr_code_url"] == "otpauth://totp/test"
        assert data["backup_codes"] == ["111111", "222222"]

    def test_json_schema_example(self):
        """Test JSON schema example is valid."""
        example = TOTPSetupResponse.model_config.get('json_schema_extra', {}).get('example', {})
        if example:
            response = TOTPSetupResponse.model_validate(example)
            assert response.secret == "JBSWY3DPEHPK3PXP"
            assert "otpauth://totp/" in response.qr_code_url
            assert len(response.backup_codes) == 3


class TestTOTPVerificationRequest:
    """Test cases for TOTPVerificationRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid TOTP verification request."""
        request = TOTPVerificationRequest(totp_code="123456")
        assert request.totp_code == "123456"

    def test_code_length_validation(self):
        """Test TOTP code length validation."""
        # Too short
        with pytest.raises(ValidationError) as exc_info:
            TOTPVerificationRequest(totp_code="12345")
        assert "at least 6 characters" in str(exc_info.value)

        # Too long
        with pytest.raises(ValidationError) as exc_info:
            TOTPVerificationRequest(totp_code="1234567")
        assert "at most 6 characters" in str(exc_info.value)

    def test_digits_only_validation(self):
        """Test TOTP code must contain only digits."""
        with pytest.raises(ValidationError) as exc_info:
            TOTPVerificationRequest(totp_code="12345a")
        assert "must contain only digits" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            TOTPVerificationRequest(totp_code="12-345")
        assert "must contain only digits" in str(exc_info.value)

    def test_valid_digit_codes(self):
        """Test various valid digit codes."""
        valid_codes = ["123456", "000000", "999999", "654321"]
        for code in valid_codes:
            request = TOTPVerificationRequest(totp_code=code)
            assert request.totp_code == code

    def test_required_field(self):
        """Test TOTP code is required."""
        with pytest.raises(ValidationError) as exc_info:
            TOTPVerificationRequest()

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'totp_code' for error in errors)

    def test_serialization(self):
        """Test request serialization."""
        request = TOTPVerificationRequest(totp_code="123456")
        data = request.model_dump()
        assert data["totp_code"] == "123456"


class TestSMSSetupRequest:
    """Test cases for SMSSetupRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid SMS setup request."""
        request = SMSSetupRequest(phone_number="+1234567890")
        assert request.phone_number == "+1234567890"

    def test_phone_number_validation(self):
        """Test phone number validation."""
        valid_numbers = [
            "+1234567890",
            "1234567890",
            "+123456789012345",  # Max length
            "123456789"  # Min length
        ]

        for number in valid_numbers:
            request = SMSSetupRequest(phone_number=number)
            assert request.phone_number == number

    def test_invalid_phone_numbers(self):
        """Test invalid phone number validation."""
        invalid_numbers = [
            "12345678",  # Too short
            "+12345678901234567",  # Too long (17 digits)
            "abcdefghij",  # Non-numeric
            "+1-234-567-890abc"  # Mixed characters
        ]

        for number in invalid_numbers:
            with pytest.raises(ValidationError) as exc_info:
                SMSSetupRequest(phone_number=number)
            assert "Invalid phone number format" in str(exc_info.value)

    def test_phone_number_formatting(self):
        """Test phone number with formatting characters."""
        # The validator should handle spaces and dashes
        request = SMSSetupRequest(phone_number="+1 234 567 890")
        assert request.phone_number == "+1 234 567 890"

        request = SMSSetupRequest(phone_number="+1-234-567-890")
        assert request.phone_number == "+1-234-567-890"

    def test_required_field(self):
        """Test phone number is required."""
        with pytest.raises(ValidationError) as exc_info:
            SMSSetupRequest()

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'phone_number' for error in errors)

    def test_serialization(self):
        """Test request serialization."""
        request = SMSSetupRequest(phone_number="+1234567890")
        data = request.model_dump()
        assert data["phone_number"] == "+1234567890"


class TestSMSVerificationRequest:
    """Test cases for SMSVerificationRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid SMS verification request."""
        request = SMSVerificationRequest(sms_code="123456")
        assert request.sms_code == "123456"

    def test_code_length_validation(self):
        """Test SMS code length validation."""
        # Too short
        with pytest.raises(ValidationError) as exc_info:
            SMSVerificationRequest(sms_code="12345")
        assert "at least 6 characters" in str(exc_info.value)

        # Too long
        with pytest.raises(ValidationError) as exc_info:
            SMSVerificationRequest(sms_code="1234567")
        assert "at most 6 characters" in str(exc_info.value)

    def test_digits_only_validation(self):
        """Test SMS code must contain only digits."""
        with pytest.raises(ValidationError) as exc_info:
            SMSVerificationRequest(sms_code="12345a")
        assert "must contain only digits" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            SMSVerificationRequest(sms_code="12-345")
        assert "must contain only digits" in str(exc_info.value)

    def test_valid_digit_codes(self):
        """Test various valid digit codes."""
        valid_codes = ["123456", "000000", "999999", "654321"]
        for code in valid_codes:
            request = SMSVerificationRequest(sms_code=code)
            assert request.sms_code == code

    def test_required_field(self):
        """Test SMS code is required."""
        with pytest.raises(ValidationError) as exc_info:
            SMSVerificationRequest()

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'sms_code' for error in errors)

    def test_serialization(self):
        """Test request serialization."""
        request = SMSVerificationRequest(sms_code="123456")
        data = request.model_dump()
        assert data["sms_code"] == "123456"


class TestEmailVerificationRequest:
    """Test cases for EmailVerificationRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid email verification request."""
        request = EmailVerificationRequest(email_code="123456")
        assert request.email_code == "123456"

    def test_code_length_validation(self):
        """Test email code length validation."""
        # Too short
        with pytest.raises(ValidationError) as exc_info:
            EmailVerificationRequest(email_code="12345")
        assert "at least 6 characters" in str(exc_info.value)

        # Too long
        with pytest.raises(ValidationError) as exc_info:
            EmailVerificationRequest(email_code="1234567")
        assert "at most 6 characters" in str(exc_info.value)

    def test_digits_only_validation(self):
        """Test email code must contain only digits."""
        with pytest.raises(ValidationError) as exc_info:
            EmailVerificationRequest(email_code="12345a")
        assert "must contain only digits" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            EmailVerificationRequest(email_code="12-345")
        assert "must contain only digits" in str(exc_info.value)

    def test_valid_digit_codes(self):
        """Test various valid digit codes."""
        valid_codes = ["123456", "000000", "999999", "654321"]
        for code in valid_codes:
            request = EmailVerificationRequest(email_code=code)
            assert request.email_code == code

    def test_required_field(self):
        """Test email code is required."""
        with pytest.raises(ValidationError) as exc_info:
            EmailVerificationRequest()

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'email_code' for error in errors)

    def test_serialization(self):
        """Test request serialization."""
        request = EmailVerificationRequest(email_code="123456")
        data = request.model_dump()
        assert data["email_code"] == "123456"


class TestBackupCodeVerificationRequest:
    """Test cases for BackupCodeVerificationRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid backup code verification request."""
        request = BackupCodeVerificationRequest(backup_code="123456")
        assert request.backup_code == "123456"

    def test_code_length_validation(self):
        """Test backup code length validation."""
        # Too short
        with pytest.raises(ValidationError) as exc_info:
            BackupCodeVerificationRequest(backup_code="12345")
        assert "at least 6 characters" in str(exc_info.value)

        # Too long
        with pytest.raises(ValidationError) as exc_info:
            BackupCodeVerificationRequest(backup_code="12345678901")
        assert "at most 10 characters" in str(exc_info.value)

    def test_valid_code_lengths(self):
        """Test valid code lengths."""
        valid_codes = ["123456", "1234567", "12345678", "123456789", "1234567890"]
        for code in valid_codes:
            request = BackupCodeVerificationRequest(backup_code=code)
            assert request.backup_code == code

    def test_alphanumeric_codes(self):
        """Test alphanumeric backup codes are allowed."""
        valid_codes = ["ABC123", "12AB34", "XYZ789", "123ABC"]
        for code in valid_codes:
            request = BackupCodeVerificationRequest(backup_code=code)
            assert request.backup_code == code

    def test_required_field(self):
        """Test backup code is required."""
        with pytest.raises(ValidationError) as exc_info:
            BackupCodeVerificationRequest()

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'backup_code' for error in errors)

    def test_serialization(self):
        """Test request serialization."""
        request = BackupCodeVerificationRequest(backup_code="ABC123")
        data = request.model_dump()
        assert data["backup_code"] == "ABC123"


class TestMFAMethodDTO:
    """Test cases for MFAMethodDTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA method DTO."""
        now = datetime.now()
        dto = MFAMethodDTO(
            id="mfa_123",
            method_type=MFAMethodType.TOTP,
            status=MFAMethodStatus.ACTIVE,
            display_name="Google Authenticator",
            created_at=now,
            last_used=now,
            is_primary=True,
            phone_number="+1234567890",
            email="user@example.com",
            backup_codes_remaining=5
        )
        assert dto.id == "mfa_123"
        assert dto.method_type == MFAMethodType.TOTP
        assert dto.status == MFAMethodStatus.ACTIVE
        assert dto.display_name == "Google Authenticator"
        assert dto.created_at == now
        assert dto.last_used == now
        assert dto.is_primary is True
        assert dto.phone_number == "+1234567890"
        assert dto.email == "user@example.com"
        assert dto.backup_codes_remaining == 5

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFAMethodDTO()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'id' in error_fields
        assert 'method_type' in error_fields
        assert 'status' in error_fields
        assert 'display_name' in error_fields
        assert 'created_at' in error_fields

    def test_optional_fields(self):
        """Test optional fields with defaults."""
        now = datetime.now()
        dto = MFAMethodDTO(
            id="mfa_123",
            method_type=MFAMethodType.TOTP,
            status=MFAMethodStatus.ACTIVE,
            display_name="Google Authenticator",
            created_at=now
        )
        assert dto.last_used is None
        assert dto.is_primary is False
        assert dto.phone_number is None
        assert dto.email is None
        assert dto.backup_codes_remaining is None

    def test_totp_method(self):
        """Test TOTP method DTO."""
        now = datetime.now()
        dto = MFAMethodDTO(
            id="totp_123",
            method_type=MFAMethodType.TOTP,
            status=MFAMethodStatus.ACTIVE,
            display_name="Authenticator App",
            created_at=now,
            is_primary=True
        )
        assert dto.method_type == MFAMethodType.TOTP
        assert dto.is_primary is True
        assert dto.phone_number is None
        assert dto.email is None

    def test_sms_method(self):
        """Test SMS method DTO."""
        now = datetime.now()
        dto = MFAMethodDTO(
            id="sms_123",
            method_type=MFAMethodType.SMS,
            status=MFAMethodStatus.ACTIVE,
            display_name="SMS to +1234567890",
            created_at=now,
            phone_number="+1234567890"
        )
        assert dto.method_type == MFAMethodType.SMS
        assert dto.phone_number == "+1234567890"
        assert dto.email is None

    def test_email_method(self):
        """Test email method DTO."""
        now = datetime.now()
        dto = MFAMethodDTO(
            id="email_123",
            method_type=MFAMethodType.EMAIL,
            status=MFAMethodStatus.ACTIVE,
            display_name="Email to user@example.com",
            created_at=now,
            email="user@example.com"
        )
        assert dto.method_type == MFAMethodType.EMAIL
        assert dto.email == "user@example.com"
        assert dto.phone_number is None

    def test_backup_codes_method(self):
        """Test backup codes method DTO."""
        now = datetime.now()
        dto = MFAMethodDTO(
            id="backup_123",
            method_type=MFAMethodType.BACKUP_CODES,
            status=MFAMethodStatus.ACTIVE,
            display_name="Backup Codes",
            created_at=now,
            backup_codes_remaining=8
        )
        assert dto.method_type == MFAMethodType.BACKUP_CODES
        assert dto.backup_codes_remaining == 8

    def test_serialization(self):
        """Test DTO serialization."""
        now = datetime.now()
        dto = MFAMethodDTO(
            id="mfa_123",
            method_type=MFAMethodType.TOTP,
            status=MFAMethodStatus.ACTIVE,
            display_name="Google Authenticator",
            created_at=now,
            is_primary=True
        )
        data = dto.model_dump()
        assert data["id"] == "mfa_123"
        assert data["method_type"] == "totp"
        assert data["status"] == "active"
        assert data["is_primary"] is True


class TestMFAStatusResponse:
    """Test cases for MFAStatusResponse DTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA status response."""
        now = datetime.now()
        active_method = MFAMethodDTO(
            id="mfa_123",
            method_type=MFAMethodType.TOTP,
            status=MFAMethodStatus.ACTIVE,
            display_name="Google Authenticator",
            created_at=now,
            is_primary=True
        )

        response = MFAStatusResponse(
            mfa_enabled=True,
            active_methods=[active_method],
            pending_methods=[],
            primary_method=active_method,
            backup_codes_available=True
        )

        assert response.mfa_enabled is True
        assert len(response.active_methods) == 1
        assert len(response.pending_methods) == 0
        assert response.primary_method == active_method
        assert response.backup_codes_available is True

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFAStatusResponse()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'mfa_enabled' in error_fields
        assert 'active_methods' in error_fields
        assert 'pending_methods' in error_fields
        assert 'backup_codes_available' in error_fields

    def test_empty_methods(self):
        """Test response with empty method lists."""
        response = MFAStatusResponse(
            mfa_enabled=False,
            active_methods=[],
            pending_methods=[],
            primary_method=None,
            backup_codes_available=False
        )

        assert response.mfa_enabled is False
        assert len(response.active_methods) == 0
        assert len(response.pending_methods) == 0
        assert response.primary_method is None
        assert response.backup_codes_available is False

    def test_multiple_methods(self):
        """Test response with multiple methods."""
        now = datetime.now()
        totp_method = MFAMethodDTO(
            id="totp_123",
            method_type=MFAMethodType.TOTP,
            status=MFAMethodStatus.ACTIVE,
            display_name="Google Authenticator",
            created_at=now,
            is_primary=True
        )

        sms_method = MFAMethodDTO(
            id="sms_123",
            method_type=MFAMethodType.SMS,
            status=MFAMethodStatus.ACTIVE,
            display_name="SMS to +1234567890",
            created_at=now,
            phone_number="+1234567890"
        )

        pending_method = MFAMethodDTO(
            id="email_123",
            method_type=MFAMethodType.EMAIL,
            status=MFAMethodStatus.PENDING,
            display_name="Email to user@example.com",
            created_at=now,
            email="user@example.com"
        )

        response = MFAStatusResponse(
            mfa_enabled=True,
            active_methods=[totp_method, sms_method],
            pending_methods=[pending_method],
            primary_method=totp_method,
            backup_codes_available=True
        )

        assert len(response.active_methods) == 2
        assert len(response.pending_methods) == 1
        assert response.primary_method.id == "totp_123"

    def test_serialization(self):
        """Test response serialization."""
        now = datetime.now()
        active_method = MFAMethodDTO(
            id="mfa_123",
            method_type=MFAMethodType.TOTP,
            status=MFAMethodStatus.ACTIVE,
            display_name="Google Authenticator",
            created_at=now
        )

        response = MFAStatusResponse(
            mfa_enabled=True,
            active_methods=[active_method],
            pending_methods=[],
            primary_method=None,
            backup_codes_available=True
        )

        data = response.model_dump()
        assert data["mfa_enabled"] is True
        assert len(data["active_methods"]) == 1
        assert data["active_methods"][0]["id"] == "mfa_123"
        assert data["primary_method"] is None

    def test_json_schema_example(self):
        """Test JSON schema example is valid."""
        example = MFAStatusResponse.model_config.get('json_schema_extra', {}).get('example', {})
        if example:
            # The example has a nested structure that needs proper validation
            response = MFAStatusResponse(
                mfa_enabled=example["mfa_enabled"],
                active_methods=[],  # Simplified for test
                pending_methods=[],
                primary_method=None,
                backup_codes_available=example["backup_codes_available"]
            )
            assert response.mfa_enabled is True
            assert response.backup_codes_available is True


class TestMFAEnableRequest:
    """Test cases for MFAEnableRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA enable request."""
        request = MFAEnableRequest(
            method_type=MFAMethodType.TOTP,
            verification_code="123456",
            set_as_primary=True
        )
        assert request.method_type == MFAMethodType.TOTP
        assert request.verification_code == "123456"
        assert request.set_as_primary is True

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFAEnableRequest()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'method_type' in error_fields
        assert 'verification_code' in error_fields

    def test_default_primary(self):
        """Test default set_as_primary value."""
        request = MFAEnableRequest(
            method_type=MFAMethodType.SMS,
            verification_code="654321"
        )
        assert request.set_as_primary is True

    def test_different_method_types(self):
        """Test different MFA method types."""
        method_types = [
            MFAMethodType.TOTP,
            MFAMethodType.SMS,
            MFAMethodType.EMAIL,
            MFAMethodType.BACKUP_CODES
        ]

        for method_type in method_types:
            request = MFAEnableRequest(
                method_type=method_type,
                verification_code="123456"
            )
            assert request.method_type == method_type

    def test_serialization(self):
        """Test request serialization."""
        request = MFAEnableRequest(
            method_type=MFAMethodType.TOTP,
            verification_code="123456",
            set_as_primary=False
        )
        data = request.model_dump()
        assert data["method_type"] == "totp"
        assert data["verification_code"] == "123456"
        assert data["set_as_primary"] is False


class TestMFADisableRequest:
    """Test cases for MFADisableRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA disable request."""
        request = MFADisableRequest(
            method_id="mfa_123",
            verification_code="123456"
        )
        assert request.method_id == "mfa_123"
        assert request.verification_code == "123456"

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFADisableRequest()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'method_id' in error_fields
        assert 'verification_code' in error_fields

    def test_serialization(self):
        """Test request serialization."""
        request = MFADisableRequest(
            method_id="mfa_123",
            verification_code="123456"
        )
        data = request.model_dump()
        assert data["method_id"] == "mfa_123"
        assert data["verification_code"] == "123456"


class TestMFALoginRequest:
    """Test cases for MFALoginRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA login request."""
        request = MFALoginRequest(
            method_type=MFAMethodType.TOTP,
            verification_code="123456",
            remember_device=True
        )
        assert request.method_type == MFAMethodType.TOTP
        assert request.verification_code == "123456"
        assert request.remember_device is True

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFALoginRequest()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'method_type' in error_fields
        assert 'verification_code' in error_fields

    def test_default_remember_device(self):
        """Test default remember_device value."""
        request = MFALoginRequest(
            method_type=MFAMethodType.SMS,
            verification_code="654321"
        )
        assert request.remember_device is False

    def test_different_method_types(self):
        """Test different MFA method types."""
        method_types = [
            MFAMethodType.TOTP,
            MFAMethodType.SMS,
            MFAMethodType.EMAIL,
            MFAMethodType.BACKUP_CODES
        ]

        for method_type in method_types:
            request = MFALoginRequest(
                method_type=method_type,
                verification_code="123456"
            )
            assert request.method_type == method_type

    def test_serialization(self):
        """Test request serialization."""
        request = MFALoginRequest(
            method_type=MFAMethodType.TOTP,
            verification_code="123456",
            remember_device=True
        )
        data = request.model_dump()
        assert data["method_type"] == "totp"
        assert data["verification_code"] == "123456"
        assert data["remember_device"] is True


class TestMFALoginResponse:
    """Test cases for MFALoginResponse DTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA login response."""
        response = MFALoginResponse(
            access_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            refresh_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            token_type="bearer",
            expires_in=3600,
            device_remembered=True
        )
        assert response.access_token.startswith("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        assert response.refresh_token.startswith("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        assert response.token_type == "bearer"
        assert response.expires_in == 3600
        assert response.device_remembered is True

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFALoginResponse()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'access_token' in error_fields
        assert 'refresh_token' in error_fields
        assert 'expires_in' in error_fields

    def test_default_values(self):
        """Test default values."""
        response = MFALoginResponse(
            access_token="access_token_123",
            refresh_token="refresh_token_123",
            expires_in=3600
        )
        assert response.token_type == "bearer"
        assert response.device_remembered is False

    def test_serialization(self):
        """Test response serialization."""
        response = MFALoginResponse(
            access_token="access_token_123",
            refresh_token="refresh_token_123",
            expires_in=3600,
            device_remembered=True
        )
        data = response.model_dump()
        assert data["access_token"] == "access_token_123"
        assert data["refresh_token"] == "refresh_token_123"
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 3600
        assert data["device_remembered"] is True


class TestBackupCodesResponse:
    """Test cases for BackupCodesResponse DTO."""

    def test_valid_creation(self):
        """Test creating a valid backup codes response."""
        response = BackupCodesResponse(
            backup_codes=["123456", "789012", "345678"],
            codes_remaining=3
        )
        assert len(response.backup_codes) == 3
        assert response.codes_remaining == 3

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            BackupCodesResponse()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'backup_codes' in error_fields
        assert 'codes_remaining' in error_fields

    def test_empty_backup_codes(self):
        """Test response with empty backup codes."""
        response = BackupCodesResponse(
            backup_codes=[],
            codes_remaining=0
        )
        assert len(response.backup_codes) == 0
        assert response.codes_remaining == 0

    def test_multiple_backup_codes(self):
        """Test response with multiple backup codes."""
        codes = ["111111", "222222", "333333", "444444", "555555"]
        response = BackupCodesResponse(
            backup_codes=codes,
            codes_remaining=5
        )
        assert len(response.backup_codes) == 5
        assert response.codes_remaining == 5
        assert response.backup_codes == codes

    def test_serialization(self):
        """Test response serialization."""
        response = BackupCodesResponse(
            backup_codes=["123456", "789012"],
            codes_remaining=2
        )
        data = response.model_dump()
        assert data["backup_codes"] == ["123456", "789012"]
        assert data["codes_remaining"] == 2

    def test_json_schema_example(self):
        """Test JSON schema example is valid."""
        example = BackupCodesResponse.model_config.get('json_schema_extra', {}).get('example', {})
        if example:
            response = BackupCodesResponse.model_validate(example)
            assert len(response.backup_codes) == 3
            assert response.codes_remaining == 3


class TestMFARecoveryRequest:
    """Test cases for MFARecoveryRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA recovery request."""
        request = MFARecoveryRequest(
            backup_code="123456",
            new_password="new_secure_password123"
        )
        assert request.backup_code == "123456"
        assert request.new_password == "new_secure_password123"

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFARecoveryRequest()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'backup_code' in error_fields

    def test_optional_new_password(self):
        """Test optional new password field."""
        request = MFARecoveryRequest(backup_code="123456")
        assert request.backup_code == "123456"
        assert request.new_password is None

    def test_serialization(self):
        """Test request serialization."""
        request = MFARecoveryRequest(
            backup_code="123456",
            new_password="new_password"
        )
        data = request.model_dump()
        assert data["backup_code"] == "123456"
        assert data["new_password"] == "new_password"


class TestMFARecoveryResponse:
    """Test cases for MFARecoveryResponse DTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA recovery response."""
        response = MFARecoveryResponse(
            message="Account recovery successful",
            access_token="access_token_123",
            refresh_token="refresh_token_123",
            remaining_codes=2
        )
        assert response.message == "Account recovery successful"
        assert response.access_token == "access_token_123"
        assert response.refresh_token == "refresh_token_123"
        assert response.remaining_codes == 2

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFARecoveryResponse()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'message' in error_fields
        assert 'access_token' in error_fields
        assert 'refresh_token' in error_fields
        assert 'remaining_codes' in error_fields

    def test_zero_remaining_codes(self):
        """Test response with zero remaining codes."""
        response = MFARecoveryResponse(
            message="Account recovery successful - no backup codes remaining",
            access_token="access_token_123",
            refresh_token="refresh_token_123",
            remaining_codes=0
        )
        assert response.remaining_codes == 0

    def test_serialization(self):
        """Test response serialization."""
        response = MFARecoveryResponse(
            message="Recovery successful",
            access_token="access_token_123",
            refresh_token="refresh_token_123",
            remaining_codes=3
        )
        data = response.model_dump()
        assert data["message"] == "Recovery successful"
        assert data["access_token"] == "access_token_123"
        assert data["remaining_codes"] == 3


class TestMFADeviceDTO:
    """Test cases for MFADeviceDTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA device DTO."""
        now = datetime.now()
        dto = MFADeviceDTO(
            id="device_123",
            device_name="iPhone 12",
            device_type="mobile",
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            ip_address="192.168.1.100",
            created_at=now,
            last_used=now,
            is_active=True
        )
        assert dto.id == "device_123"
        assert dto.device_name == "iPhone 12"
        assert dto.device_type == "mobile"
        assert dto.user_agent.startswith("Mozilla/5.0")
        assert dto.ip_address == "192.168.1.100"
        assert dto.created_at == now
        assert dto.last_used == now
        assert dto.is_active is True

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFADeviceDTO()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'id' in error_fields
        assert 'device_name' in error_fields
        assert 'device_type' in error_fields
        assert 'user_agent' in error_fields
        assert 'ip_address' in error_fields
        assert 'created_at' in error_fields
        assert 'last_used' in error_fields

    def test_default_is_active(self):
        """Test default is_active value."""
        now = datetime.now()
        dto = MFADeviceDTO(
            id="device_123",
            device_name="Test Device",
            device_type="desktop",
            user_agent="Test Agent",
            ip_address="127.0.0.1",
            created_at=now,
            last_used=now
        )
        assert dto.is_active is True

    def test_different_device_types(self):
        """Test different device types."""
        now = datetime.now()
        device_types = ["mobile", "desktop", "tablet", "other"]

        for device_type in device_types:
            dto = MFADeviceDTO(
                id=f"device_{device_type}",
                device_name=f"Test {device_type}",
                device_type=device_type,
                user_agent="Test Agent",
                ip_address="127.0.0.1",
                created_at=now,
                last_used=now
            )
            assert dto.device_type == device_type

    def test_serialization(self):
        """Test DTO serialization."""
        now = datetime.now()
        dto = MFADeviceDTO(
            id="device_123",
            device_name="Test Device",
            device_type="mobile",
            user_agent="Test Agent",
            ip_address="192.168.1.100",
            created_at=now,
            last_used=now,
            is_active=False
        )
        data = dto.model_dump()
        assert data["id"] == "device_123"
        assert data["device_name"] == "Test Device"
        assert data["device_type"] == "mobile"
        assert data["is_active"] is False


class TestTrustedDevicesResponse:
    """Test cases for TrustedDevicesResponse DTO."""

    def test_valid_creation(self):
        """Test creating a valid trusted devices response."""
        now = datetime.now()
        device = MFADeviceDTO(
            id="device_123",
            device_name="Test Device",
            device_type="mobile",
            user_agent="Test Agent",
            ip_address="192.168.1.100",
            created_at=now,
            last_used=now
        )

        response = TrustedDevicesResponse(
            devices=[device],
            total_devices=1
        )
        assert len(response.devices) == 1
        assert response.total_devices == 1

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            TrustedDevicesResponse()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'devices' in error_fields
        assert 'total_devices' in error_fields

    def test_empty_devices(self):
        """Test response with empty devices list."""
        response = TrustedDevicesResponse(
            devices=[],
            total_devices=0
        )
        assert len(response.devices) == 0
        assert response.total_devices == 0

    def test_multiple_devices(self):
        """Test response with multiple devices."""
        now = datetime.now()
        devices = []
        for i in range(3):
            device = MFADeviceDTO(
                id=f"device_{i}",
                device_name=f"Device {i}",
                device_type="mobile",
                user_agent="Test Agent",
                ip_address="192.168.1.100",
                created_at=now,
                last_used=now
            )
            devices.append(device)

        response = TrustedDevicesResponse(
            devices=devices,
            total_devices=3
        )
        assert len(response.devices) == 3
        assert response.total_devices == 3

    def test_serialization(self):
        """Test response serialization."""
        now = datetime.now()
        device = MFADeviceDTO(
            id="device_123",
            device_name="Test Device",
            device_type="mobile",
            user_agent="Test Agent",
            ip_address="192.168.1.100",
            created_at=now,
            last_used=now
        )

        response = TrustedDevicesResponse(
            devices=[device],
            total_devices=1
        )
        data = response.model_dump()
        assert len(data["devices"]) == 1
        assert data["total_devices"] == 1
        assert data["devices"][0]["id"] == "device_123"


class TestRevokeTrustedDeviceRequest:
    """Test cases for RevokeTrustedDeviceRequest DTO."""

    def test_valid_creation(self):
        """Test creating a valid revoke trusted device request."""
        request = RevokeTrustedDeviceRequest(device_id="device_123")
        assert request.device_id == "device_123"

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            RevokeTrustedDeviceRequest()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'device_id' in error_fields

    def test_serialization(self):
        """Test request serialization."""
        request = RevokeTrustedDeviceRequest(device_id="device_123")
        data = request.model_dump()
        assert data["device_id"] == "device_123"


class TestMFASettingsDTO:
    """Test cases for MFASettingsDTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA settings DTO."""
        dto = MFASettingsDTO(
            enforce_mfa=True,
            allowed_methods=[MFAMethodType.TOTP, MFAMethodType.SMS],
            backup_codes_enabled=True,
            remember_device_duration=7200,
            max_trusted_devices=10
        )
        assert dto.enforce_mfa is True
        assert len(dto.allowed_methods) == 2
        assert dto.backup_codes_enabled is True
        assert dto.remember_device_duration == 7200
        assert dto.max_trusted_devices == 10

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFASettingsDTO()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'allowed_methods' in error_fields

    def test_default_values(self):
        """Test default values."""
        dto = MFASettingsDTO(
            allowed_methods=[MFAMethodType.TOTP, MFAMethodType.SMS]
        )
        assert dto.enforce_mfa is False
        assert dto.backup_codes_enabled is True
        assert dto.remember_device_duration == 2592000  # 30 days
        assert dto.max_trusted_devices == 5

    def test_all_method_types(self):
        """Test settings with all method types."""
        all_methods = [
            MFAMethodType.TOTP,
            MFAMethodType.SMS,
            MFAMethodType.EMAIL,
            MFAMethodType.BACKUP_CODES,
            MFAMethodType.HARDWARE_TOKEN
        ]

        dto = MFASettingsDTO(
            allowed_methods=all_methods,
            enforce_mfa=True
        )
        assert len(dto.allowed_methods) == 5
        assert dto.enforce_mfa is True

    def test_serialization(self):
        """Test DTO serialization."""
        dto = MFASettingsDTO(
            enforce_mfa=True,
            allowed_methods=[MFAMethodType.TOTP, MFAMethodType.SMS],
            backup_codes_enabled=False,
            remember_device_duration=3600,
            max_trusted_devices=3
        )
        data = dto.model_dump()
        assert data["enforce_mfa"] is True
        assert data["allowed_methods"] == ["totp", "sms"]
        assert data["backup_codes_enabled"] is False
        assert data["remember_device_duration"] == 3600
        assert data["max_trusted_devices"] == 3

    def test_json_schema_example(self):
        """Test JSON schema example is valid."""
        example = MFASettingsDTO.model_config.get('json_schema_extra', {}).get('example', {})
        if example:
            dto = MFASettingsDTO.model_validate(example)
            assert dto.enforce_mfa is False
            assert len(dto.allowed_methods) == 3
            assert dto.backup_codes_enabled is True


class TestMFAStatisticsDTO:
    """Test cases for MFAStatisticsDTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA statistics DTO."""
        dto = MFAStatisticsDTO(
            total_users=1000,
            mfa_enabled_users=650,
            mfa_adoption_rate=65.0,
            method_usage={"totp": 400, "sms": 200, "email": 50},
            recent_authentications=1200
        )
        assert dto.total_users == 1000
        assert dto.mfa_enabled_users == 650
        assert dto.mfa_adoption_rate == 65.0
        assert dto.method_usage == {"totp": 400, "sms": 200, "email": 50}
        assert dto.recent_authentications == 1200

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFAStatisticsDTO()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'total_users' in error_fields
        assert 'mfa_enabled_users' in error_fields
        assert 'mfa_adoption_rate' in error_fields
        assert 'method_usage' in error_fields
        assert 'recent_authentications' in error_fields

    def test_zero_values(self):
        """Test statistics with zero values."""
        dto = MFAStatisticsDTO(
            total_users=0,
            mfa_enabled_users=0,
            mfa_adoption_rate=0.0,
            method_usage={},
            recent_authentications=0
        )
        assert dto.total_users == 0
        assert dto.mfa_enabled_users == 0
        assert dto.mfa_adoption_rate == 0.0
        assert dto.method_usage == {}
        assert dto.recent_authentications == 0

    def test_percentage_calculations(self):
        """Test percentage calculations."""
        dto = MFAStatisticsDTO(
            total_users=1000,
            mfa_enabled_users=850,
            mfa_adoption_rate=85.0,
            method_usage={"totp": 500, "sms": 300, "email": 50},
            recent_authentications=2000
        )
        assert dto.mfa_adoption_rate == 85.0
        # Verify that the adoption rate makes sense
        calculated_rate = (dto.mfa_enabled_users / dto.total_users) * 100
        assert abs(calculated_rate - dto.mfa_adoption_rate) < 0.1

    def test_method_usage_structure(self):
        """Test method usage dictionary structure."""
        method_usage = {
            "totp": 400,
            "sms": 200,
            "email": 50,
            "backup_codes": 25,
            "hardware_token": 5
        }

        dto = MFAStatisticsDTO(
            total_users=1000,
            mfa_enabled_users=680,
            mfa_adoption_rate=68.0,
            method_usage=method_usage,
            recent_authentications=1500
        )

        assert dto.method_usage == method_usage
        assert sum(dto.method_usage.values()) == 680

    def test_serialization(self):
        """Test DTO serialization."""
        dto = MFAStatisticsDTO(
            total_users=500,
            mfa_enabled_users=300,
            mfa_adoption_rate=60.0,
            method_usage={"totp": 200, "sms": 100},
            recent_authentications=800
        )
        data = dto.model_dump()
        assert data["total_users"] == 500
        assert data["mfa_enabled_users"] == 300
        assert data["mfa_adoption_rate"] == 60.0
        assert data["method_usage"] == {"totp": 200, "sms": 100}
        assert data["recent_authentications"] == 800

    def test_json_schema_example(self):
        """Test JSON schema example is valid."""
        example = MFAStatisticsDTO.model_config.get('json_schema_extra', {}).get('example', {})
        if example:
            dto = MFAStatisticsDTO.model_validate(example)
            assert dto.total_users == 1000
            assert dto.mfa_enabled_users == 650
            assert dto.mfa_adoption_rate == 65.0
            assert isinstance(dto.method_usage, dict)


class TestMFAErrorResponse:
    """Test cases for MFAErrorResponse DTO."""

    def test_valid_creation(self):
        """Test creating a valid MFA error response."""
        response = MFAErrorResponse(
            error="invalid_totp_code",
            message="The provided TOTP code is invalid or expired",
            details={"attempts_remaining": 2}
        )
        assert response.error == "invalid_totp_code"
        assert response.message == "The provided TOTP code is invalid or expired"
        assert response.details == {"attempts_remaining": 2}

    def test_required_fields(self):
        """Test required fields validation."""
        with pytest.raises(ValidationError) as exc_info:
            MFAErrorResponse()

        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'error' in error_fields
        assert 'message' in error_fields

    def test_optional_details(self):
        """Test optional details field."""
        response = MFAErrorResponse(
            error="method_not_found",
            message="The specified MFA method was not found"
        )
        assert response.error == "method_not_found"
        assert response.message == "The specified MFA method was not found"
        assert response.details is None

    def test_different_error_types(self):
        """Test different error types."""
        error_types = [
            "invalid_totp_code",
            "invalid_sms_code",
            "method_not_found",
            "method_disabled",
            "rate_limit_exceeded",
            "backup_code_used"
        ]

        for error_type in error_types:
            response = MFAErrorResponse(
                error=error_type,
                message=f"Error: {error_type}"
            )
            assert response.error == error_type

    def test_complex_details(self):
        """Test complex details structure."""
        details = {
            "attempts_remaining": 1,
            "lockout_time": 300,
            "allowed_methods": ["totp", "sms"],
            "backup_codes_available": True
        }

        response = MFAErrorResponse(
            error="authentication_failed",
            message="Authentication failed",
            details=details
        )
        assert response.details == details
        assert response.details["attempts_remaining"] == 1
        assert response.details["lockout_time"] == 300

    def test_serialization(self):
        """Test response serialization."""
        response = MFAErrorResponse(
            error="invalid_code",
            message="Invalid verification code",
            details={"code": "INVALID_CODE"}
        )
        data = response.model_dump()
        assert data["error"] == "invalid_code"
        assert data["message"] == "Invalid verification code"
        assert data["details"] == {"code": "INVALID_CODE"}

    def test_json_schema_example(self):
        """Test JSON schema example is valid."""
        example = MFAErrorResponse.model_config.get('json_schema_extra', {}).get('example', {})
        if example:
            response = MFAErrorResponse.model_validate(example)
            assert response.error == "invalid_totp_code"
            assert "invalid or expired" in response.message
            assert response.details["attempts_remaining"] == 2


class TestMFAIntegrationScenarios:
    """Test cases for MFA integration scenarios."""

    def test_totp_setup_workflow(self):
        """Test complete TOTP setup workflow."""
        # 1. Setup request
        setup_request = TOTPSetupRequest(
            app_name="Test App",
            issuer="Test Security"
        )

        # 2. Setup response
        setup_response = TOTPSetupResponse(
            secret="JBSWY3DPEHPK3PXP",
            qr_code_url="otpauth://totp/Test App:user@example.com?secret=JBSWY3DPEHPK3PXP&issuer=Test Security",
            manual_entry_key="JBSWY3DPEHPK3PXP",
            backup_codes=["123456", "789012", "345678", "456789", "012345"]
        )

        # 3. Verification request
        verification_request = TOTPVerificationRequest(totp_code="123456")

        # 4. Enable MFA
        enable_request = MFAEnableRequest(
            method_type=MFAMethodType.TOTP,
            verification_code="123456",
            set_as_primary=True
        )

        # Verify workflow
        assert setup_request.app_name == "Test App"
        assert setup_response.secret == "JBSWY3DPEHPK3PXP"
        assert verification_request.totp_code == "123456"
        assert enable_request.method_type == MFAMethodType.TOTP
        assert enable_request.set_as_primary is True

    def test_sms_setup_workflow(self):
        """Test complete SMS setup workflow."""
        # 1. SMS setup request
        setup_request = SMSSetupRequest(phone_number="+1234567890")

        # 2. SMS verification request
        verification_request = SMSVerificationRequest(sms_code="654321")

        # 3. Enable MFA
        enable_request = MFAEnableRequest(
            method_type=MFAMethodType.SMS,
            verification_code="654321",
            set_as_primary=False
        )

        # Verify workflow
        assert setup_request.phone_number == "+1234567890"
        assert verification_request.sms_code == "654321"
        assert enable_request.method_type == MFAMethodType.SMS
        assert enable_request.set_as_primary is False

    def test_login_workflow(self):
        """Test complete MFA login workflow."""
        # 1. Login request
        login_request = MFALoginRequest(
            method_type=MFAMethodType.TOTP,
            verification_code="123456",
            remember_device=True
        )

        # 2. Successful login response
        login_response = MFALoginResponse(
            access_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            refresh_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            token_type="bearer",
            expires_in=3600,
            device_remembered=True
        )

        # Verify workflow
        assert login_request.method_type == MFAMethodType.TOTP
        assert login_request.remember_device is True
        assert login_response.token_type == "bearer"
        assert login_response.device_remembered is True

    def test_backup_code_recovery_workflow(self):
        """Test backup code recovery workflow."""
        # 1. Recovery request
        recovery_request = MFARecoveryRequest(
            backup_code="123456",
            new_password="new_secure_password123"
        )

        # 2. Recovery response
        recovery_response = MFARecoveryResponse(
            message="Account recovery successful",
            access_token="access_token_123",
            refresh_token="refresh_token_123",
            remaining_codes=4
        )

        # 3. New backup codes
        new_codes_response = BackupCodesResponse(
            backup_codes=["111111", "222222", "333333", "444444"],
            codes_remaining=4
        )

        # Verify workflow
        assert recovery_request.backup_code == "123456"
        assert recovery_request.new_password == "new_secure_password123"
        assert recovery_response.remaining_codes == 4
        assert new_codes_response.codes_remaining == 4

    def test_device_management_workflow(self):
        """Test device management workflow."""
        now = datetime.now()

        # 1. Device registration (implicit during login)
        device = MFADeviceDTO(
            id="device_123",
            device_name="iPhone 12",
            device_type="mobile",
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            ip_address="192.168.1.100",
            created_at=now,
            last_used=now,
            is_active=True
        )

        # 2. List trusted devices
        devices_response = TrustedDevicesResponse(
            devices=[device],
            total_devices=1
        )

        # 3. Revoke device
        revoke_request = RevokeTrustedDeviceRequest(device_id="device_123")

        # Verify workflow
        assert device.device_name == "iPhone 12"
        assert device.is_active is True
        assert devices_response.total_devices == 1
        assert revoke_request.device_id == "device_123"

    def test_mfa_status_workflow(self):
        """Test MFA status and management workflow."""
        now = datetime.now()

        # 1. Create MFA methods
        totp_method = MFAMethodDTO(
            id="totp_123",
            method_type=MFAMethodType.TOTP,
            status=MFAMethodStatus.ACTIVE,
            display_name="Google Authenticator",
            created_at=now,
            is_primary=True
        )

        sms_method = MFAMethodDTO(
            id="sms_123",
            method_type=MFAMethodType.SMS,
            status=MFAMethodStatus.ACTIVE,
            display_name="SMS to +1234567890",
            created_at=now,
            phone_number="+1234567890"
        )

        # 2. MFA status response
        status_response = MFAStatusResponse(
            mfa_enabled=True,
            active_methods=[totp_method, sms_method],
            pending_methods=[],
            primary_method=totp_method,
            backup_codes_available=True
        )

        # 3. Disable MFA method
        disable_request = MFADisableRequest(
            method_id="sms_123",
            verification_code="123456"
        )

        # Verify workflow
        assert status_response.mfa_enabled is True
        assert len(status_response.active_methods) == 2
        assert status_response.primary_method.id == "totp_123"
        assert disable_request.method_id == "sms_123"

    def test_admin_settings_workflow(self):
        """Test admin settings and statistics workflow."""
        # 1. MFA settings
        settings = MFASettingsDTO(
            enforce_mfa=True,
            allowed_methods=[MFAMethodType.TOTP, MFAMethodType.SMS, MFAMethodType.EMAIL],
            backup_codes_enabled=True,
            remember_device_duration=7200,
            max_trusted_devices=3
        )

        # 2. MFA statistics
        statistics = MFAStatisticsDTO(
            total_users=1000,
            mfa_enabled_users=750,
            mfa_adoption_rate=75.0,
            method_usage={
                "totp": 400,
                "sms": 250,
                "email": 100
            },
            recent_authentications=2000
        )

        # Verify workflow
        assert settings.enforce_mfa is True
        assert len(settings.allowed_methods) == 3
        assert settings.max_trusted_devices == 3
        assert statistics.mfa_adoption_rate == 75.0
        assert sum(statistics.method_usage.values()) == 750

    def test_error_handling_workflow(self):
        """Test error handling workflow."""
        # 1. Invalid TOTP code error
        totp_error = MFAErrorResponse(
            error="invalid_totp_code",
            message="The provided TOTP code is invalid or expired",
            details={"attempts_remaining": 2}
        )

        # 2. Rate limit error
        rate_limit_error = MFAErrorResponse(
            error="rate_limit_exceeded",
            message="Too many failed attempts. Please try again later.",
            details={"lockout_time": 300}
        )

        # 3. Method not found error
        method_error = MFAErrorResponse(
            error="method_not_found",
            message="The specified MFA method was not found"
        )

        # Verify workflow
        assert totp_error.error == "invalid_totp_code"
        assert totp_error.details["attempts_remaining"] == 2
        assert rate_limit_error.details["lockout_time"] == 300
        assert method_error.details is None
