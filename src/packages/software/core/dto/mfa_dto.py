#!/usr/bin/env python3
"""
Data Transfer Objects for Multi-Factor Authentication (MFA) functionality.
Provides DTOs for TOTP setup, SMS verification, backup codes, and MFA management.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Enum, Field, field_validator


class MFAMethodType(str):
    """Types of MFA methods available."""

    TOTP = "totp"  # Time-based One-Time Password (Google Authenticator, etc.)
    SMS = "sms"  # SMS-based verification
    EMAIL = "email"  # Email-based verification
    BACKUP_CODES = "backup_codes"  # Backup recovery codes
    HARDWARE_TOKEN = "hardware_token"  # Hardware security keys (future)


class MFAMethodStatus(str, Enum):
    """Status of MFA methods."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DISABLED = "disabled"


class TOTPSetupRequest(BaseModel):
    """Request to initiate TOTP setup."""

    app_name: str = Field(default="Software", description="Name of the application")
    issuer: str = Field(default="Software Security", description="Issuer name for TOTP")


class TOTPSetupResponse(BaseModel):
    """Response containing TOTP setup information."""

    secret: str = Field(..., description="Base32 encoded secret key")
    qr_code_url: str = Field(..., description="QR code URL for easy setup")
    manual_entry_key: str = Field(
        ..., description="Manual entry key for authenticator apps"
    )
    backup_codes: list[str] = Field(..., description="One-time backup codes")        json_schema_extra = {
            "example": {
                "secret": "JBSWY3DPEHPK3PXP",
                "qr_code_url": "otpauth://totp/Software:user@example.com?secret=JBSWY3DPEHPK3PXP&issuer=Software",
                "manual_entry_key": "JBSWY3DPEHPK3PXP",
                "backup_codes": ["123456", "789012", "345678"],
            }
        }


class TOTPVerificationRequest(BaseModel):
    """Request to verify TOTP code."""

    totp_code: str = Field(
        ..., min_length=6, max_length=6, description="6-digit TOTP code"
    )

    @field_validator("totp_code")
    @classmethod
    def validate_totp_code(cls, v):
        if not v.isdigit():
            raise ValueError("TOTP code must contain only digits")
        return v


class SMSSetupRequest(BaseModel):
    """Request to setup SMS-based MFA."""

    phone_number: str = Field(..., description="Phone number for SMS verification")

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v):
        # Basic phone number validation
        import re

        if not re.match(r"^\+?1?\d{9,15}$", v.replace(" ", "").replace("-", "")):
            raise ValueError("Invalid phone number format")
        return v


class SMSVerificationRequest(BaseModel):
    """Request to verify SMS code."""

    sms_code: str = Field(
        ..., min_length=6, max_length=6, description="6-digit SMS verification code"
    )

    @field_validator("sms_code")
    @classmethod
    def validate_sms_code(cls, v):
        if not v.isdigit():
            raise ValueError("SMS code must contain only digits")
        return v


class EmailVerificationRequest(BaseModel):
    """Request to verify email-based MFA code."""

    email_code: str = Field(
        ..., min_length=6, max_length=6, description="6-digit email verification code"
    )

    @field_validator("email_code")
    @classmethod
    def validate_email_code(cls, v):
        if not v.isdigit():
            raise ValueError("Email code must contain only digits")
        return v


class BackupCodeVerificationRequest(BaseModel):
    """Request to verify backup code."""

    backup_code: str = Field(
        ..., min_length=6, max_length=10, description="Backup recovery code"
    )


class MFAMethodDTO(BaseModel):
    """Data transfer object for MFA method information."""

    id: str = Field(..., description="Unique identifier for the MFA method")
    method_type: MFAMethodType = Field(..., description="Type of MFA method")
    status: MFAMethodStatus = Field(..., description="Current status of the method")
    display_name: str = Field(..., description="Human-readable name for the method")
    created_at: datetime = Field(..., description="When the method was created")
    last_used: datetime | None = Field(
        None, description="When the method was last used"
    )
    is_primary: bool = Field(
        False, description="Whether this is the primary MFA method"
    )

    # Method-specific details
    phone_number: str | None = Field(None, description="Phone number for SMS methods")
    email: str | None = Field(None, description="Email for email-based methods")
    backup_codes_remaining: int | None = Field(
        None, description="Remaining backup codes"
    )


class MFAStatusResponse(BaseModel):
    """Response containing user's MFA status."""

    mfa_enabled: bool = Field(..., description="Whether MFA is enabled for the user")
    active_methods: list[MFAMethodDTO] = Field(
        ..., description="List of active MFA methods"
    )
    pending_methods: list[MFAMethodDTO] = Field(
        ..., description="List of pending MFA methods"
    )
    primary_method: MFAMethodDTO | None = Field(None, description="Primary MFA method")
    backup_codes_available: bool = Field(
        ..., description="Whether backup codes are available"
    )        json_schema_extra = {
            "example": {
                "mfa_enabled": True,
                "active_methods": [
                    {
                        "id": "mfa_123",
                        "method_type": "totp",
                        "status": "active",
                        "display_name": "Google Authenticator",
                        "created_at": "2024-01-01T00:00:00Z",
                        "is_primary": True,
                    }
                ],
                "pending_methods": [],
                "primary_method": None,
                "backup_codes_available": True,
            }
        }


class MFAEnableRequest(BaseModel):
    """Request to enable MFA for the user."""

    method_type: MFAMethodType = Field(..., description="Type of MFA method to enable")
    verification_code: str = Field(
        ..., description="Verification code to confirm setup"
    )
    set_as_primary: bool = Field(True, description="Set this as the primary MFA method")


class MFADisableRequest(BaseModel):
    """Request to disable MFA method."""

    method_id: str = Field(..., description="ID of the MFA method to disable")
    verification_code: str = Field(
        ..., description="Verification code to confirm disable"
    )


class MFALoginRequest(BaseModel):
    """Request for MFA verification during login."""

    method_type: MFAMethodType = Field(..., description="Type of MFA method being used")
    verification_code: str = Field(..., description="MFA verification code")
    remember_device: bool = Field(
        False, description="Remember this device for future logins"
    )


class MFALoginResponse(BaseModel):
    """Response for successful MFA verification."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    device_remembered: bool = Field(False, description="Whether device was remembered")


class BackupCodesResponse(BaseModel):
    """Response containing backup codes."""

    backup_codes: list[str] = Field(..., description="List of backup codes")
    codes_remaining: int = Field(..., description="Number of unused backup codes")        json_schema_extra = {
            "example": {
                "backup_codes": ["123456", "789012", "345678"],
                "codes_remaining": 3,
            }
        }


class MFARecoveryRequest(BaseModel):
    """Request to recover account using backup codes."""

    backup_code: str = Field(..., description="Backup recovery code")
    new_password: str | None = Field(
        None, description="New password for account recovery"
    )


class MFARecoveryResponse(BaseModel):
    """Response for successful MFA recovery."""

    message: str = Field(..., description="Recovery success message")
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    remaining_codes: int = Field(..., description="Remaining backup codes")


class MFADeviceDTO(BaseModel):
    """Data transfer object for trusted devices."""

    id: str = Field(..., description="Unique device identifier")
    device_name: str = Field(..., description="Human-readable device name")
    device_type: str = Field(..., description="Type of device")
    user_agent: str = Field(..., description="User agent string")
    ip_address: str = Field(..., description="IP address when device was registered")
    created_at: datetime = Field(..., description="When device was registered")
    last_used: datetime = Field(..., description="When device was last used")
    is_active: bool = Field(True, description="Whether device is still trusted")


class TrustedDevicesResponse(BaseModel):
    """Response containing trusted devices."""

    devices: list[MFADeviceDTO] = Field(..., description="List of trusted devices")
    total_devices: int = Field(..., description="Total number of trusted devices")


class RevokeTrustedDeviceRequest(BaseModel):
    """Request to revoke a trusted device."""

    device_id: str = Field(..., description="ID of the device to revoke")


class MFASettingsDTO(BaseModel):
    """Data transfer object for MFA settings."""

    enforce_mfa: bool = Field(
        False, description="Whether MFA is enforced for all users"
    )
    allowed_methods: list[MFAMethodType] = Field(..., description="Allowed MFA methods")
    backup_codes_enabled: bool = Field(
        True, description="Whether backup codes are enabled"
    )
    remember_device_duration: int = Field(
        2592000, description="Device remember duration in seconds"
    )
    max_trusted_devices: int = Field(
        5, description="Maximum number of trusted devices per user"
    )        json_schema_extra = {
            "example": {
                "enforce_mfa": False,
                "allowed_methods": ["totp", "sms", "email"],
                "backup_codes_enabled": True,
                "remember_device_duration": 2592000,
                "max_trusted_devices": 5,
            }
        }


class MFAStatisticsDTO(BaseModel):
    """Data transfer object for MFA usage statistics."""

    total_users: int = Field(..., description="Total number of users")
    mfa_enabled_users: int = Field(..., description="Number of users with MFA enabled")
    mfa_adoption_rate: float = Field(..., description="MFA adoption rate percentage")
    method_usage: dict = Field(..., description="Usage statistics by method type")
    recent_authentications: int = Field(..., description="Recent MFA authentications")        json_schema_extra = {
            "example": {
                "total_users": 1000,
                "mfa_enabled_users": 650,
                "mfa_adoption_rate": 65.0,
                "method_usage": {"totp": 400, "sms": 200, "email": 50},
                "recent_authentications": 1200,
            }
        }


# Error DTOs
class MFAErrorResponse(BaseModel):
    """Error response for MFA operations."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict | None = Field(None, description="Additional error details")        json_schema_extra = {
            "example": {
                "error": "invalid_totp_code",
                "message": "The provided TOTP code is invalid or expired",
                "details": {"attempts_remaining": 2},
            }
        }
