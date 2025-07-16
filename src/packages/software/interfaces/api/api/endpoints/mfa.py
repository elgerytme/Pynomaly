#!/usr/bin/env python3
"""
Multi-Factor Authentication (MFA) API endpoints.
Provides comprehensive MFA functionality including TOTP, SMS, email, and backup codes.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer

from monorepo.application.dto.mfa_dto import (
    BackupCodesResponse,
    EmailVerificationRequest,
    MFADisableRequest,
    MFAEnableRequest,
    MFALoginRequest,
    MFALoginResponse,
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
from monorepo.domain.services.mfa_service import MFAService
from monorepo.infrastructure.auth import get_current_user
from monorepo.infrastructure.cache import get_cache
from monorepo.infrastructure.security.audit_logger import (
    AuditLevel,
    SecurityEventType,
    get_audit_logger,
)
from monorepo.presentation.api.auth_deps import require_admin_simple
from monorepo.presentation.api.dependencies.auth import get_auth

router = APIRouter(prefix="/mfa", tags=["mfa"])
security = HTTPBearer()
audit_logger = get_audit_logger()


def get_mfa_service() -> MFAService:
    """Get MFA service instance."""
    cache = get_cache()
    return MFAService(redis_client=cache)


def get_client_info(request: Request) -> dict:
    """Extract client information from request."""
    return {
        "device_name": request.headers.get("X-Device-Name", "Unknown Device"),
        "device_type": request.headers.get("X-Device-Type", "Unknown"),
        "user_agent": request.headers.get("User-Agent", ""),
        "ip_address": request.client.host if request.client else "unknown",
    }


@router.post("/totp/setup", response_model=TOTPSetupResponse)
async def setup_totp(
    request: TOTPSetupRequest,
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Initialize TOTP (Time-based One-Time Password) setup.

    Returns QR code and manual entry key for authenticator apps.
    """
    try:
        # Generate TOTP secret
        secret = mfa_service.generate_totp_secret(current_user.id)

        # Create setup response with QR code and backup codes
        setup_response = mfa_service.create_totp_setup_response(
            user_id=current_user.id,
            user_email=current_user.email,
            secret=secret,
            app_name=request.app_name,
            issuer=request.issuer,
        )

        audit_logger.log_security_event(
            SecurityEventType.MFA_TOTP_SETUP_INITIATED,
            f"TOTP setup initiated for user {current_user.id}",
            level=AuditLevel.INFO,
            user_id=current_user.id,
            details={"method": "totp", "app_name": request.app_name},
        )

        return setup_response

    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_TOTP_SETUP_FAILED,
            f"TOTP setup failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"method": "totp", "error": str(e)},
            risk_score=20,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set up TOTP authentication",
        )


@router.post("/totp/verify")
async def verify_totp_setup(
    request: TOTPVerificationRequest,
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Verify TOTP setup by confirming the first code.

    This confirms the TOTP setup and enables the method.
    """
    try:
        # Confirm TOTP setup
        success = mfa_service.confirm_totp_setup(current_user.id, request.totp_code)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid TOTP code or setup not initiated",
            )

        audit_logger.log_security_event(
            SecurityEventType.MFA_TOTP_ENABLED,
            f"TOTP enabled for user {current_user.id}",
            level=AuditLevel.INFO,
            user_id=current_user.id,
            details={"method": "totp", "enabled": True},
        )

        return {"message": "TOTP authentication enabled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_TOTP_VERIFICATION_FAILED,
            f"TOTP verification failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"method": "totp", "error": str(e)},
            risk_score=30,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify TOTP code",
        )


@router.post("/sms/setup")
async def setup_sms(
    request: SMSSetupRequest,
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Set up SMS-based MFA.

    Sends verification code to the provided phone number.
    """
    try:
        # Send SMS verification code
        success = mfa_service.send_sms_code(current_user.id, request.phone_number)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send SMS verification code",
            )

        return {"message": "SMS verification code sent successfully"}

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_SMS_SETUP_FAILED,
            f"SMS setup failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"method": "sms", "error": str(e)},
            risk_score=20,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set up SMS authentication",
        )


@router.post("/sms/verify")
async def verify_sms_setup(
    request: SMSVerificationRequest,
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Verify SMS setup by confirming the SMS code.

    This confirms the SMS setup and enables the method.
    """
    try:
        # Verify SMS code
        success = mfa_service.verify_sms_code(current_user.id, request.sms_code)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid SMS code or expired",
            )

        return {"message": "SMS authentication enabled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_SMS_VERIFICATION_FAILED,
            f"SMS verification failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"method": "sms", "error": str(e)},
            risk_score=30,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify SMS code",
        )


@router.post("/email/setup")
async def setup_email(
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Set up email-based MFA.

    Sends verification code to the user's email address.
    """
    try:
        # Send email verification code
        success = mfa_service.send_email_code(current_user.id, current_user.email)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send email verification code",
            )

        return {"message": "Email verification code sent successfully"}

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_EMAIL_SETUP_FAILED,
            f"Email setup failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"method": "email", "error": str(e)},
            risk_score=20,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set up email authentication",
        )


@router.post("/email/verify")
async def verify_email_setup(
    request: EmailVerificationRequest,
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Verify email setup by confirming the email code.

    This confirms the email setup and enables the method.
    """
    try:
        # Verify email code
        success = mfa_service.verify_email_code(current_user.id, request.email_code)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email code or expired",
            )

        return {"message": "Email authentication enabled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_EMAIL_VERIFICATION_FAILED,
            f"Email verification failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"method": "email", "error": str(e)},
            risk_score=30,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify email code",
        )


@router.get("/status", response_model=MFAStatusResponse)
async def get_mfa_status(
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Get the current MFA status for the user.

    Returns active methods, pending methods, and availability of backup codes.
    """
    try:
        # Get user's MFA methods
        methods = mfa_service.get_mfa_methods(current_user.id)

        active_methods = [m for m in methods if m.status.value == "active"]
        pending_methods = [m for m in methods if m.status.value == "pending"]

        # Find primary method
        primary_method = None
        for method in active_methods:
            if method.is_primary:
                primary_method = method
                break

        # Check backup codes availability
        backup_codes_available = mfa_service.get_backup_codes_count(current_user.id) > 0

        return MFAStatusResponse(
            mfa_enabled=len(active_methods) > 0,
            active_methods=active_methods,
            pending_methods=pending_methods,
            primary_method=primary_method,
            backup_codes_available=backup_codes_available,
        )

    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_STATUS_CHECK_FAILED,
            f"MFA status check failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"error": str(e)},
            risk_score=10,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve MFA status",
        )


@router.post("/enable")
async def enable_mfa(
    request: MFAEnableRequest,
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Enable an MFA method after verification.

    Requires verification code to confirm the method works.
    """
    try:
        # Verify the method based on type
        success = False

        if request.method_type == MFAMethodType.TOTP:
            success = mfa_service.confirm_totp_setup(
                current_user.id, request.verification_code
            )
        elif request.method_type == MFAMethodType.SMS:
            success = mfa_service.verify_sms_code(
                current_user.id, request.verification_code
            )
        elif request.method_type == MFAMethodType.EMAIL:
            success = mfa_service.verify_email_code(
                current_user.id, request.verification_code
            )
        elif request.method_type == MFAMethodType.BACKUP_CODES:
            success = mfa_service.verify_backup_code(
                current_user.id, request.verification_code
            )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification code",
            )

        audit_logger.log_security_event(
            SecurityEventType.MFA_METHOD_ENABLED,
            f"MFA method {request.method_type.value} enabled for user {current_user.id}",
            level=AuditLevel.INFO,
            user_id=current_user.id,
            details={
                "method": request.method_type.value,
                "enabled": True,
                "primary": request.set_as_primary,
            },
        )

        return {"message": f"{request.method_type.value} MFA enabled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_ENABLE_FAILED,
            f"MFA enable failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"method": request.method_type.value, "error": str(e)},
            risk_score=20,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enable MFA method",
        )


@router.post("/disable")
async def disable_mfa(
    request: MFADisableRequest,
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Disable an MFA method.

    Requires verification code to confirm the action.
    """
    try:
        # Parse method type from method ID
        method_type = None
        if request.method_id.startswith("totp_"):
            method_type = MFAMethodType.TOTP
        elif request.method_id.startswith("backup_"):
            method_type = MFAMethodType.BACKUP_CODES

        if not method_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid method ID"
            )

        # Verify the code first
        success = False
        if method_type == MFAMethodType.TOTP:
            success = mfa_service.verify_totp_code(
                current_user.id, request.verification_code
            )
        elif method_type == MFAMethodType.BACKUP_CODES:
            success = mfa_service.verify_backup_code(
                current_user.id, request.verification_code
            )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification code",
            )

        # Disable the method
        mfa_service.disable_mfa_method(current_user.id, method_type)

        audit_logger.log_security_event(
            SecurityEventType.MFA_METHOD_DISABLED,
            f"MFA method {method_type.value} disabled for user {current_user.id}",
            level=AuditLevel.INFO,
            user_id=current_user.id,
            details={"method": method_type.value, "disabled": True},
        )

        return {"message": f"{method_type.value} MFA disabled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_DISABLE_FAILED,
            f"MFA disable failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"method_id": request.method_id, "error": str(e)},
            risk_score=20,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disable MFA method",
        )


@router.post("/verify", response_model=MFALoginResponse)
async def verify_mfa_login(
    request: MFALoginRequest,
    req: Request,
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
    auth_service=Depends(get_auth),
):
    """
    Verify MFA during login process.

    Used after initial username/password authentication.
    """
    try:
        # Verify the MFA code
        success = False

        if request.method_type == MFAMethodType.TOTP:
            success = mfa_service.verify_totp_code(
                current_user.id, request.verification_code
            )
        elif request.method_type == MFAMethodType.SMS:
            success = mfa_service.verify_sms_code(
                current_user.id, request.verification_code
            )
        elif request.method_type == MFAMethodType.EMAIL:
            success = mfa_service.verify_email_code(
                current_user.id, request.verification_code
            )
        elif request.method_type == MFAMethodType.BACKUP_CODES:
            success = mfa_service.verify_backup_code(
                current_user.id, request.verification_code
            )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid MFA code"
            )

        # Generate access tokens
        access_token = auth_service.create_access_token(data={"sub": current_user.id})
        refresh_token = auth_service.create_refresh_token(data={"sub": current_user.id})

        # Remember device if requested
        device_remembered = False
        if request.remember_device:
            client_info = get_client_info(req)
            mfa_service.remember_device(current_user.id, client_info)
            device_remembered = True

        audit_logger.log_security_event(
            SecurityEventType.MFA_LOGIN_SUCCESS,
            f"MFA login successful for user {current_user.id}",
            level=AuditLevel.INFO,
            user_id=current_user.id,
            details={
                "method": request.method_type.value,
                "device_remembered": device_remembered,
                "success": True,
            },
        )

        return MFALoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=3600,  # 1 hour
            device_remembered=device_remembered,
        )

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_LOGIN_FAILED,
            f"MFA login failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"method": request.method_type.value, "error": str(e)},
            risk_score=40,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA verification failed",
        )


@router.get("/backup-codes", response_model=BackupCodesResponse)
async def get_backup_codes(
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Get backup codes for the user.

    Returns remaining backup codes count.
    """
    try:
        codes_remaining = mfa_service.get_backup_codes_count(current_user.id)

        return BackupCodesResponse(
            backup_codes=[],  # Don't expose actual codes via API
            codes_remaining=codes_remaining,
        )

    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_BACKUP_CODES_ACCESS_FAILED,
            f"Backup codes access failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"error": str(e)},
            risk_score=10,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve backup codes",
        )


@router.post("/backup-codes/regenerate", response_model=BackupCodesResponse)
async def regenerate_backup_codes(
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Regenerate backup codes for the user.

    This invalidates all existing backup codes.
    """
    try:
        # Generate new backup codes
        backup_codes = mfa_service.generate_backup_codes(current_user.id)

        audit_logger.log_security_event(
            SecurityEventType.MFA_BACKUP_CODES_REGENERATED,
            f"Backup codes regenerated for user {current_user.id}",
            level=AuditLevel.INFO,
            user_id=current_user.id,
            details={"method": "backup_codes", "count": len(backup_codes)},
        )

        return BackupCodesResponse(
            backup_codes=backup_codes, codes_remaining=len(backup_codes)
        )

    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_BACKUP_CODES_REGENERATION_FAILED,
            f"Backup codes regeneration failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"error": str(e)},
            risk_score=20,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate backup codes",
        )


@router.post("/recovery", response_model=MFARecoveryResponse)
async def recover_with_backup_code(
    request: MFARecoveryRequest,
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
    auth_service=Depends(get_auth),
):
    """
    Recover account access using backup codes.

    Used when primary MFA method is unavailable.
    """
    try:
        # Verify backup code
        success = mfa_service.verify_backup_code(current_user.id, request.backup_code)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid backup code"
            )

        # Generate new access tokens
        access_token = auth_service.create_access_token(data={"sub": current_user.id})
        refresh_token = auth_service.create_refresh_token(data={"sub": current_user.id})

        # Get remaining codes count
        remaining_codes = mfa_service.get_backup_codes_count(current_user.id)

        audit_logger.log_security_event(
            SecurityEventType.MFA_RECOVERY_SUCCESS,
            f"Account recovery successful for user {current_user.id}",
            level=AuditLevel.INFO,
            user_id=current_user.id,
            details={
                "method": "backup_codes",
                "remaining_codes": remaining_codes,
                "success": True,
            },
        )

        return MFARecoveryResponse(
            message="Account recovery successful",
            access_token=access_token,
            refresh_token=refresh_token,
            remaining_codes=remaining_codes,
        )

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_RECOVERY_FAILED,
            f"Account recovery failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"error": str(e)},
            risk_score=50,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account recovery failed",
        )


@router.get("/trusted-devices", response_model=TrustedDevicesResponse)
async def get_trusted_devices(
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Get list of trusted devices for the user.

    Returns devices that are remembered for MFA bypass.
    """
    try:
        devices = mfa_service.get_trusted_devices(current_user.id)

        return TrustedDevicesResponse(devices=devices, total_devices=len(devices))

    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_TRUSTED_DEVICES_ACCESS_FAILED,
            f"Trusted devices access failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"error": str(e)},
            risk_score=10,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trusted devices",
        )


@router.post("/trusted-devices/revoke")
async def revoke_trusted_device(
    request: RevokeTrustedDeviceRequest,
    current_user=Depends(get_current_user),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Revoke a trusted device.

    Removes device from trusted list, requiring MFA on next login.
    """
    try:
        success = mfa_service.revoke_trusted_device(request.device_id, current_user.id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Trusted device not found"
            )

        audit_logger.log_security_event(
            SecurityEventType.MFA_DEVICE_REVOKED,
            f"Trusted device revoked for user {current_user.id}",
            level=AuditLevel.INFO,
            user_id=current_user.id,
            details={"device_id": request.device_id},
        )

        return {"message": "Trusted device revoked successfully"}

    except HTTPException:
        raise
    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_DEVICE_REVOCATION_FAILED,
            f"Device revocation failed for user {current_user.id}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_user.id,
            details={"device_id": request.device_id, "error": str(e)},
            risk_score=20,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke trusted device",
        )


# Admin endpoints
@router.get("/admin/settings", response_model=MFASettingsDTO)
async def get_mfa_settings(
    current_admin=Depends(require_admin_simple),
):
    """
    Get MFA settings for the organization.

    Admin only endpoint.
    """
    try:
        # This would typically come from database settings
        return MFASettingsDTO(
            enforce_mfa=False,
            allowed_methods=[
                MFAMethodType.TOTP,
                MFAMethodType.SMS,
                MFAMethodType.EMAIL,
            ],
            backup_codes_enabled=True,
            remember_device_duration=2592000,  # 30 days
            max_trusted_devices=5,
        )

    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_SETTINGS_ACCESS_FAILED,
            f"MFA settings access failed for admin {current_admin.id if current_admin else 'unknown'}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_admin.id if current_admin else None,
            details={"error": str(e)},
            risk_score=10,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve MFA settings",
        )


@router.get("/admin/statistics", response_model=MFAStatisticsDTO)
async def get_mfa_statistics(
    current_admin=Depends(require_admin_simple),
    mfa_service: MFAService = Depends(get_mfa_service),
):
    """
    Get MFA usage statistics.

    Admin only endpoint.
    """
    try:
        statistics = mfa_service.get_mfa_statistics()

        audit_logger.log_security_event(
            SecurityEventType.MFA_STATISTICS_ACCESS,
            f"MFA statistics accessed by admin {current_admin.id if current_admin else 'unknown'}",
            level=AuditLevel.INFO,
            user_id=current_admin.id if current_admin else None,
            details={"statistics": True},
        )

        return statistics

    except Exception as e:
        audit_logger.log_security_event(
            SecurityEventType.MFA_STATISTICS_ACCESS_FAILED,
            f"MFA statistics access failed for admin {current_admin.id if current_admin else 'unknown'}: {str(e)}",
            level=AuditLevel.ERROR,
            user_id=current_admin.id if current_admin else None,
            details={"error": str(e)},
            risk_score=10,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve MFA statistics",
        )
