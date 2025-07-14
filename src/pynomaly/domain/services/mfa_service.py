#!/usr/bin/env python3
"""
Multi-Factor Authentication (MFA) service for enhanced security.
Provides TOTP, SMS, email, and backup code authentication methods.
"""

import base64
import hashlib
import hmac
import secrets
from datetime import UTC, datetime
from io import BytesIO

# Optional dependencies for MFA
try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False
    pyotp = None

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False
    qrcode = None

from pynomaly.domain.protocols.audit_logger_protocol import (
    AuditLevel,
    AuditLoggerProtocol,
    SecurityEventType,
)
from pynomaly.domain.value_objects.mfa_types import (
    MFADeviceDTO,
    MFAMethodDTO,
    MFAMethodStatus,
    MFAMethodType,
    MFAStatisticsDTO,
    TOTPSetupResponse,
)


class MFAService:
    """Service for handling multi-factor authentication operations."""

    def __init__(
        self, 
        audit_logger: AuditLoggerProtocol,
        redis_client=None, 
        email_service=None, 
        sms_service=None
    ):
        """Initialize MFA service with required audit logger and optional dependencies."""
        self.audit_logger = audit_logger
        self.redis_client = redis_client
        self.email_service = email_service
        self.sms_service = sms_service

        # Configuration
        self.totp_window = 1  # Allow 1 time step tolerance
        self.sms_code_expiry = 300  # 5 minutes
        self.email_code_expiry = 600  # 10 minutes
        self.backup_codes_count = 10
        self.device_remember_duration = 2592000  # 30 days

    def generate_totp_secret(self, user_id: str) -> str:
        """Generate a new TOTP secret for the user."""
        if not PYOTP_AVAILABLE:
            raise RuntimeError("TOTP functionality requires pyotp library")

        secret = pyotp.random_base32()

        # Store secret temporarily (will be confirmed when verified)
        if self.redis_client:
            self.redis_client.setex(
                f"mfa_totp_setup:{user_id}",
                3600,  # 1 hour expiry
                secret
            )

        self.audit_logger.log_security_event(
            SecurityEventType.MFA_TOTP_SETUP_INITIATED,
            f"TOTP setup initiated for user {user_id}",
            level=AuditLevel.INFO,
            user_id=user_id,
            details={"method": "totp"}
        )

        return secret

    def create_totp_setup_response(
        self,
        user_id: str,
        user_email: str,
        secret: str,
        app_name: str = "Pynomaly",
        issuer: str = "Pynomaly Security"
    ) -> TOTPSetupResponse:
        """Create TOTP setup response with QR code and backup codes."""
        if not PYOTP_AVAILABLE:
            raise RuntimeError("TOTP functionality requires pyotp library")

        # Create TOTP URI
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=issuer
        )

        # Generate QR code if available
        qr_code_url = totp_uri  # Fallback to URI if QR code generation fails

        if QRCODE_AVAILABLE:
            try:
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(totp_uri)
                qr.make(fit=True)

                # Convert QR code to base64 image
                img = qr.make_image(fill_color="black", back_color="white")
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                qr_code_base64 = base64.b64encode(buffered.getvalue()).decode()
                qr_code_url = f"data:image/png;base64,{qr_code_base64}"
            except Exception:
                # Fallback to URI if QR code generation fails
                qr_code_url = totp_uri

        # Generate backup codes
        backup_codes = self.generate_backup_codes(user_id)

        return TOTPSetupResponse(
            secret=secret,
            qr_code_url=qr_code_url,
            manual_entry_key=secret,
            backup_codes=backup_codes
        )

    def verify_totp_code(self, user_id: str, totp_code: str, secret: str = None) -> bool:
        """Verify TOTP code for the user."""
        if not PYOTP_AVAILABLE:
            raise RuntimeError("TOTP functionality requires pyotp library")

        try:
            # Get secret from setup or storage
            if not secret:
                if self.redis_client:
                    secret = self.redis_client.get(f"mfa_totp_confirmed:{user_id}")
                    if secret:
                        secret = secret.decode('utf-8')

                if not secret:
                    return False

            # Verify TOTP code
            totp = pyotp.TOTP(secret)
            is_valid = totp.verify(totp_code, valid_window=self.totp_window)

            if is_valid:
                # Prevent replay attacks
                if self.redis_client:
                    used_key = f"mfa_totp_used:{user_id}:{totp_code}"
                    if self.redis_client.get(used_key):
                        return False
                    self.redis_client.setex(used_key, 60, "used")  # 1 minute window

                self.audit_logger.log_security_event(
                    SecurityEventType.MFA_TOTP_VERIFIED,
                    f"TOTP verification successful for user {user_id}",
                    level=AuditLevel.INFO,
                    user_id=user_id,
                    details={"method": "totp", "success": True}
                )
            else:
                self.audit_logger.log_security_event(
                    SecurityEventType.MFA_TOTP_FAILED,
                    f"TOTP verification failed for user {user_id}",
                    level=AuditLevel.WARNING,
                    user_id=user_id,
                    details={"method": "totp", "success": False},
                    risk_score=30
                )

            return is_valid

        except Exception as e:
            self.audit_logger.log_security_event(
                SecurityEventType.MFA_TOTP_FAILED,
                f"TOTP verification error for user {user_id}: {str(e)}",
                level=AuditLevel.ERROR,
                user_id=user_id,
                details={"method": "totp", "error": str(e)},
                risk_score=20
            )
            return False

    def confirm_totp_setup(self, user_id: str, totp_code: str) -> bool:
        """Confirm TOTP setup by verifying the first code."""
        if not self.redis_client:
            return False

        # Get pending secret
        secret = self.redis_client.get(f"mfa_totp_setup:{user_id}")
        if not secret:
            return False

        secret = secret.decode('utf-8')

        # Verify the code
        if self.verify_totp_code(user_id, totp_code, secret):
            # Move secret to confirmed storage
            self.redis_client.setex(f"mfa_totp_confirmed:{user_id}", 86400 * 365, secret)
            self.redis_client.delete(f"mfa_totp_setup:{user_id}")

            self.audit_logger.log_security_event(
                SecurityEventType.MFA_TOTP_ENABLED,
                f"TOTP setup confirmed for user {user_id}",
                level=AuditLevel.INFO,
                user_id=user_id,
                details={"method": "totp", "enabled": True}
            )

            return True

        return False

    def send_sms_code(self, user_id: str, phone_number: str) -> bool:
        """Send SMS verification code."""
        if not self.sms_service:
            return False

        # Generate 6-digit code
        sms_code = f"{secrets.randbelow(1000000):06d}"

        # Store code temporarily
        if self.redis_client:
            self.redis_client.setex(
                f"mfa_sms_code:{user_id}",
                self.sms_code_expiry,
                sms_code
            )

        # Send SMS
        try:
            message = f"Your Pynomaly verification code is: {sms_code}. This code expires in 5 minutes."
            success = self.sms_service.send_sms(phone_number, message)

            if success:
                self.audit_logger.log_security_event(
                    SecurityEventType.MFA_SMS_SENT,
                    f"SMS code sent to user {user_id}",
                    level=AuditLevel.INFO,
                    user_id=user_id,
                    details={"method": "sms", "phone": phone_number[-4:]}  # Log last 4 digits
                )

            return success

        except Exception as e:
            self.audit_logger.log_security_event(
                SecurityEventType.MFA_SMS_FAILED,
                f"SMS sending failed for user {user_id}: {str(e)}",
                level=AuditLevel.ERROR,
                user_id=user_id,
                details={"method": "sms", "error": str(e)},
                risk_score=10
            )
            return False

    def verify_sms_code(self, user_id: str, sms_code: str) -> bool:
        """Verify SMS code."""
        if not self.redis_client:
            return False

        stored_code = self.redis_client.get(f"mfa_sms_code:{user_id}")
        if not stored_code:
            return False

        stored_code = stored_code.decode('utf-8')
        is_valid = hmac.compare_digest(stored_code, sms_code)

        if is_valid:
            # Remove used code
            self.redis_client.delete(f"mfa_sms_code:{user_id}")

            self.audit_logger.log_security_event(
                SecurityEventType.MFA_SMS_VERIFIED,
                f"SMS verification successful for user {user_id}",
                level=AuditLevel.INFO,
                user_id=user_id,
                details={"method": "sms", "success": True}
            )
        else:
            self.audit_logger.log_security_event(
                SecurityEventType.MFA_SMS_FAILED,
                f"SMS verification failed for user {user_id}",
                level=AuditLevel.WARNING,
                user_id=user_id,
                details={"method": "sms", "success": False},
                risk_score=30
            )

        return is_valid

    def send_email_code(self, user_id: str, email: str) -> bool:
        """Send email verification code."""
        if not self.email_service:
            return False

        # Generate 6-digit code
        email_code = f"{secrets.randbelow(1000000):06d}"

        # Store code temporarily
        if self.redis_client:
            self.redis_client.setex(
                f"mfa_email_code:{user_id}",
                self.email_code_expiry,
                email_code
            )

        # Send email
        try:
            subject = "Pynomaly Security Verification Code"
            message = f"""
            Your Pynomaly verification code is: {email_code}
            
            This code expires in 10 minutes.
            
            If you did not request this code, please contact support immediately.
            """

            success = self.email_service.send_email(email, subject, message)

            if success:
                self.audit_logger.log_security_event(
                    SecurityEventType.MFA_EMAIL_SENT,
                    f"Email code sent to user {user_id}",
                    level=AuditLevel.INFO,
                    user_id=user_id,
                    details={"method": "email", "email": email}
                )

            return success

        except Exception as e:
            self.audit_logger.log_security_event(
                SecurityEventType.MFA_EMAIL_FAILED,
                f"Email sending failed for user {user_id}: {str(e)}",
                level=AuditLevel.ERROR,
                user_id=user_id,
                details={"method": "email", "error": str(e)},
                risk_score=10
            )
            return False

    def verify_email_code(self, user_id: str, email_code: str) -> bool:
        """Verify email code."""
        if not self.redis_client:
            return False

        stored_code = self.redis_client.get(f"mfa_email_code:{user_id}")
        if not stored_code:
            return False

        stored_code = stored_code.decode('utf-8')
        is_valid = hmac.compare_digest(stored_code, email_code)

        if is_valid:
            # Remove used code
            self.redis_client.delete(f"mfa_email_code:{user_id}")

            self.audit_logger.log_security_event(
                SecurityEventType.MFA_EMAIL_VERIFIED,
                f"Email verification successful for user {user_id}",
                level=AuditLevel.INFO,
                user_id=user_id,
                details={"method": "email", "success": True}
            )
        else:
            self.audit_logger.log_security_event(
                SecurityEventType.MFA_EMAIL_FAILED,
                f"Email verification failed for user {user_id}",
                level=AuditLevel.WARNING,
                user_id=user_id,
                details={"method": "email", "success": False},
                risk_score=30
            )

        return is_valid

    def generate_backup_codes(self, user_id: str) -> list[str]:
        """Generate backup codes for the user."""
        backup_codes = []

        for _ in range(self.backup_codes_count):
            # Generate 8-digit backup code
            code = f"{secrets.randbelow(100000000):08d}"
            backup_codes.append(code)

        # Store hashed versions
        if self.redis_client:
            hashed_codes = []
            for code in backup_codes:
                hashed = hashlib.sha256(code.encode()).hexdigest()
                hashed_codes.append(hashed)

            self.redis_client.setex(
                f"mfa_backup_codes:{user_id}",
                86400 * 365,  # 1 year
                "|".join(hashed_codes)
            )

        self.audit_logger.log_security_event(
            SecurityEventType.MFA_BACKUP_CODES_GENERATED,
            f"Backup codes generated for user {user_id}",
            level=AuditLevel.INFO,
            user_id=user_id,
            details={"method": "backup_codes", "count": len(backup_codes)}
        )

        return backup_codes

    def verify_backup_code(self, user_id: str, backup_code: str) -> bool:
        """Verify backup code and remove it after use."""
        if not self.redis_client:
            return False

        stored_codes = self.redis_client.get(f"mfa_backup_codes:{user_id}")
        if not stored_codes:
            return False

        stored_codes = stored_codes.decode('utf-8').split('|')
        code_hash = hashlib.sha256(backup_code.encode()).hexdigest()

        if code_hash in stored_codes:
            # Remove used code
            stored_codes.remove(code_hash)

            if stored_codes:
                self.redis_client.setex(
                    f"mfa_backup_codes:{user_id}",
                    86400 * 365,
                    "|".join(stored_codes)
                )
            else:
                self.redis_client.delete(f"mfa_backup_codes:{user_id}")

            self.audit_logger.log_security_event(
                SecurityEventType.MFA_BACKUP_CODE_USED,
                f"Backup code used for user {user_id}",
                level=AuditLevel.INFO,
                user_id=user_id,
                details={"method": "backup_codes", "remaining": len(stored_codes)}
            )

            return True

        self.audit_logger.log_security_event(
            SecurityEventType.MFA_BACKUP_CODE_FAILED,
            f"Invalid backup code for user {user_id}",
            level=AuditLevel.WARNING,
            user_id=user_id,
            details={"method": "backup_codes", "success": False},
            risk_score=40
        )

        return False

    def get_backup_codes_count(self, user_id: str) -> int:
        """Get the number of remaining backup codes."""
        if not self.redis_client:
            return 0

        stored_codes = self.redis_client.get(f"mfa_backup_codes:{user_id}")
        if not stored_codes:
            return 0

        return len(stored_codes.decode('utf-8').split('|'))

    def remember_device(self, user_id: str, device_info: dict) -> str:
        """Remember a device for MFA bypass."""
        device_id = secrets.token_urlsafe(32)

        if self.redis_client:
            device_data = {
                "device_id": device_id,
                "user_id": user_id,
                "device_name": device_info.get("device_name", "Unknown Device"),
                "device_type": device_info.get("device_type", "Unknown"),
                "user_agent": device_info.get("user_agent", ""),
                "ip_address": device_info.get("ip_address", ""),
                "created_at": datetime.now(UTC).isoformat(),
                "last_used": datetime.now(UTC).isoformat()
            }

            self.redis_client.setex(
                f"mfa_trusted_device:{device_id}",
                self.device_remember_duration,
                "|".join(f"{k}:{v}" for k, v in device_data.items())
            )

        self.audit_logger.log_security_event(
            SecurityEventType.MFA_DEVICE_REMEMBERED,
            f"Device remembered for user {user_id}",
            level=AuditLevel.INFO,
            user_id=user_id,
            details={"device_id": device_id, "device_name": device_info.get("device_name")}
        )

        return device_id

    def is_device_trusted(self, device_id: str, user_id: str) -> bool:
        """Check if a device is trusted for MFA bypass."""
        if not self.redis_client:
            return False

        device_data = self.redis_client.get(f"mfa_trusted_device:{device_id}")
        if not device_data:
            return False

        # Parse device data
        device_data = device_data.decode('utf-8')
        device_info = {}
        for item in device_data.split('|'):
            if ':' in item:
                key, value = item.split(':', 1)
                device_info[key] = value

        # Verify device belongs to user
        if device_info.get("user_id") != user_id:
            return False

        # Update last used time
        device_info["last_used"] = datetime.now(UTC).isoformat()
        updated_data = "|".join(f"{k}:{v}" for k, v in device_info.items())
        self.redis_client.setex(
            f"mfa_trusted_device:{device_id}",
            self.device_remember_duration,
            updated_data
        )

        return True

    def revoke_trusted_device(self, device_id: str, user_id: str) -> bool:
        """Revoke a trusted device."""
        if not self.redis_client:
            return False

        # Verify device belongs to user
        if not self.is_device_trusted(device_id, user_id):
            return False

        self.redis_client.delete(f"mfa_trusted_device:{device_id}")

        self.audit_logger.log_security_event(
            SecurityEventType.MFA_DEVICE_REVOKED,
            f"Trusted device revoked for user {user_id}",
            level=AuditLevel.INFO,
            user_id=user_id,
            details={"device_id": device_id}
        )

        return True

    def get_trusted_devices(self, user_id: str) -> list[MFADeviceDTO]:
        """Get all trusted devices for a user."""
        if not self.redis_client:
            return []

        devices = []

        # This is a simplified implementation
        # In production, you'd want to store device info in a database
        # and use Redis only for quick lookups

        return devices

    def disable_mfa_method(self, user_id: str, method_type: MFAMethodType) -> bool:
        """Disable an MFA method for the user."""
        if not self.redis_client:
            return False

        if method_type == MFAMethodType.TOTP:
            self.redis_client.delete(f"mfa_totp_confirmed:{user_id}")
        elif method_type == MFAMethodType.BACKUP_CODES:
            self.redis_client.delete(f"mfa_backup_codes:{user_id}")

        self.audit_logger.log_security_event(
            SecurityEventType.MFA_METHOD_DISABLED,
            f"MFA method {method_type.value} disabled for user {user_id}",
            level=AuditLevel.INFO,
            user_id=user_id,
            details={"method": method_type.value, "disabled": True}
        )

        return True

    def get_mfa_methods(self, user_id: str) -> list[MFAMethodDTO]:
        """Get all MFA methods for a user."""
        methods = []

        if not self.redis_client:
            return methods

        # Check TOTP
        if self.redis_client.get(f"mfa_totp_confirmed:{user_id}"):
            methods.append(MFAMethodDTO(
                id=f"totp_{user_id}",
                method_type=MFAMethodType.TOTP,
                status=MFAMethodStatus.ACTIVE,
                display_name="Authenticator App",
                created_at=datetime.now(UTC),
                is_primary=True
            ))

        # Check backup codes
        backup_count = self.get_backup_codes_count(user_id)
        if backup_count > 0:
            methods.append(MFAMethodDTO(
                id=f"backup_{user_id}",
                method_type=MFAMethodType.BACKUP_CODES,
                status=MFAMethodStatus.ACTIVE,
                display_name="Backup Codes",
                created_at=datetime.now(UTC),
                backup_codes_remaining=backup_count
            ))

        return methods

    def get_mfa_statistics(self) -> MFAStatisticsDTO:
        """Get MFA usage statistics."""
        # This would typically query a database
        # This is a simplified implementation
        return MFAStatisticsDTO(
            total_users=1000,
            mfa_enabled_users=650,
            mfa_adoption_rate=65.0,
            method_usage={
                "totp": 400,
                "sms": 200,
                "email": 50,
                "backup_codes": 350
            },
            recent_authentications=1200
        )
