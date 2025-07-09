"""Role-Based Access Control (RBAC) service for enterprise security."""

from __future__ import annotations

import hmac
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from pynomaly.domain.entities.security import (
    AccessRequest,
    ActionType,
    AuditEvent,
    PermissionType,
    SecurityPolicy,
    User,
    UserRole,
)


class RBACService:
    """Role-Based Access Control service for managing users, roles, and permissions."""

    def __init__(self, jwt_secret: str, security_policy: SecurityPolicy):
        self.jwt_secret = jwt_secret
        self.security_policy = security_policy
        self.logger = logging.getLogger(__name__)

        # In-memory storage (would be replaced with persistent storage in production)
        self.users: dict[UUID, User] = {}
        self.users_by_username: dict[str, UUID] = {}
        self.users_by_email: dict[str, UUID] = {}
        self.active_sessions: dict[str, UUID] = {}  # session_id -> user_id
        self.access_requests: dict[UUID, AccessRequest] = {}

        # Security monitoring
        self.failed_login_attempts: dict[str, list[datetime]] = {}  # IP -> timestamps
        self.suspicious_activities: list[dict[str, Any]] = []

        self.logger.info("RBAC service initialized")

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: set[UserRole] | None = None,
        created_by: UUID | None = None,
    ) -> User:
        """Create a new user with secure password hashing."""

        # Validate input
        if username in self.users_by_username:
            raise ValueError(f"Username '{username}' already exists")

        if email in self.users_by_email:
            raise ValueError(f"Email '{email}' already exists")

        # Validate password against policy
        self._validate_password(password)

        # Generate secure password hash
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(password, salt)

        # Create user
        user = User(
            user_id=uuid4(),
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            roles=roles or set(),
            created_by=created_by,
        )

        # Store user
        self.users[user.user_id] = user
        self.users_by_username[username] = user.user_id
        self.users_by_email[email] = user.user_id

        # Audit log
        await self._log_audit_event(
            user_id=created_by,
            action=ActionType.USER_CREATED,
            resource_type="user",
            resource_id=str(user.user_id),
            resource_name=username,
            success=True,
            additional_data={"roles": [role.value for role in user.roles]},
        )

        self.logger.info(f"Created user: {username}")
        return user

    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> str | None:
        """Authenticate user and return JWT token if successful."""

        user_id = self.users_by_username.get(username)
        if not user_id:
            await self._handle_failed_login(username, ip_address, "User not found")
            return None

        user = self.users[user_id]

        # Check if account is locked
        if user.is_locked:
            await self._handle_failed_login(username, ip_address, "Account locked")
            return None

        # Check if account is active
        if not user.is_active:
            await self._handle_failed_login(username, ip_address, "Account inactive")
            return None

        # Check IP whitelist if enabled
        if self.security_policy.ip_whitelist_enabled and ip_address:
            if not self._is_ip_allowed(ip_address):
                await self._handle_failed_login(username, ip_address, "IP not allowed")
                return None

        # Verify password
        if not self._verify_password(password, user.password_hash, user.salt):
            user.failed_login_attempts += 1

            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.security_policy.max_failed_login_attempts:
                user.is_locked = True
                await self._log_audit_event(
                    user_id=user.user_id,
                    action=ActionType.LOGIN_FAILED,
                    resource_type="user",
                    resource_id=str(user.user_id),
                    resource_name=username,
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    additional_data={"reason": "Account locked due to multiple failed attempts"},
                    security_level="CRITICAL",
                )
                self.logger.warning(f"Account locked for user {username} due to failed login attempts")

            await self._handle_failed_login(username, ip_address, "Invalid password")
            return None

        # Reset failed login attempts on successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()
        user.last_activity = datetime.utcnow()

        # Check if password has expired
        if user.password_expires_at and user.password_expires_at < datetime.utcnow():
            user.must_change_password = True

        # Generate JWT token
        session_id = secrets.token_urlsafe(32)
        token_payload = {
            "user_id": str(user.user_id),
            "username": username,
            "session_id": session_id,
            "roles": [role.value for role in user.roles],
            "iat": datetime.utcnow().timestamp(),
            "exp": (datetime.utcnow() + timedelta(seconds=self.security_policy.session_timeout)).timestamp(),
        }

        token = jwt.encode(token_payload, self.jwt_secret, algorithm="HS256")

        # Store active session
        self.active_sessions[session_id] = user.user_id

        # Audit log
        await self._log_audit_event(
            user_id=user.user_id,
            username=username,
            action=ActionType.LOGIN,
            resource_type="session",
            resource_id=session_id,
            success=True,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.logger.info(f"User authenticated successfully: {username}")
        return token

    async def validate_token(self, token: str) -> User | None:
        """Validate JWT token and return user if valid."""

        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            user_id = UUID(payload["user_id"])
            session_id = payload["session_id"]

            # Check if session exists
            if session_id not in self.active_sessions:
                return None

            user = self.users.get(user_id)
            if not user:
                return None

            # Check if user session is still valid
            if not user.is_session_valid():
                await self.logout_user(session_id)
                return None

            # Update last activity
            user.last_activity = datetime.utcnow()

            return user

        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
            return None
        except Exception as e:
            self.logger.error(f"Error validating token: {e}")
            return None

    async def logout_user(self, session_id: str) -> bool:
        """Logout user by session ID."""

        if session_id not in self.active_sessions:
            return False

        user_id = self.active_sessions[session_id]
        user = self.users.get(user_id)

        # Remove session
        del self.active_sessions[session_id]

        # Audit log
        if user:
            await self._log_audit_event(
                user_id=user.user_id,
                username=user.username,
                action=ActionType.LOGOUT,
                resource_type="session",
                resource_id=session_id,
                success=True,
            )

            self.logger.info(f"User logged out: {user.username}")

        return True

    async def check_permission(
        self,
        user_id: UUID,
        permission: PermissionType,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> bool:
        """Check if user has specific permission."""

        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False

        # Check if user has the permission
        has_permission = user.has_permission(permission)

        # Log permission check for sensitive actions
        if permission in [
            PermissionType.VIEW_SENSITIVE_DATA,
            PermissionType.MANAGE_SECRETS,
            PermissionType.ADMIN_SYSTEM,
            PermissionType.DELETE_DATA,
        ]:
            await self._log_audit_event(
                user_id=user.user_id,
                username=user.username,
                action=ActionType.SENSITIVE_DATA_ACCESSED if has_permission else ActionType.LOGIN_FAILED,
                resource_type=resource_type or "permission",
                resource_id=resource_id,
                success=has_permission,
                additional_data={
                    "permission": permission.value,
                    "granted": has_permission,
                },
                security_level="WARNING" if not has_permission else "INFO",
            )

        return has_permission

    async def assign_role(
        self,
        user_id: UUID,
        role: UserRole,
        assigned_by: UUID,
    ) -> bool:
        """Assign role to user."""

        user = self.users.get(user_id)
        if not user:
            return False

        if role in user.roles:
            return True  # Already has role

        user.roles.add(role)
        user.updated_at = datetime.utcnow()
        user.updated_by = assigned_by

        # Audit log
        await self._log_audit_event(
            user_id=assigned_by,
            action=ActionType.ROLE_ASSIGNED,
            resource_type="user",
            resource_id=str(user_id),
            resource_name=user.username,
            success=True,
            additional_data={
                "role": role.value,
                "target_user": user.username,
            },
        )

        self.logger.info(f"Assigned role {role.value} to user {user.username}")
        return True

    async def revoke_role(
        self,
        user_id: UUID,
        role: UserRole,
        revoked_by: UUID,
    ) -> bool:
        """Revoke role from user."""

        user = self.users.get(user_id)
        if not user:
            return False

        if role not in user.roles:
            return True  # Already doesn't have role

        user.roles.discard(role)
        user.updated_at = datetime.utcnow()
        user.updated_by = revoked_by

        # Audit log
        await self._log_audit_event(
            user_id=revoked_by,
            action=ActionType.ROLE_ASSIGNED,
            resource_type="user",
            resource_id=str(user_id),
            resource_name=user.username,
            success=True,
            additional_data={
                "role": role.value,
                "target_user": user.username,
                "action": "revoked",
            },
        )

        self.logger.info(f"Revoked role {role.value} from user {user.username}")
        return True

    async def grant_custom_permission(
        self,
        user_id: UUID,
        permission: PermissionType,
        granted_by: UUID,
        expiry: datetime | None = None,
    ) -> bool:
        """Grant custom permission to user."""

        user = self.users.get(user_id)
        if not user:
            return False

        user.custom_permissions.add(permission)
        user.updated_at = datetime.utcnow()
        user.updated_by = granted_by

        # Audit log
        await self._log_audit_event(
            user_id=granted_by,
            action=ActionType.PERMISSION_GRANTED,
            resource_type="user",
            resource_id=str(user_id),
            resource_name=user.username,
            success=True,
            additional_data={
                "permission": permission.value,
                "target_user": user.username,
                "expiry": expiry.isoformat() if expiry else None,
            },
            security_level="WARNING",
        )

        self.logger.info(f"Granted permission {permission.value} to user {user.username}")
        return True

    async def revoke_custom_permission(
        self,
        user_id: UUID,
        permission: PermissionType,
        revoked_by: UUID,
    ) -> bool:
        """Revoke custom permission from user."""

        user = self.users.get(user_id)
        if not user:
            return False

        user.custom_permissions.discard(permission)
        user.updated_at = datetime.utcnow()
        user.updated_by = revoked_by

        # Audit log
        await self._log_audit_event(
            user_id=revoked_by,
            action=ActionType.PERMISSION_REVOKED,
            resource_type="user",
            resource_id=str(user_id),
            resource_name=user.username,
            success=True,
            additional_data={
                "permission": permission.value,
                "target_user": user.username,
            },
            security_level="WARNING",
        )

        self.logger.info(f"Revoked permission {permission.value} from user {user.username}")
        return True

    async def create_access_request(
        self,
        requester_id: UUID,
        permission: PermissionType,
        resource_type: str,
        resource_id: str | None,
        justification: str,
        requested_duration: timedelta | None = None,
    ) -> AccessRequest:
        """Create access request for elevated permissions."""

        requester = self.users.get(requester_id)
        if not requester:
            raise ValueError("Requester not found")

        request = AccessRequest(
            request_id=uuid4(),
            requester_id=requester_id,
            requester_username=requester.username,
            requested_permission=permission,
            resource_type=resource_type,
            resource_id=resource_id,
            justification=justification,
            requested_start_time=datetime.utcnow(),
            requested_end_time=datetime.utcnow() + requested_duration if requested_duration else None,
        )

        self.access_requests[request.request_id] = request

        # Audit log
        await self._log_audit_event(
            user_id=requester_id,
            username=requester.username,
            action=ActionType.PERMISSION_GRANTED,  # Using this as closest match
            resource_type="access_request",
            resource_id=str(request.request_id),
            success=True,
            additional_data={
                "permission": permission.value,
                "justification": justification,
                "status": "requested",
            },
        )

        self.logger.info(f"Access request created by {requester.username} for {permission.value}")
        return request

    async def approve_access_request(
        self,
        request_id: UUID,
        approver_id: UUID,
        approval_comments: str | None = None,
        custom_duration: timedelta | None = None,
    ) -> bool:
        """Approve access request."""

        request = self.access_requests.get(request_id)
        if not request:
            return False

        approver = self.users.get(approver_id)
        if not approver:
            return False

        # Check if approver has permission to approve this request
        if not approver.has_permission(PermissionType.ADMIN_SECURITY):
            return False

        # Update request
        request.approval_status = "approved"
        request.approver_id = approver_id
        request.approver_username = approver.username
        request.approval_timestamp = datetime.utcnow()
        request.approval_comments = approval_comments
        request.granted_start_time = datetime.utcnow()

        if custom_duration:
            request.granted_end_time = datetime.utcnow() + custom_duration
        elif request.requested_end_time:
            request.granted_end_time = request.requested_end_time

        # Grant the permission temporarily
        await self.grant_custom_permission(
            request.requester_id,
            request.requested_permission,
            approver_id,
            request.granted_end_time,
        )

        # Audit log
        await self._log_audit_event(
            user_id=approver_id,
            username=approver.username,
            action=ActionType.PERMISSION_GRANTED,
            resource_type="access_request",
            resource_id=str(request_id),
            success=True,
            additional_data={
                "requester": request.requester_username,
                "permission": request.requested_permission.value,
                "approval_comments": approval_comments,
            },
            security_level="WARNING",
        )

        self.logger.info(f"Access request approved by {approver.username} for {request.requester_username}")
        return True

    def _validate_password(self, password: str) -> None:
        """Validate password against security policy."""

        if len(password) < self.security_policy.password_min_length:
            raise ValueError(f"Password must be at least {self.security_policy.password_min_length} characters")

        if self.security_policy.password_require_uppercase and not any(c.isupper() for c in password):
            raise ValueError("Password must contain at least one uppercase letter")

        if self.security_policy.password_require_lowercase and not any(c.islower() for c in password):
            raise ValueError("Password must contain at least one lowercase letter")

        if self.security_policy.password_require_numbers and not any(c.isdigit() for c in password):
            raise ValueError("Password must contain at least one number")

        if self.security_policy.password_require_special_chars:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                raise ValueError("Password must contain at least one special character")

    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password using PBKDF2."""

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )

        password_hash = kdf.derive(password.encode())
        return password_hash.hex()

    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash."""

        computed_hash = self._hash_password(password, salt)
        return hmac.compare_digest(computed_hash, password_hash)

    def _is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is in allowed ranges."""

        # Simple implementation - would use proper CIDR matching in production
        for allowed_range in self.security_policy.allowed_ip_ranges:
            if ip_address.startswith(allowed_range):
                return True

        return len(self.security_policy.allowed_ip_ranges) == 0

    async def _handle_failed_login(
        self,
        username: str,
        ip_address: str | None,
        reason: str,
    ) -> None:
        """Handle failed login attempt."""

        # Track failed attempts by IP
        if ip_address:
            if ip_address not in self.failed_login_attempts:
                self.failed_login_attempts[ip_address] = []

            self.failed_login_attempts[ip_address].append(datetime.utcnow())

            # Clean old attempts (older than 1 hour)
            cutoff = datetime.utcnow() - timedelta(hours=1)
            self.failed_login_attempts[ip_address] = [
                timestamp for timestamp in self.failed_login_attempts[ip_address]
                if timestamp > cutoff
            ]

        # Audit log
        await self._log_audit_event(
            user_id=None,
            username=username,
            action=ActionType.LOGIN_FAILED,
            resource_type="authentication",
            success=False,
            ip_address=ip_address,
            additional_data={"reason": reason},
            security_level="WARNING",
        )

        self.logger.warning(f"Failed login attempt for {username} from {ip_address}: {reason}")

    async def _log_audit_event(
        self,
        action: ActionType,
        resource_type: str,
        success: bool,
        user_id: UUID | None = None,
        username: str | None = None,
        resource_id: str | None = None,
        resource_name: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        session_id: str | None = None,
        additional_data: dict[str, Any] | None = None,
        security_level: str = "INFO",
    ) -> None:
        """Log audit event."""

        event = AuditEvent(
            event_id=uuid4(),
            user_id=user_id,
            username=username,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            success=success,
            additional_data=additional_data or {},
            security_level=security_level,
            compliance_relevant=True,
        )

        # In production, this would be sent to a centralized audit log system
        self.logger.info(f"Audit: {action.value} - {resource_type} - Success: {success}")

    def get_user_sessions(self, user_id: UUID) -> list[str]:
        """Get active sessions for user."""

        return [
            session_id for session_id, uid in self.active_sessions.items()
            if uid == user_id
        ]

    def get_security_metrics(self) -> dict[str, Any]:
        """Get security metrics for monitoring."""

        total_users = len(self.users)
        active_users = sum(1 for user in self.users.values() if user.is_active)
        locked_users = sum(1 for user in self.users.values() if user.is_locked)
        active_sessions = len(self.active_sessions)

        # Calculate failed login attempts in last hour
        recent_failed_attempts = 0
        cutoff = datetime.utcnow() - timedelta(hours=1)

        for attempts in self.failed_login_attempts.values():
            recent_failed_attempts += sum(1 for timestamp in attempts if timestamp > cutoff)

        return {
            "users": {
                "total": total_users,
                "active": active_users,
                "locked": locked_users,
                "locked_percentage": (locked_users / total_users * 100) if total_users > 0 else 0,
            },
            "sessions": {
                "active": active_sessions,
                "average_per_user": active_sessions / active_users if active_users > 0 else 0,
            },
            "security": {
                "failed_login_attempts_last_hour": recent_failed_attempts,
                "suspicious_activities": len(self.suspicious_activities),
            },
            "access_requests": {
                "pending": sum(1 for req in self.access_requests.values() if req.approval_status == "pending"),
                "approved": sum(1 for req in self.access_requests.values() if req.approval_status == "approved"),
                "rejected": sum(1 for req in self.access_requests.values() if req.approval_status == "rejected"),
            },
        }
