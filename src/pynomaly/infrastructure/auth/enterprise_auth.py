"""Enterprise authentication and authorization system with LDAP, SAML, OAuth, MFA, and RBAC."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import jwt
from pydantic import BaseModel, Field, SecretStr

logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    """Authentication methods."""
    
    PASSWORD = "password"
    LDAP = "ldap"
    SAML = "saml"
    OAUTH2 = "oauth2"
    OIDC = "oidc"
    API_KEY = "api_key"
    JWT = "jwt"
    CERTIFICATE = "certificate"


class MFAMethod(str, Enum):
    """Multi-factor authentication methods."""
    
    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    HARDWARE_TOKEN = "hardware_token"
    BIOMETRIC = "biometric"


class Permission(str, Enum):
    """System permissions."""
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"
    
    # Model permissions
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    MODEL_DELETE = "model:delete"
    
    # Analytics permissions
    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_CREATE = "analytics:create"
    ANALYTICS_MANAGE = "analytics:manage"
    
    # System permissions
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_ADMIN = "system:admin"
    
    # User management
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    USER_ADMIN = "user:admin"


@dataclass
class User:
    """User entity."""
    
    user_id: str
    username: str
    email: str
    full_name: str
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    roles: Set[str] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    groups: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    auth_methods: Set[AuthMethod] = field(default_factory=set)
    mfa_enabled: bool = False
    mfa_methods: Set[MFAMethod] = field(default_factory=set)


@dataclass
class Role:
    """Role entity."""
    
    role_id: str
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AuthSession:
    """Authentication session."""
    
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    auth_method: AuthMethod
    mfa_verified: bool = False
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


class AuthenticationResult(BaseModel):
    """Authentication result."""
    
    success: bool
    user: Optional[User] = None
    session: Optional[AuthSession] = None
    error_message: Optional[str] = None
    requires_mfa: bool = False
    mfa_methods: List[MFAMethod] = Field(default_factory=list)
    next_step: Optional[str] = None


class AuthProvider(ABC):
    """Abstract authentication provider."""
    
    @abstractmethod
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        **kwargs
    ) -> AuthenticationResult:
        """Authenticate user with credentials."""
        pass
    
    @abstractmethod
    async def get_user_info(self, user_id: str) -> Optional[User]:
        """Get user information."""
        pass


class LDAPAuthProvider(AuthProvider):
    """LDAP authentication provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_uri = config.get("server_uri", "ldap://localhost:389")
        self.bind_dn = config.get("bind_dn")
        self.bind_password = config.get("bind_password")
        self.user_search_base = config.get("user_search_base", "ou=users,dc=example,dc=com")
        self.user_search_filter = config.get("user_search_filter", "(uid={username})")
        self.group_search_base = config.get("group_search_base", "ou=groups,dc=example,dc=com")
        self.attributes = config.get("attributes", {
            "email": "mail",
            "full_name": "cn",
            "groups": "memberOf"
        })
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        **kwargs
    ) -> AuthenticationResult:
        """Authenticate user against LDAP."""
        try:
            import ldap3
            from ldap3 import Server, Connection, ALL
            
            username = credentials.get("username")
            password = credentials.get("password")
            
            if not username or not password:
                return AuthenticationResult(
                    success=False,
                    error_message="Username and password required"
                )
            
            # Connect to LDAP server
            server = Server(self.server_uri, get_info=ALL)
            
            # Search for user
            search_conn = Connection(server, self.bind_dn, self.bind_password)
            if not search_conn.bind():
                return AuthenticationResult(
                    success=False,
                    error_message="LDAP server connection failed"
                )
            
            search_filter = self.user_search_filter.format(username=username)
            search_conn.search(
                self.user_search_base,
                search_filter,
                attributes=list(self.attributes.values())
            )
            
            if not search_conn.entries:
                return AuthenticationResult(
                    success=False,
                    error_message="User not found in LDAP"
                )
            
            user_dn = search_conn.entries[0].entry_dn
            user_attrs = search_conn.entries[0].entry_attributes_as_dict
            
            # Authenticate user
            auth_conn = Connection(server, user_dn, password)
            if not auth_conn.bind():
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid credentials"
                )
            
            # Create user object
            user = User(
                user_id=username,
                username=username,
                email=user_attrs.get(self.attributes.get("email", "mail"), [""])[0],
                full_name=user_attrs.get(self.attributes.get("full_name", "cn"), [""])[0],
                auth_methods={AuthMethod.LDAP}
            )
            
            # Extract groups
            groups_attr = self.attributes.get("groups", "memberOf")
            if groups_attr in user_attrs:
                user.groups = set(user_attrs[groups_attr])
            
            return AuthenticationResult(
                success=True,
                user=user
            )
            
        except ImportError:
            logger.error("ldap3 library not available")
            return AuthenticationResult(
                success=False,
                error_message="LDAP authentication not available"
            )
        except Exception as e:
            logger.error(f"LDAP authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                error_message="LDAP authentication failed"
            )
    
    async def get_user_info(self, user_id: str) -> Optional[User]:
        """Get user information from LDAP."""
        try:
            import ldap3
            from ldap3 import Server, Connection, ALL
            
            server = Server(self.server_uri, get_info=ALL)
            conn = Connection(server, self.bind_dn, self.bind_password)
            
            if not conn.bind():
                return None
            
            search_filter = self.user_search_filter.format(username=user_id)
            conn.search(
                self.user_search_base,
                search_filter,
                attributes=list(self.attributes.values())
            )
            
            if not conn.entries:
                return None
            
            user_attrs = conn.entries[0].entry_attributes_as_dict
            
            user = User(
                user_id=user_id,
                username=user_id,
                email=user_attrs.get(self.attributes.get("email", "mail"), [""])[0],
                full_name=user_attrs.get(self.attributes.get("full_name", "cn"), [""])[0],
                auth_methods={AuthMethod.LDAP}
            )
            
            # Extract groups
            groups_attr = self.attributes.get("groups", "memberOf")
            if groups_attr in user_attrs:
                user.groups = set(user_attrs[groups_attr])
            
            return user
            
        except Exception as e:
            logger.error(f"Failed to get LDAP user info: {e}")
            return None


class SAMLAuthProvider(AuthProvider):
    """SAML authentication provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.idp_metadata_url = config.get("idp_metadata_url")
        self.sp_entity_id = config.get("sp_entity_id")
        self.sp_acs_url = config.get("sp_acs_url")
        self.certificate_file = config.get("certificate_file")
        self.private_key_file = config.get("private_key_file")
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        **kwargs
    ) -> AuthenticationResult:
        """Authenticate user with SAML response."""
        try:
            from onelogin.saml2.auth import OneLogin_Saml2_Auth
            from onelogin.saml2.settings import OneLogin_Saml2_Settings
            
            saml_response = credentials.get("saml_response")
            if not saml_response:
                return AuthenticationResult(
                    success=False,
                    error_message="SAML response required"
                )
            
            # Configure SAML settings
            settings = {
                "sp": {
                    "entityId": self.sp_entity_id,
                    "assertionConsumerService": {
                        "url": self.sp_acs_url,
                        "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                    }
                },
                "idp": {
                    "entityId": self.config.get("idp_entity_id"),
                    "singleSignOnService": {
                        "url": self.config.get("idp_sso_url"),
                        "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
                    }
                }
            }
            
            # Process SAML response
            req = kwargs.get("request", {})
            auth = OneLogin_Saml2_Auth(req, settings)
            auth.process_response()
            
            if not auth.is_authenticated():
                return AuthenticationResult(
                    success=False,
                    error_message="SAML authentication failed"
                )
            
            # Extract user attributes
            attributes = auth.get_attributes()
            nameid = auth.get_nameid()
            
            user = User(
                user_id=nameid,
                username=nameid,
                email=attributes.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress", [""])[0],
                full_name=attributes.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name", [""])[0],
                auth_methods={AuthMethod.SAML}
            )
            
            return AuthenticationResult(
                success=True,
                user=user
            )
            
        except ImportError:
            logger.error("python3-saml library not available")
            return AuthenticationResult(
                success=False,
                error_message="SAML authentication not available"
            )
        except Exception as e:
            logger.error(f"SAML authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                error_message="SAML authentication failed"
            )
    
    async def get_user_info(self, user_id: str) -> Optional[User]:
        """Get user information (SAML doesn't support direct lookup)."""
        return None


class OAuthProvider(AuthProvider):
    """OAuth 2.0 / OpenID Connect authentication provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.authorization_endpoint = config.get("authorization_endpoint")
        self.token_endpoint = config.get("token_endpoint")
        self.userinfo_endpoint = config.get("userinfo_endpoint")
        self.jwks_uri = config.get("jwks_uri")
        self.scopes = config.get("scopes", ["openid", "profile", "email"])
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        **kwargs
    ) -> AuthenticationResult:
        """Authenticate user with OAuth code or token."""
        try:
            import aiohttp
            
            # Handle authorization code flow
            if "code" in credentials:
                return await self._handle_authorization_code(credentials, **kwargs)
            
            # Handle access token validation
            if "access_token" in credentials:
                return await self._handle_access_token(credentials, **kwargs)
            
            return AuthenticationResult(
                success=False,
                error_message="OAuth code or access_token required"
            )
            
        except Exception as e:
            logger.error(f"OAuth authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                error_message="OAuth authentication failed"
            )
    
    async def _handle_authorization_code(
        self,
        credentials: Dict[str, Any],
        **kwargs
    ) -> AuthenticationResult:
        """Handle OAuth authorization code flow."""
        import aiohttp
        
        code = credentials.get("code")
        redirect_uri = credentials.get("redirect_uri")
        
        # Exchange code for token
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.token_endpoint, data=token_data) as response:
                if response.status != 200:
                    return AuthenticationResult(
                        success=False,
                        error_message="Failed to exchange OAuth code"
                    )
                
                token_response = await response.json()
                access_token = token_response.get("access_token")
                
                if not access_token:
                    return AuthenticationResult(
                        success=False,
                        error_message="No access token received"
                    )
                
                # Get user info
                headers = {"Authorization": f"Bearer {access_token}"}
                async with session.get(self.userinfo_endpoint, headers=headers) as user_response:
                    if user_response.status != 200:
                        return AuthenticationResult(
                            success=False,
                            error_message="Failed to get user info"
                        )
                    
                    user_info = await user_response.json()
                    
                    user = User(
                        user_id=user_info.get("sub"),
                        username=user_info.get("preferred_username", user_info.get("sub")),
                        email=user_info.get("email", ""),
                        full_name=user_info.get("name", ""),
                        is_verified=user_info.get("email_verified", False),
                        auth_methods={AuthMethod.OAUTH2}
                    )
                    
                    return AuthenticationResult(
                        success=True,
                        user=user
                    )
    
    async def _handle_access_token(
        self,
        credentials: Dict[str, Any],
        **kwargs
    ) -> AuthenticationResult:
        """Handle OAuth access token validation."""
        import aiohttp
        
        access_token = credentials.get("access_token")
        
        # Validate token and get user info
        headers = {"Authorization": f"Bearer {access_token}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.userinfo_endpoint, headers=headers) as response:
                if response.status != 200:
                    return AuthenticationResult(
                        success=False,
                        error_message="Invalid access token"
                    )
                
                user_info = await response.json()
                
                user = User(
                    user_id=user_info.get("sub"),
                    username=user_info.get("preferred_username", user_info.get("sub")),
                    email=user_info.get("email", ""),
                    full_name=user_info.get("name", ""),
                    is_verified=user_info.get("email_verified", False),
                    auth_methods={AuthMethod.OAUTH2}
                )
                
                return AuthenticationResult(
                    success=True,
                    user=user
                )
    
    async def get_user_info(self, user_id: str) -> Optional[User]:
        """Get user information (requires active session)."""
        return None


class MFAProvider:
    """Multi-factor authentication provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.totp_issuer = config.get("totp_issuer", "Pynomaly")
        self.sms_provider = config.get("sms_provider")
        self.email_provider = config.get("email_provider")
    
    async def generate_totp_secret(self, user: User) -> str:
        """Generate TOTP secret for user."""
        try:
            import pyotp
            
            secret = pyotp.random_base32()
            return secret
            
        except ImportError:
            logger.error("pyotp library not available")
            return secrets.token_hex(16)
    
    async def generate_totp_qr_code(self, user: User, secret: str) -> str:
        """Generate TOTP QR code URL."""
        try:
            import pyotp
            
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=user.email,
                issuer_name=self.totp_issuer
            )
            
            return provisioning_uri
            
        except ImportError:
            logger.error("pyotp library not available")
            return ""
    
    async def verify_totp(self, secret: str, token: str) -> bool:
        """Verify TOTP token."""
        try:
            import pyotp
            
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)
            
        except ImportError:
            logger.error("pyotp library not available")
            return False
    
    async def send_sms_code(self, user: User, code: str) -> bool:
        """Send SMS verification code."""
        # Implementation would integrate with SMS provider (Twilio, AWS SNS, etc.)
        logger.info(f"SMS code sent to user {user.user_id}: {code}")
        return True
    
    async def send_email_code(self, user: User, code: str) -> bool:
        """Send email verification code."""
        # Implementation would integrate with email provider
        logger.info(f"Email code sent to user {user.user_id}: {code}")
        return True
    
    async def generate_verification_code(self) -> str:
        """Generate 6-digit verification code."""
        return f"{secrets.randbelow(1000000):06d}"


class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default system roles."""
        # Admin role
        admin_role = Role(
            role_id="admin",
            name="Administrator",
            description="Full system access",
            permissions={p for p in Permission},
            is_system_role=True
        )
        self.roles["admin"] = admin_role
        
        # Analyst role
        analyst_role = Role(
            role_id="analyst",
            name="Data Analyst",
            description="Data analysis and model access",
            permissions={
                Permission.DATA_READ,
                Permission.DATA_EXPORT,
                Permission.MODEL_READ,
                Permission.ANALYTICS_VIEW,
                Permission.ANALYTICS_CREATE
            },
            is_system_role=True
        )
        self.roles["analyst"] = analyst_role
        
        # Data Scientist role
        data_scientist_role = Role(
            role_id="data_scientist",
            name="Data Scientist",
            description="Model development and training",
            permissions={
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.DATA_EXPORT,
                Permission.MODEL_READ,
                Permission.MODEL_WRITE,
                Permission.MODEL_TRAIN,
                Permission.ANALYTICS_VIEW,
                Permission.ANALYTICS_CREATE,
                Permission.ANALYTICS_MANAGE
            },
            is_system_role=True
        )
        self.roles["data_scientist"] = data_scientist_role
        
        # Viewer role
        viewer_role = Role(
            role_id="viewer",
            name="Viewer",
            description="Read-only access",
            permissions={
                Permission.DATA_READ,
                Permission.MODEL_READ,
                Permission.ANALYTICS_VIEW
            },
            is_system_role=True
        )
        self.roles["viewer"] = viewer_role
    
    async def create_role(self, role: Role) -> bool:
        """Create new role."""
        try:
            if role.role_id in self.roles:
                return False
            
            self.roles[role.role_id] = role
            logger.info(f"Created role: {role.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create role: {e}")
            return False
    
    async def assign_role_to_user(self, user_id: str, role_id: str) -> bool:
        """Assign role to user."""
        try:
            if role_id not in self.roles:
                return False
            
            if user_id not in self.user_roles:
                self.user_roles[user_id] = set()
            
            self.user_roles[user_id].add(role_id)
            logger.info(f"Assigned role {role_id} to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role: {e}")
            return False
    
    async def remove_role_from_user(self, user_id: str, role_id: str) -> bool:
        """Remove role from user."""
        try:
            if user_id in self.user_roles:
                self.user_roles[user_id].discard(role_id)
                logger.info(f"Removed role {role_id} from user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove role: {e}")
            return False
    
    async def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user."""
        permissions = set()
        
        user_role_ids = self.user_roles.get(user_id, set())
        for role_id in user_role_ids:
            if role_id in self.roles:
                permissions.update(self.roles[role_id].permissions)
        
        return permissions
    
    async def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user_permissions = await self.get_user_permissions(user_id)
        return permission in user_permissions
    
    async def get_user_roles(self, user_id: str) -> List[Role]:
        """Get all roles for user."""
        user_role_ids = self.user_roles.get(user_id, set())
        return [self.roles[role_id] for role_id in user_role_ids if role_id in self.roles]
    
    async def list_roles(self) -> List[Role]:
        """List all available roles."""
        return list(self.roles.values())


class SessionManager:
    """Authentication session manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sessions: Dict[str, AuthSession] = {}
        self.session_timeout = timedelta(hours=config.get("session_timeout_hours", 8))
        self.jwt_secret = config.get("jwt_secret", secrets.token_hex(32))
        self.jwt_algorithm = config.get("jwt_algorithm", "HS256")
    
    async def create_session(
        self,
        user: User,
        auth_method: AuthMethod,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuthSession:
        """Create new authentication session."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + self.session_timeout
        
        session = AuthSession(
            session_id=session_id,
            user_id=user.user_id,
            created_at=datetime.now(),
            expires_at=expires_at,
            auth_method=auth_method,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {user.user_id}")
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[AuthSession]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        
        if session and session.expires_at > datetime.now():
            return session
        elif session:
            # Session expired
            await self.delete_session(session_id)
        
        return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False
    
    async def generate_jwt_token(self, user: User, session: AuthSession) -> str:
        """Generate JWT token for user session."""
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "email": user.email,
            "session_id": session.session_id,
            "auth_method": session.auth_method.value,
            "iat": int(time.time()),
            "exp": int(session.expires_at.timestamp())
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        return token
    
    async def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check if session still exists
            session_id = payload.get("session_id")
            if session_id and await self.get_session(session_id):
                return payload
            
            return None
            
        except jwt.InvalidTokenError:
            return None
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.expires_at <= current_time
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)