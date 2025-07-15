"""WebSocket authentication and authorization service for real-time monitoring."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import jwt
from pydantic import BaseModel, Field


class Permission(BaseModel):
    """Permission definition for WebSocket operations."""
    
    resource: str  # dashboard, metrics, alerts, system
    action: str    # read, write, subscribe, manage
    scope: Optional[str] = None  # specific resource ID or pattern


class Role(BaseModel):
    """Role with associated permissions."""
    
    name: str
    description: str
    permissions: List[Permission] = Field(default_factory=list)
    is_system_role: bool = False


class User(BaseModel):
    """User with roles and metadata."""
    
    user_id: UUID
    username: str
    email: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    is_active: bool = True
    is_admin: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None


class WebSocketSession(BaseModel):
    """WebSocket session with authentication state."""
    
    session_id: UUID = Field(default_factory=uuid4)
    connection_id: str
    user_id: Optional[UUID] = None
    username: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    permissions: List[Permission] = Field(default_factory=list)
    subscriptions: Set[str] = Field(default_factory=set)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    connected_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_authenticated: bool = False
    rate_limit_tokens: int = 100  # For rate limiting
    rate_limit_last_refill: datetime = Field(default_factory=datetime.utcnow)


class AuthenticationConfig(BaseModel):
    """Authentication configuration."""
    
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    require_authentication: bool = True
    allow_anonymous_read: bool = False
    rate_limit_requests_per_minute: int = 60
    session_timeout_minutes: int = 120
    max_connections_per_user: int = 5


class WebSocketAuthService:
    """Authentication and authorization service for WebSocket connections."""
    
    def __init__(self, config: AuthenticationConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # In-memory storage (in production, use persistent storage)
        self.users: Dict[UUID, User] = {}
        self.roles: Dict[str, Role] = {}
        self.sessions: Dict[str, WebSocketSession] = {}  # connection_id -> session
        self.user_sessions: Dict[UUID, Set[str]] = {}  # user_id -> connection_ids
        
        # Initialize default roles
        self._initialize_default_roles()
        
        self.logger.info("WebSocket authentication service initialized")
    
    def _initialize_default_roles(self):
        """Initialize default roles and permissions."""
        
        # Admin role with full permissions
        admin_role = Role(
            name="admin",
            description="Full system administrator",
            permissions=[
                Permission(resource="*", action="*"),
            ],
            is_system_role=True
        )
        
        # Dashboard viewer role
        viewer_role = Role(
            name="viewer",
            description="Dashboard viewer with read-only access",
            permissions=[
                Permission(resource="dashboard", action="read"),
                Permission(resource="metrics", action="read"),
                Permission(resource="dashboard", action="subscribe"),
                Permission(resource="metrics", action="subscribe"),
            ],
            is_system_role=True
        )
        
        # Dashboard editor role
        editor_role = Role(
            name="editor",
            description="Dashboard editor with read/write access",
            permissions=[
                Permission(resource="dashboard", action="read"),
                Permission(resource="dashboard", action="write"),
                Permission(resource="dashboard", action="subscribe"),
                Permission(resource="metrics", action="read"),
                Permission(resource="metrics", action="subscribe"),
                Permission(resource="alerts", action="read"),
                Permission(resource="alerts", action="subscribe"),
            ],
            is_system_role=True
        )
        
        # System monitor role
        monitor_role = Role(
            name="monitor",
            description="System monitoring with alerts management",
            permissions=[
                Permission(resource="dashboard", action="read"),
                Permission(resource="dashboard", action="subscribe"),
                Permission(resource="metrics", action="read"),
                Permission(resource="metrics", action="subscribe"),
                Permission(resource="alerts", action="read"),
                Permission(resource="alerts", action="write"),
                Permission(resource="alerts", action="subscribe"),
                Permission(resource="system", action="read"),
            ],
            is_system_role=True
        )
        
        # Store roles
        self.roles["admin"] = admin_role
        self.roles["viewer"] = viewer_role
        self.roles["editor"] = editor_role
        self.roles["monitor"] = monitor_role
    
    async def authenticate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate JWT token and return user claims."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                return None
            
            return payload
            
        except jwt.InvalidTokenError as e:
            self.logger.debug(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error authenticating token: {e}")
            return None
    
    async def create_session(
        self,
        connection_id: str,
        token: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[WebSocketSession]:
        """Create a new WebSocket session."""
        
        session = WebSocketSession(
            connection_id=connection_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Handle authentication
        if token:
            claims = await self.authenticate_token(token)
            if claims:
                user_id_str = claims.get("user_id")
                if user_id_str:
                    try:
                        user_id = UUID(user_id_str)
                        user = self.users.get(user_id)
                        
                        if user and user.is_active:
                            # Check connection limits
                            user_connections = self.user_sessions.get(user_id, set())
                            if len(user_connections) >= self.config.max_connections_per_user:
                                self.logger.warning(f"User {user.username} exceeded connection limit")
                                return None
                            
                            # Set up authenticated session
                            session.user_id = user_id
                            session.username = user.username
                            session.roles = user.roles.copy()
                            session.permissions = self._get_user_permissions(user)
                            session.is_authenticated = True
                            
                            # Update user's last login
                            user.last_login = datetime.utcnow()
                            
                            # Track user sessions
                            if user_id not in self.user_sessions:
                                self.user_sessions[user_id] = set()
                            self.user_sessions[user_id].add(connection_id)
                            
                            self.logger.info(f"Authenticated session for user {user.username}")
                        else:
                            self.logger.warning(f"User {user_id} not found or inactive")
                            return None
                    except ValueError:
                        self.logger.warning(f"Invalid user_id in token: {user_id_str}")
                        return None
        
        # Handle anonymous access if allowed
        elif self.config.allow_anonymous_read:
            session.roles = ["viewer"]
            session.permissions = self._get_role_permissions("viewer")
            self.logger.debug("Created anonymous session with viewer permissions")
        
        # Reject if authentication required but not provided
        elif self.config.require_authentication:
            self.logger.debug("Authentication required but not provided")
            return None
        
        # Store session
        self.sessions[connection_id] = session
        
        return session
    
    async def close_session(self, connection_id: str) -> bool:
        """Close a WebSocket session."""
        
        if connection_id not in self.sessions:
            return False
        
        session = self.sessions[connection_id]
        
        # Remove from user sessions tracking
        if session.user_id and session.user_id in self.user_sessions:
            self.user_sessions[session.user_id].discard(connection_id)
            if not self.user_sessions[session.user_id]:
                del self.user_sessions[session.user_id]
        
        # Remove session
        del self.sessions[connection_id]
        
        self.logger.debug(f"Closed session for connection {connection_id}")
        return True
    
    async def check_permission(
        self,
        connection_id: str,
        resource: str,
        action: str,
        scope: Optional[str] = None
    ) -> bool:
        """Check if a session has permission for a specific action."""
        
        if connection_id not in self.sessions:
            return False
        
        session = self.sessions[connection_id]
        
        # Update activity timestamp
        session.last_activity = datetime.utcnow()
        
        # Check session timeout
        if self._is_session_expired(session):
            await self.close_session(connection_id)
            return False
        
        # Check rate limiting
        if not self._check_rate_limit(session):
            return False
        
        # Check permissions
        for permission in session.permissions:
            if self._permission_matches(permission, resource, action, scope):
                return True
        
        return False
    
    def _permission_matches(
        self,
        permission: Permission,
        resource: str,
        action: str,
        scope: Optional[str] = None
    ) -> bool:
        """Check if a permission matches the requested access."""
        
        # Check wildcard permissions
        if permission.resource == "*" and permission.action == "*":
            return True
        
        # Check resource match
        if permission.resource != "*" and permission.resource != resource:
            return False
        
        # Check action match
        if permission.action != "*" and permission.action != action:
            return False
        
        # Check scope if specified
        if permission.scope and scope:
            if permission.scope != scope and not scope.startswith(permission.scope):
                return False
        
        return True
    
    def _get_user_permissions(self, user: User) -> List[Permission]:
        """Get all permissions for a user based on their roles."""
        permissions = []
        
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role:
                permissions.extend(role.permissions)
        
        # Add admin permissions if user is admin
        if user.is_admin:
            admin_role = self.roles.get("admin")
            if admin_role:
                permissions.extend(admin_role.permissions)
        
        return permissions
    
    def _get_role_permissions(self, role_name: str) -> List[Permission]:
        """Get permissions for a specific role."""
        role = self.roles.get(role_name)
        return role.permissions if role else []
    
    def _is_session_expired(self, session: WebSocketSession) -> bool:
        """Check if a session has expired."""
        timeout = timedelta(minutes=self.config.session_timeout_minutes)
        return datetime.utcnow() - session.last_activity > timeout
    
    def _check_rate_limit(self, session: WebSocketSession) -> bool:
        """Check and update rate limiting for a session."""
        now = datetime.utcnow()
        
        # Refill rate limit tokens based on time elapsed
        time_elapsed = (now - session.rate_limit_last_refill).total_seconds()
        tokens_to_add = int(time_elapsed * self.config.rate_limit_requests_per_minute / 60)
        
        if tokens_to_add > 0:
            session.rate_limit_tokens = min(
                session.rate_limit_tokens + tokens_to_add,
                self.config.rate_limit_requests_per_minute
            )
            session.rate_limit_last_refill = now
        
        # Check if request is allowed
        if session.rate_limit_tokens > 0:
            session.rate_limit_tokens -= 1
            return True
        
        return False
    
    async def subscribe_to_topic(
        self,
        connection_id: str,
        topic: str
    ) -> bool:
        """Subscribe a session to a topic with permission check."""
        
        # Check subscription permission
        if not await self.check_permission(connection_id, topic, "subscribe"):
            return False
        
        if connection_id in self.sessions:
            session = self.sessions[connection_id]
            session.subscriptions.add(topic)
            self.logger.debug(f"Session {connection_id} subscribed to {topic}")
            return True
        
        return False
    
    async def unsubscribe_from_topic(
        self,
        connection_id: str,
        topic: str
    ) -> bool:
        """Unsubscribe a session from a topic."""
        
        if connection_id in self.sessions:
            session = self.sessions[connection_id]
            session.subscriptions.discard(topic)
            self.logger.debug(f"Session {connection_id} unsubscribed from {topic}")
            return True
        
        return False
    
    async def get_session_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        
        if connection_id not in self.sessions:
            return None
        
        session = self.sessions[connection_id]
        
        return {
            "session_id": str(session.session_id),
            "connection_id": session.connection_id,
            "user_id": str(session.user_id) if session.user_id else None,
            "username": session.username,
            "roles": session.roles,
            "is_authenticated": session.is_authenticated,
            "connected_at": session.connected_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "subscriptions": list(session.subscriptions),
            "rate_limit_tokens": session.rate_limit_tokens
        }
    
    async def create_user(
        self,
        username: str,
        email: Optional[str] = None,
        roles: Optional[List[str]] = None,
        is_admin: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        """Create a new user."""
        
        user = User(
            user_id=uuid4(),
            username=username,
            email=email,
            roles=roles or ["viewer"],
            is_admin=is_admin,
            metadata=metadata or {}
        )
        
        self.users[user.user_id] = user
        
        self.logger.info(f"Created user {username} with roles {user.roles}")
        return user
    
    async def generate_token(self, user_id: UUID) -> Optional[str]:
        """Generate JWT token for a user."""
        
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return None
        
        try:
            payload = {
                "user_id": str(user_id),
                "username": user.username,
                "roles": user.roles,
                "is_admin": user.is_admin,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours)
            }
            
            token = jwt.encode(
                payload,
                self.config.jwt_secret,
                algorithm=self.config.jwt_algorithm
            )
            
            return token
            
        except Exception as e:
            self.logger.error(f"Error generating token for user {user_id}: {e}")
            return None
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        
        expired_connections = []
        
        for connection_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_connections.append(connection_id)
        
        for connection_id in expired_connections:
            await self.close_session(connection_id)
        
        if expired_connections:
            self.logger.info(f"Cleaned up {len(expired_connections)} expired sessions")
    
    async def get_user_sessions(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get all active sessions for a user."""
        
        user_connections = self.user_sessions.get(user_id, set())
        sessions_info = []
        
        for connection_id in user_connections:
            session_info = await self.get_session_info(connection_id)
            if session_info:
                sessions_info.append(session_info)
        
        return sessions_info
    
    async def revoke_user_sessions(self, user_id: UUID) -> int:
        """Revoke all sessions for a user."""
        
        user_connections = self.user_sessions.get(user_id, set()).copy()
        revoked_count = 0
        
        for connection_id in user_connections:
            if await self.close_session(connection_id):
                revoked_count += 1
        
        self.logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
        return revoked_count
    
    async def get_security_summary(self) -> Dict[str, Any]:
        """Get security and authentication summary."""
        
        active_sessions = len(self.sessions)
        authenticated_sessions = sum(
            1 for session in self.sessions.values() 
            if session.is_authenticated
        )
        
        unique_users = len(self.user_sessions)
        
        return {
            "authentication_config": {
                "require_authentication": self.config.require_authentication,
                "allow_anonymous_read": self.config.allow_anonymous_read,
                "session_timeout_minutes": self.config.session_timeout_minutes,
                "max_connections_per_user": self.config.max_connections_per_user,
                "rate_limit_per_minute": self.config.rate_limit_requests_per_minute
            },
            "current_status": {
                "total_sessions": active_sessions,
                "authenticated_sessions": authenticated_sessions,
                "anonymous_sessions": active_sessions - authenticated_sessions,
                "unique_users_connected": unique_users,
                "total_users": len(self.users),
                "total_roles": len(self.roles)
            },
            "security_metrics": {
                "authentication_rate": authenticated_sessions / max(active_sessions, 1),
                "average_connections_per_user": active_sessions / max(unique_users, 1)
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# Convenience function for creating auth service
def create_websocket_auth_service(
    jwt_secret: Optional[str] = None,
    require_authentication: bool = True
) -> WebSocketAuthService:
    """Create and configure WebSocket authentication service."""
    
    config = AuthenticationConfig(
        jwt_secret=jwt_secret or secrets.token_urlsafe(32),
        require_authentication=require_authentication
    )
    
    return WebSocketAuthService(config)