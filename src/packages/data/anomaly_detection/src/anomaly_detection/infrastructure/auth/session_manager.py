"""Session management utilities."""

import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class Session:
    """User session data."""
    
    def __init__(self, session_id: str, user_id: str, expires_at: datetime):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.utcnow()
        self.expires_at = expires_at
        self.last_activity = datetime.utcnow()
        self.data: Dict[str, Any] = {}
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    def extend(self, duration: timedelta) -> None:
        """Extend session expiration."""
        self.expires_at = datetime.utcnow() + duration
        self.last_activity = datetime.utcnow()


class SessionManager:
    """Manages user sessions."""
    
    def __init__(self, default_timeout: int = 3600):
        """Initialize session manager."""
        self.sessions: Dict[str, Session] = {}
        self.default_timeout = default_timeout
        
    def create_session(
        self,
        user_id: str,
        timeout: Optional[int] = None
    ) -> str:
        """Create a new session."""
        session_id = secrets.token_urlsafe(32)
        timeout = timeout or self.default_timeout
        expires_at = datetime.utcnow() + timedelta(seconds=timeout)
        
        session = Session(session_id, user_id, expires_at)
        self.sessions[session_id] = session
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        session = self.sessions.get(session_id)
        
        if session and session.is_expired:
            del self.sessions[session_id]
            return None
            
        if session:
            session.last_activity = datetime.utcnow()
        
        return session
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data."""
        session = self.get_session(session_id)
        if session:
            session.data.update(data)
            return True
        return False
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            
        return len(expired_sessions)
    
    def get_user_sessions(self, user_id: str) -> list[Session]:
        """Get all active sessions for a user."""
        return [
            session for session in self.sessions.values()
            if session.user_id == user_id and not session.is_expired
        ]