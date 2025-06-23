"""User action tracking and API access logging middleware.

This module provides comprehensive tracking of user actions and API access including:
- Request/response logging
- User action tracking  
- API endpoint access monitoring
- Session activity tracking
- Performance metrics
- Security event correlation
"""

from __future__ import annotations

import json
import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

from .audit_logger import AuditLogger, SecurityEventType, AuditLevel, audit_context
from .security_monitor import SecurityMonitor

logger = logging.getLogger(__name__)


@dataclass
class RequestInfo:
    """Information about an HTTP request."""
    
    request_id: str
    method: str
    path: str
    query_params: Dict[str, Any]
    headers: Dict[str, str]
    user_agent: Optional[str]
    ip_address: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    body_size: int = 0
    content_type: Optional[str] = None


@dataclass
class ResponseInfo:
    """Information about an HTTP response."""
    
    status_code: int
    headers: Dict[str, str]
    body_size: int
    processing_time: float
    timestamp: datetime
    error_message: Optional[str] = None


@dataclass
class UserAction:
    """Represents a user action."""
    
    action_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    action_type: str
    resource: Optional[str]
    description: str
    timestamp: datetime
    ip_address: str
    user_agent: Optional[str]
    
    # Request details
    http_method: str
    endpoint: str
    query_params: Dict[str, Any]
    
    # Response details
    status_code: int
    processing_time: float
    success: bool
    
    # Metadata
    metadata: Dict[str, Any]


class SensitiveDataFilter:
    """Filter for removing sensitive data from logs."""
    
    def __init__(self):
        # Sensitive field patterns
        self.sensitive_fields = {
            'password', 'passwd', 'pass', 'pwd',
            'secret', 'token', 'key', 'api_key', 'apikey',
            'authorization', 'auth', 'bearer',
            'credit_card', 'creditcard', 'ccn', 'pan',
            'ssn', 'social_security', 'social_security_number',
            'phone', 'telephone', 'mobile',
            'email', 'mail',
        }
        
        # Sensitive header patterns
        self.sensitive_headers = {
            'authorization', 'cookie', 'set-cookie',
            'x-api-key', 'x-auth-token', 'x-csrf-token',
            'www-authenticate', 'proxy-authorization'
        }
        
        # Sensitive URL patterns
        self.sensitive_paths = {
            '/auth/', '/login', '/logout', '/register',
            '/password', '/reset', '/token',
            '/api/auth/', '/api/users/', '/admin/'
        }
    
    def filter_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter sensitive headers."""
        filtered = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in self.sensitive_headers):
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value
        return filtered
    
    def filter_query_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive query parameters."""
        filtered = {}
        for key, value in params.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value
        return filtered
    
    def filter_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive body fields."""
        if not isinstance(body, dict):
            return body
        
        filtered = {}
        for key, value in body.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = self.filter_body(value)
            else:
                filtered[key] = value
        return filtered
    
    def is_sensitive_path(self, path: str) -> bool:
        """Check if path contains sensitive information."""
        path_lower = path.lower()
        return any(sensitive in path_lower for sensitive in self.sensitive_paths)


class UserActionTracker:
    """Service for tracking user actions."""
    
    def __init__(self, 
                 audit_logger: Optional[AuditLogger] = None,
                 security_monitor: Optional[SecurityMonitor] = None):
        """Initialize user action tracker.
        
        Args:
            audit_logger: Audit logger instance
            security_monitor: Security monitor instance
        """
        self.audit_logger = audit_logger or AuditLogger()
        self.security_monitor = security_monitor
        self.data_filter = SensitiveDataFilter()
        
        # Configuration
        self.log_request_bodies = True
        self.log_response_bodies = False
        self.max_body_size = 10240  # 10KB
        self.track_file_downloads = True
        self.track_data_modifications = True
        
        # Session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.user_sessions: Dict[str, Set[str]] = {}
    
    async def track_request_start(self, request: Request) -> RequestInfo:
        """Track the start of a request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Request information
        """
        request_id = str(uuid.uuid4())
        
        # Extract user information
        user_id = None
        session_id = None
        
        # Try to get user from request state (set by auth middleware)
        if hasattr(request.state, 'user') and request.state.user:
            user_id = getattr(request.state.user, 'id', None)
        
        # Get session ID from cookies or headers
        session_id = request.cookies.get('session_id') or request.headers.get('X-Session-ID')
        
        # Extract request details
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get('user-agent')
        
        # Parse query parameters
        query_params = dict(request.query_params)
        filtered_params = self.data_filter.filter_query_params(query_params)
        
        # Filter headers
        headers = dict(request.headers)
        filtered_headers = self.data_filter.filter_headers(headers)
        
        request_info = RequestInfo(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=filtered_params,
            headers=filtered_headers,
            user_agent=user_agent,
            ip_address=ip_address,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            session_id=session_id,
            content_type=request.headers.get('content-type')
        )
        
        # Store request info in request state
        request.state.request_info = request_info
        request.state.start_time = time.time()
        
        # Update session tracking
        if session_id:
            self._update_session_activity(session_id, user_id, ip_address, user_agent)
        
        return request_info
    
    async def track_request_end(self, 
                               request: Request, 
                               response: Response,
                               request_info: RequestInfo) -> UserAction:
        """Track the end of a request.
        
        Args:
            request: FastAPI request object
            response: FastAPI response object
            request_info: Request information from start
            
        Returns:
            User action record
        """
        end_time = time.time()
        start_time = getattr(request.state, 'start_time', end_time)
        processing_time = end_time - start_time
        
        # Create response info
        response_info = ResponseInfo(
            status_code=response.status_code,
            headers=self.data_filter.filter_headers(dict(response.headers)),
            body_size=len(getattr(response, 'body', b'')),
            processing_time=processing_time,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Determine action type and description
        action_type, description = self._classify_action(request_info, response_info)
        
        # Create user action
        user_action = UserAction(
            action_id=str(uuid.uuid4()),
            user_id=request_info.user_id,
            session_id=request_info.session_id,
            action_type=action_type,
            resource=self._extract_resource(request_info),
            description=description,
            timestamp=request_info.timestamp,
            ip_address=request_info.ip_address,
            user_agent=request_info.user_agent,
            http_method=request_info.method,
            endpoint=request_info.path,
            query_params=request_info.query_params,
            status_code=response_info.status_code,
            processing_time=processing_time,
            success=200 <= response_info.status_code < 400,
            metadata={
                'request_id': request_info.request_id,
                'request_size': request_info.body_size,
                'response_size': response_info.body_size,
                'content_type': request_info.content_type,
                'is_sensitive_path': self.data_filter.is_sensitive_path(request_info.path)
            }
        )
        
        # Log the action
        await self._log_user_action(user_action)
        
        # Report to security monitor if available
        if self.security_monitor:
            await self._report_to_security_monitor(user_action)
        
        return user_action
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check X-Forwarded-For header first
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(',')[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip.strip()
        
        # Fall back to client host
        if request.client:
            return request.client.host
        
        return 'unknown'
    
    def _update_session_activity(self, 
                                session_id: str, 
                                user_id: Optional[str],
                                ip_address: str,
                                user_agent: Optional[str]) -> None:
        """Update session activity tracking."""
        current_time = time.time()
        
        # Update session info
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                'created_at': current_time,
                'user_id': user_id,
                'ip_addresses': {ip_address},
                'user_agents': {user_agent} if user_agent else set(),
                'request_count': 0,
                'last_activity': current_time
            }
        
        session_info = self.active_sessions[session_id]
        session_info['last_activity'] = current_time
        session_info['request_count'] += 1
        session_info['ip_addresses'].add(ip_address)
        if user_agent:
            session_info['user_agents'].add(user_agent)
        
        # Update user session mapping
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
            session_info['user_id'] = user_id
    
    def _classify_action(self, 
                        request_info: RequestInfo, 
                        response_info: ResponseInfo) -> tuple[str, str]:
        """Classify the type of action and generate description."""
        method = request_info.method.upper()
        path = request_info.path
        status_code = response_info.status_code
        
        # Authentication actions
        if 'auth' in path or 'login' in path:
            if method == 'POST' and status_code == 200:
                return 'authentication', f'User login to {path}'
            elif method == 'POST' and status_code >= 400:
                return 'authentication_failed', f'Failed login attempt to {path}'
            elif method == 'DELETE' or 'logout' in path:
                return 'authentication', f'User logout from {path}'
        
        # Data access actions
        if method == 'GET':
            if status_code == 200:
                return 'data_read', f'Read data from {path}'
            elif status_code == 403:
                return 'access_denied', f'Access denied to {path}'
            elif status_code == 404:
                return 'resource_not_found', f'Resource not found: {path}'
        
        # Data modification actions
        elif method == 'POST':
            if 200 <= status_code < 300:
                return 'data_create', f'Created resource at {path}'
            else:
                return 'data_create_failed', f'Failed to create resource at {path}'
        
        elif method == 'PUT' or method == 'PATCH':
            if 200 <= status_code < 300:
                return 'data_update', f'Updated resource at {path}'
            else:
                return 'data_update_failed', f'Failed to update resource at {path}'
        
        elif method == 'DELETE':
            if 200 <= status_code < 300:
                return 'data_delete', f'Deleted resource at {path}'
            else:
                return 'data_delete_failed', f'Failed to delete resource at {path}'
        
        # API access
        if path.startswith('/api/'):
            return 'api_access', f'{method} request to API endpoint {path}'
        
        # File downloads
        if 'download' in path or request_info.headers.get('accept', '').startswith('application/'):
            return 'file_download', f'File download from {path}'
        
        # Default
        return 'http_request', f'{method} request to {path}'
    
    def _extract_resource(self, request_info: RequestInfo) -> Optional[str]:
        """Extract resource identifier from request."""
        path_parts = request_info.path.strip('/').split('/')
        
        # Try to find resource ID in path
        if len(path_parts) >= 2:
            # Common patterns: /api/resource/id, /resource/id
            if path_parts[0] == 'api' and len(path_parts) >= 3:
                return f"{path_parts[1]}/{path_parts[2] if len(path_parts) > 2 else ''}"
            else:
                return f"{path_parts[0]}/{path_parts[1] if len(path_parts) > 1 else ''}"
        
        return request_info.path
    
    async def _log_user_action(self, action: UserAction) -> None:
        """Log user action to audit trail."""
        # Create audit context
        correlation_id = action.action_id
        
        with audit_context(
            correlation_id=correlation_id,
            user_id=action.user_id,
            session_id=action.session_id,
            ip_address=action.ip_address,
            user_agent=action.user_agent
        ):
            # Log as security event if authentication related
            if action.action_type in ['authentication', 'authentication_failed', 'access_denied']:
                event_type = {
                    'authentication': SecurityEventType.AUTH_LOGIN_SUCCESS,
                    'authentication_failed': SecurityEventType.AUTH_LOGIN_FAILURE,
                    'access_denied': SecurityEventType.AUTHZ_ACCESS_DENIED
                }.get(action.action_type, SecurityEventType.API_ENDPOINT_ACCESS)
                
                self.audit_logger.log_security_event(
                    event_type=event_type,
                    message=action.description,
                    level=AuditLevel.WARNING if not action.success else AuditLevel.INFO,
                    details={
                        'action_type': action.action_type,
                        'endpoint': action.endpoint,
                        'method': action.http_method,
                        'status_code': action.status_code,
                        'processing_time': action.processing_time,
                        'resource': action.resource,
                        'metadata': action.metadata
                    }
                )
            else:
                # Log as general audit event
                self.audit_logger.log_audit_event(
                    event_type='user_action',
                    action=action.action_type,
                    resource=action.resource,
                    message=action.description,
                    level=AuditLevel.ERROR if not action.success else AuditLevel.INFO,
                    details={
                        'endpoint': action.endpoint,
                        'method': action.http_method,
                        'status_code': action.status_code,
                        'processing_time': action.processing_time,
                        'query_params': action.query_params,
                        'metadata': action.metadata
                    }
                )
    
    async def _report_to_security_monitor(self, action: UserAction) -> None:
        """Report action to security monitor for analysis."""
        event_data = {
            'event_type': {
                'authentication_failed': SecurityEventType.AUTH_LOGIN_FAILURE,
                'access_denied': SecurityEventType.AUTHZ_ACCESS_DENIED,
            }.get(action.action_type),
            'user_id': action.user_id,
            'ip_address': action.ip_address,
            'user_agent': action.user_agent,
            'session_id': action.session_id,
            'timestamp': action.timestamp.timestamp(),
            'details': {
                'endpoint': action.endpoint,
                'method': action.http_method,
                'status_code': action.status_code,
                'action_type': action.action_type,
                'resource': action.resource
            }
        }
        
        if event_data['event_type']:
            await self.security_monitor.process_event(event_data)
    
    def get_user_activity_summary(self, user_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get activity summary for a user.
        
        Args:
            user_id: User ID
            hours: Number of hours to look back
            
        Returns:
            Activity summary
        """
        # This would typically query a database
        # For now, return session information
        user_sessions = self.user_sessions.get(user_id, set())
        
        summary = {
            'user_id': user_id,
            'active_sessions': len(user_sessions),
            'session_details': []
        }
        
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        for session_id in user_sessions:
            session_info = self.active_sessions.get(session_id)
            if session_info and session_info['last_activity'] > cutoff_time:
                summary['session_details'].append({
                    'session_id': session_id,
                    'created_at': datetime.fromtimestamp(session_info['created_at']),
                    'last_activity': datetime.fromtimestamp(session_info['last_activity']),
                    'request_count': session_info['request_count'],
                    'ip_addresses': list(session_info['ip_addresses']),
                    'user_agents': list(session_info['user_agents'])
                })
        
        return summary


class UserTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking user actions and API access."""
    
    def __init__(self, app, tracker: Optional[UserActionTracker] = None):
        """Initialize user tracking middleware.
        
        Args:
            app: FastAPI application
            tracker: User action tracker instance
        """
        super().__init__(app)
        self.tracker = tracker or UserActionTracker()
        logger.info("User tracking middleware initialized")
    
    async def dispatch(self, request: Request, call_next):
        """Process request and track user actions."""
        # Track request start
        request_info = await self.tracker.track_request_start(request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Track request end
            await self.tracker.track_request_end(request, response, request_info)
            
            return response
            
        except Exception as e:
            # Track failed request
            error_response = Response(
                content=str(e),
                status_code=500,
                headers={"Content-Type": "text/plain"}
            )
            
            await self.tracker.track_request_end(request, error_response, request_info)
            
            # Re-raise the exception
            raise


# Global user tracker instance
_user_tracker: Optional[UserActionTracker] = None


def get_user_tracker() -> UserActionTracker:
    """Get global user action tracker instance."""
    global _user_tracker
    if _user_tracker is None:
        _user_tracker = UserActionTracker()
    return _user_tracker


def init_user_tracker(
    audit_logger: Optional[AuditLogger] = None,
    security_monitor: Optional[SecurityMonitor] = None
) -> UserActionTracker:
    """Initialize global user action tracker.
    
    Args:
        audit_logger: Audit logger instance
        security_monitor: Security monitor instance
        
    Returns:
        User action tracker instance
    """
    global _user_tracker
    _user_tracker = UserActionTracker(audit_logger, security_monitor)
    return _user_tracker