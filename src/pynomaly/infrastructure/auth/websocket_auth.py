"""WebSocket and HTMX authentication middleware."""

import json
import logging
from typing import Optional
from urllib.parse import parse_qs

from fastapi import HTTPException, WebSocket, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .jwt_auth import JWTAuthService, get_auth

logger = logging.getLogger(__name__)


class WebSocketAuthMiddleware:
    """Authentication middleware for WebSocket connections."""
    
    def __init__(self, auth_service: JWTAuthService):
        """Initialize with auth service.
        
        Args:
            auth_service: JWT authentication service
        """
        self.auth_service = auth_service
    
    async def authenticate_websocket(self, websocket: WebSocket) -> Optional[dict]:
        """Authenticate WebSocket connection.
        
        WebSocket authentication can be done via:
        1. Query parameters: ?token=jwt_token or ?api_key=api_key
        2. Headers (if supported by client)
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            User data if authenticated, None otherwise
        """
        try:
            # Try query parameters first
            query_params = parse_qs(str(websocket.url.query))
            
            # Check for JWT token
            if 'token' in query_params:
                token = query_params['token'][0]
                try:
                    token_payload = self.auth_service.decode_token(token)
                    user = self.auth_service.get_user(token_payload.sub)
                    if user and user.is_active:
                        logger.info(f"WebSocket authenticated user: {user.username}")
                        return user
                except Exception as e:
                    logger.warning(f"WebSocket JWT authentication failed: {e}")
            
            # Check for API key
            if 'api_key' in query_params:
                api_key = query_params['api_key'][0]
                try:
                    user = self.auth_service.authenticate_api_key(api_key)
                    if user:
                        logger.info(f"WebSocket authenticated user via API key: {user.username}")
                        return user
                except Exception as e:
                    logger.warning(f"WebSocket API key authentication failed: {e}")
            
            # Try headers as fallback (some WebSocket clients support this)
            headers = dict(websocket.headers)
            if 'authorization' in headers:
                auth_header = headers['authorization']
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]
                    
                    # Check if it's an API key
                    if token.startswith('pyn_'):
                        try:
                            user = self.auth_service.authenticate_api_key(token)
                            if user:
                                logger.info(f"WebSocket authenticated user via header API key: {user.username}")
                                return user
                        except Exception as e:
                            logger.warning(f"WebSocket header API key authentication failed: {e}")
                    else:
                        # JWT token
                        try:
                            token_payload = self.auth_service.decode_token(token)
                            user = self.auth_service.get_user(token_payload.sub)
                            if user and user.is_active:
                                logger.info(f"WebSocket authenticated user via header JWT: {user.username}")
                                return user
                        except Exception as e:
                            logger.warning(f"WebSocket header JWT authentication failed: {e}")
            
            logger.warning("WebSocket authentication failed: no valid credentials")
            return None
            
        except Exception as e:
            logger.error(f"WebSocket authentication error: {e}")
            return None
    
    async def require_authentication(self, websocket: WebSocket) -> dict:
        """Require authentication for WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            User data
            
        Raises:
            WebSocketException: If authentication fails
        """
        user = await self.authenticate_websocket(websocket)
        if not user:
            await websocket.close(code=4001, reason="Authentication required")
            raise Exception("Authentication required")
        return user


class HTMXAuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for HTMX requests."""
    
    def __init__(self, app, auth_service: JWTAuthService):
        """Initialize with app and auth service.
        
        Args:
            app: ASGI application
            auth_service: JWT authentication service
        """
        super().__init__(app)
        self.auth_service = auth_service
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and add authentication context for HTMX.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Check if this is an HTMX request
        is_htmx = request.headers.get('hx-request') == 'true'
        
        if is_htmx:
            # Add authentication context for HTMX requests
            user = await self._authenticate_request(request)
            if user:
                # Add user to request state for handlers to access
                request.state.user = user
                request.state.authenticated = True
            else:
                request.state.user = None
                request.state.authenticated = False
        
        # Process the request
        response = await call_next(request)
        
        # Add authentication headers for HTMX responses if needed
        if is_htmx and hasattr(request.state, 'authenticated'):
            if not request.state.authenticated:
                # Add header to trigger authentication on client side
                response.headers['HX-Trigger'] = json.dumps({
                    'authRequired': {'message': 'Authentication required'}
                })
        
        return response
    
    async def _authenticate_request(self, request: Request) -> Optional[dict]:
        """Authenticate HTTP request.
        
        Args:
            request: HTTP request
            
        Returns:
            User data if authenticated, None otherwise
        """
        try:
            # Check Authorization header
            auth_header = request.headers.get('authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header[7:]
                
                # Check if it's an API key
                if token.startswith('pyn_'):
                    try:
                        user = self.auth_service.authenticate_api_key(token)
                        if user:
                            return user
                    except Exception as e:
                        logger.warning(f"API key authentication failed: {e}")
                else:
                    # JWT token
                    try:
                        token_payload = self.auth_service.decode_token(token)
                        user = self.auth_service.get_user(token_payload.sub)
                        if user and user.is_active:
                            return user
                    except Exception as e:
                        logger.warning(f"JWT authentication failed: {e}")
            
            # Check cookies for session-based auth (fallback)
            session_token = request.cookies.get('session_token')
            if session_token:
                try:
                    token_payload = self.auth_service.decode_token(session_token)
                    user = self.auth_service.get_user(token_payload.sub)
                    if user and user.is_active:
                        return user
                except Exception as e:
                    logger.warning(f"Session token authentication failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Request authentication error: {e}")
            return None


def create_websocket_auth_dependency(auth_service: JWTAuthService):
    """Create WebSocket authentication dependency.
    
    Args:
        auth_service: JWT authentication service
        
    Returns:
        Authentication dependency function
    """
    middleware = WebSocketAuthMiddleware(auth_service)
    
    async def websocket_auth_dependency(websocket: WebSocket):
        """WebSocket authentication dependency."""
        return await middleware.require_authentication(websocket)
    
    return websocket_auth_dependency


def get_htmx_user(request: Request) -> Optional[dict]:
    """Get authenticated user from HTMX request state.
    
    Args:
        request: HTTP request
        
    Returns:
        User data if authenticated, None otherwise
    """
    return getattr(request.state, 'user', None)


def require_htmx_auth(request: Request) -> dict:
    """Require authentication for HTMX request.
    
    Args:
        request: HTTP request
        
    Returns:
        User data
        
    Raises:
        HTTPException: If not authenticated
    """
    user = get_htmx_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user
