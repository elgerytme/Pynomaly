"""Security middleware for FastAPI."""

from typing import Callable

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials

from .auth import get_current_user_from_token, security


class AuthMiddleware:
    """Authentication middleware."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        
        # Protected endpoints that require authentication
        self.protected_paths = [
            "/api/v1/assets",
            "/api/v1/lineage", 
            "/api/v1/pipeline",
            "/api/v1/quality",
            "/api/v1/dashboard"
        ]
        
        # Public endpoints that don't require authentication
        self.public_paths = [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login"
        ]
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request through auth middleware."""
        path = request.url.path
        
        # Skip authentication for public paths
        if any(path.startswith(public_path) for public_path in self.public_paths):
            return await call_next(request)
        
        # Require authentication for protected paths
        if any(path.startswith(protected_path) for protected_path in self.protected_paths):
            try:
                # Get authorization header
                authorization = request.headers.get("Authorization")
                if not authorization:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authorization header missing",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                
                # Parse bearer token
                scheme, token = authorization.split(" ", 1)
                if scheme.lower() != "bearer":
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid authentication scheme",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                
                # Validate token and get user
                credentials = HTTPAuthorizationCredentials(
                    scheme=scheme,
                    credentials=token
                )
                user = get_current_user_from_token(credentials)
                
                # Add user to request state
                request.state.user = user
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Authentication failed: {e}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        return await call_next(request)


class SecurityHeadersMiddleware:
    """Security headers middleware."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY" 
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self';"
        )
        
        return response


def setup_security_middleware(app: FastAPI) -> None:
    """Setup security middleware for the application."""
    app.middleware("http")(SecurityHeadersMiddleware(app))
    app.middleware("http")(AuthMiddleware(app))