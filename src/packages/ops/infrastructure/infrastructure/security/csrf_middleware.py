"""
Enhanced CSRF Protection Middleware

Provides comprehensive CSRF protection with:
- Double-submit cookie pattern
- SameSite cookie attributes
- Middleware integration
- Configurable protection levels
"""

import secrets

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from pynomaly.infrastructure.config.settings import get_settings


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """Enhanced CSRF protection middleware with double-submit cookie pattern."""

    def __init__(
        self,
        app: ASGIApp,
        cookie_name: str = "csrftoken",
        header_name: str = "X-CSRFToken",
        exempt_paths: set[str] | None = None,
        require_https: bool = True,
        max_age: int = 3600,
    ):
        super().__init__(app)
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.exempt_paths = exempt_paths or {
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        }
        self.require_https = require_https
        self.max_age = max_age
        self.settings = get_settings()

        # Safe HTTP methods that don't require CSRF protection
        self.safe_methods = {"GET", "HEAD", "OPTIONS", "TRACE"}

    async def dispatch(self, request: Request, call_next):
        """Process request with CSRF protection."""

        # Skip protection for exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)

        # Skip protection for safe HTTP methods
        if request.method in self.safe_methods:
            response = await call_next(request)
            # Set CSRF cookie for safe methods
            self._set_csrf_cookie(request, response)
            return response

        # Validate CSRF token for unsafe methods
        if not self._validate_csrf_token(request):
            return self._csrf_failure_response(request)

        # Process the request
        response = await call_next(request)

        # Refresh CSRF cookie
        self._set_csrf_cookie(request, response)

        return response

    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from CSRF protection."""
        # Check exact matches
        if path in self.exempt_paths:
            return True

        # Check wildcard patterns
        for exempt_path in self.exempt_paths:
            if exempt_path.endswith("*") and path.startswith(exempt_path[:-1]):
                return True

        return False

    def _validate_csrf_token(self, request: Request) -> bool:
        """Validate CSRF token using double-submit cookie pattern."""

        # Get token from cookie
        cookie_token = request.cookies.get(self.cookie_name)
        if not cookie_token:
            return False

        # Get token from header or form data
        header_token = request.headers.get(self.header_name)

        # For form submissions, also check form data
        form_token = None
        if hasattr(request, "_form") and request._form:
            form_token = request._form.get("csrfmiddlewaretoken")

        # At least one token source must be present
        submitted_token = header_token or form_token
        if not submitted_token:
            return False

        # Tokens must match (constant-time comparison)
        return secrets.compare_digest(cookie_token, submitted_token)

    def _set_csrf_cookie(self, request: Request, response: Response) -> None:
        """Set CSRF cookie with secure attributes."""

        # Generate new token if not present
        current_token = request.cookies.get(self.cookie_name)
        if not current_token:
            current_token = secrets.token_urlsafe(32)

        # Set cookie with secure attributes
        response.set_cookie(
            key=self.cookie_name,
            value=current_token,
            max_age=self.max_age,
            httponly=False,  # Must be accessible to JavaScript
            secure=self.require_https and self.settings.environment == "production",
            samesite="strict" if self.settings.environment == "production" else "lax",
            path="/",
        )

    def _csrf_failure_response(self, request: Request) -> Response:
        """Return appropriate response for CSRF validation failure."""

        error_data = {
            "error": "CSRF_FAILURE",
            "message": "CSRF token missing or invalid",
            "code": "CSRF_001",
        }

        # Return JSON for API requests
        if request.url.path.startswith(
            "/api/"
        ) or "application/json" in request.headers.get("accept", ""):
            return JSONResponse(status_code=403, content=error_data)

        # Return HTML for web requests
        return Response(
            content=self._get_csrf_error_html(), status_code=403, media_type="text/html"
        )

    def _get_csrf_error_html(self) -> str:
        """Get HTML content for CSRF error page."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CSRF Protection Error</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 600px;
                    margin: 50px auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                .error-box {
                    background: #fee;
                    border: 1px solid #fcc;
                    border-radius: 4px;
                    padding: 20px;
                    margin: 20px 0;
                }
                .error-title {
                    color: #c33;
                    margin: 0 0 10px 0;
                }
                .retry-btn {
                    background: #007cba;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    margin-top: 15px;
                }
            </style>
        </head>
        <body>
            <div class="error-box">
                <h2 class="error-title">Security Error: CSRF Protection</h2>
                <p>Your request could not be processed due to a security check failure.</p>
                <p>This usually happens when:</p>
                <ul>
                    <li>Your session has expired</li>
                    <li>You've been inactive for too long</li>
                    <li>There's a network connectivity issue</li>
                </ul>
                <p>Please try again by refreshing the page.</p>
                <button class="retry-btn" onclick="window.location.reload()">
                    Refresh Page
                </button>
            </div>
        </body>
        </html>
        """


class CSRFTokenGenerator:
    """Utility class for generating and managing CSRF tokens."""

    @staticmethod
    def generate_token() -> str:
        """Generate a cryptographically secure CSRF token."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def generate_token_pair() -> tuple[str, str]:
        """Generate a token pair for double-submit cookie pattern."""
        token = CSRFTokenGenerator.generate_token()
        return token, token

    @staticmethod
    def validate_token_pair(cookie_token: str, submitted_token: str) -> bool:
        """Validate token pair using constant-time comparison."""
        if not cookie_token or not submitted_token:
            return False
        return secrets.compare_digest(cookie_token, submitted_token)


class CSRFConfig:
    """Configuration class for CSRF protection."""

    def __init__(self):
        self.settings = get_settings()

    @property
    def cookie_name(self) -> str:
        """Get CSRF cookie name."""
        return getattr(self.settings.security, "csrf_cookie_name", "csrftoken")

    @property
    def header_name(self) -> str:
        """Get CSRF header name."""
        return getattr(self.settings.security, "csrf_header_name", "X-CSRFToken")

    @property
    def max_age(self) -> int:
        """Get CSRF cookie max age."""
        return getattr(self.settings.security, "csrf_max_age", 3600)

    @property
    def require_https(self) -> bool:
        """Check if HTTPS is required for CSRF cookies."""
        return self.settings.environment == "production"

    @property
    def exempt_paths(self) -> set[str]:
        """Get paths exempt from CSRF protection."""
        default_exempt = {
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/auth/login",
            "/api/auth/refresh",
        }

        custom_exempt = getattr(self.settings.security, "csrf_exempt_paths", set())

        return default_exempt.union(custom_exempt)


def get_csrf_middleware(app: ASGIApp) -> CSRFProtectionMiddleware:
    """Factory function to create configured CSRF middleware."""
    config = CSRFConfig()

    return CSRFProtectionMiddleware(
        app=app,
        cookie_name=config.cookie_name,
        header_name=config.header_name,
        exempt_paths=config.exempt_paths,
        require_https=config.require_https,
        max_age=config.max_age,
    )
