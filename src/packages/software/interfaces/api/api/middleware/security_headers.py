"""Security headers middleware for enhanced web security."""

from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all responses.

    This middleware adds essential security headers to protect against
    common web vulnerabilities like XSS, clickjacking, and CSRF attacks.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to the response."""
        response = await call_next(request)

        # Content Security Policy (CSP)
        # This is a comprehensive CSP that allows for the current application structure
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com https://cdn.tailwindcss.com https://d3js.org",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.tailwindcss.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self' ws: wss:",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "upgrade-insecure-requests",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # X-Frame-Options - Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options - Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection - Enable XSS filtering
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy - Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Strict-Transport-Security - Enforce HTTPS (only if using HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        # X-Permitted-Cross-Domain-Policies - Restrict cross-domain policies
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

        # Feature-Policy / Permissions-Policy - Control browser features
        permissions_policy = [
            "accelerometer=()",
            "camera=()",
            "geolocation=()",
            "gyroscope=()",
            "magnetometer=()",
            "microphone=()",
            "payment=()",
            "usb=()",
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions_policy)

        # Cache-Control for sensitive pages
        if request.url.path in ["/login", "/admin", "/users"]:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        # Remove server information
        if "Server" in response.headers:
            del response.headers["Server"]

        return response


class CSPViolationReporter:
    """
    CSP violation reporter for monitoring Content Security Policy violations.
    """

    @staticmethod
    def report_violation(request: Request, violation_data: dict):
        """Report CSP violation."""
        # In production, this would log to a monitoring service
        print(f"CSP Violation: {violation_data}")
        print(f"Request URL: {request.url}")
        print(f"User Agent: {request.headers.get('User-Agent', 'Unknown')}")

        # You could integrate with:
        # - Sentry
        # - LogRocket
        # - DataDog
        # - Custom logging service
