"""Security headers enforcement middleware.

This module provides comprehensive security headers to protect against:
- XSS attacks
- Clickjacking
- MIME type sniffing
- Information disclosure
- Mixed content attacks
"""

from __future__ import annotations

import logging
from enum import Enum

from fastapi import Request, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class CSPDirective(str, Enum):
    """Content Security Policy directives."""

    DEFAULT_SRC = "default-src"
    SCRIPT_SRC = "script-src"
    STYLE_SRC = "style-src"
    IMG_SRC = "img-src"
    CONNECT_SRC = "connect-src"
    FONT_SRC = "font-src"
    OBJECT_SRC = "object-src"
    MEDIA_SRC = "media-src"
    FRAME_SRC = "frame-src"
    FRAME_ANCESTORS = "frame-ancestors"
    CHILD_SRC = "child-src"
    WORKER_SRC = "worker-src"
    MANIFEST_SRC = "manifest-src"
    BASE_URI = "base-uri"
    FORM_ACTION = "form-action"
    UPGRADE_INSECURE_REQUESTS = "upgrade-insecure-requests"
    BLOCK_ALL_MIXED_CONTENT = "block-all-mixed-content"


class CSPSource(str, Enum):
    """Content Security Policy source values."""

    SELF = "'self'"
    NONE = "'none'"
    UNSAFE_INLINE = "'unsafe-inline'"
    UNSAFE_EVAL = "'unsafe-eval'"
    STRICT_DYNAMIC = "'strict-dynamic'"
    UNSAFE_HASHES = "'unsafe-hashes'"
    DATA = "data:"
    BLOB = "blob:"
    FILESYSTEM = "filesystem:"


class CSPConfig(BaseModel):
    """Configuration for Content Security Policy."""

    default_src: list[str] = Field(default_factory=lambda: [CSPSource.SELF])
    script_src: list[str] = Field(default_factory=lambda: [CSPSource.SELF])
    style_src: list[str] = Field(
        default_factory=lambda: [CSPSource.SELF, CSPSource.UNSAFE_INLINE]
    )
    img_src: list[str] = Field(default_factory=lambda: [CSPSource.SELF, CSPSource.DATA])
    connect_src: list[str] = Field(default_factory=lambda: [CSPSource.SELF])
    font_src: list[str] = Field(default_factory=lambda: [CSPSource.SELF])
    object_src: list[str] = Field(default_factory=lambda: [CSPSource.NONE])
    media_src: list[str] = Field(default_factory=lambda: [CSPSource.SELF])
    frame_src: list[str] = Field(default_factory=lambda: [CSPSource.NONE])
    frame_ancestors: list[str] = Field(default_factory=lambda: [CSPSource.NONE])
    base_uri: list[str] = Field(default_factory=lambda: [CSPSource.SELF])
    form_action: list[str] = Field(default_factory=lambda: [CSPSource.SELF])

    # Additional directives
    upgrade_insecure_requests: bool = True
    block_all_mixed_content: bool = True

    # Report URI for CSP violations
    report_uri: str | None = None
    report_to: str | None = None

    def to_header_value(self) -> str:
        """Convert CSP config to header value."""
        directives = []

        # Add source-based directives
        for directive, sources in [
            (CSPDirective.DEFAULT_SRC, self.default_src),
            (CSPDirective.SCRIPT_SRC, self.script_src),
            (CSPDirective.STYLE_SRC, self.style_src),
            (CSPDirective.IMG_SRC, self.img_src),
            (CSPDirective.CONNECT_SRC, self.connect_src),
            (CSPDirective.FONT_SRC, self.font_src),
            (CSPDirective.OBJECT_SRC, self.object_src),
            (CSPDirective.MEDIA_SRC, self.media_src),
            (CSPDirective.FRAME_SRC, self.frame_src),
            (CSPDirective.FRAME_ANCESTORS, self.frame_ancestors),
            (CSPDirective.BASE_URI, self.base_uri),
            (CSPDirective.FORM_ACTION, self.form_action),
        ]:
            if sources:
                directives.append(f"{directive} {' '.join(sources)}")

        # Add boolean directives
        if self.upgrade_insecure_requests:
            directives.append(CSPDirective.UPGRADE_INSECURE_REQUESTS)

        if self.block_all_mixed_content:
            directives.append(CSPDirective.BLOCK_ALL_MIXED_CONTENT)

        # Add reporting directives
        if self.report_uri:
            directives.append(f"report-uri {self.report_uri}")

        if self.report_to:
            directives.append(f"report-to {self.report_to}")

        return "; ".join(directives)


class SecurityHeaders(BaseModel):
    """Configuration for security headers."""

    # Content Security Policy
    csp_config: CSPConfig | None = Field(default_factory=CSPConfig)
    csp_report_only: bool = False

    # X-Frame-Options
    x_frame_options: str = "DENY"  # DENY, SAMEORIGIN, or ALLOW-FROM uri

    # X-Content-Type-Options
    x_content_type_options: str = "nosniff"

    # X-XSS-Protection (deprecated but still used)
    x_xss_protection: str = "1; mode=block"

    # Referrer Policy
    referrer_policy: str = "strict-origin-when-cross-origin"

    # Strict Transport Security
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False

    # Permissions Policy (formerly Feature Policy)
    permissions_policy: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "camera": ["none"],
            "microphone": ["none"],
            "geolocation": ["none"],
            "interest-cohort": ["none"],  # Disable FLoC
            "accelerometer": ["none"],
            "gyroscope": ["none"],
            "magnetometer": ["none"],
            "payment": ["none"],
            "usb": ["none"],
        }
    )

    # Cross-Origin Policies
    cross_origin_embedder_policy: str = "require-corp"
    cross_origin_opener_policy: str = "same-origin"
    cross_origin_resource_policy: str = "same-origin"

    # Cache Control for sensitive responses
    cache_control_sensitive: str = "no-cache, no-store, must-revalidate, private"

    # Custom security headers
    custom_headers: dict[str, str] = Field(default_factory=dict)

    # Headers to remove (security through obscurity)
    remove_headers: set[str] = Field(
        default_factory=lambda: {
            "Server",
            "X-Powered-By",
            "X-AspNet-Version",
            "X-AspNetMvc-Version",
        }
    )

    def get_hsts_header(self) -> str | None:
        """Get HSTS header value."""
        if not self.hsts_enabled:
            return None

        value = f"max-age={self.hsts_max_age}"

        if self.hsts_include_subdomains:
            value += "; includeSubDomains"

        if self.hsts_preload:
            value += "; preload"

        return value

    def get_permissions_policy_header(self) -> str | None:
        """Get Permissions Policy header value."""
        if not self.permissions_policy:
            return None

        policies = []
        for feature, allowlist in self.permissions_policy.items():
            if allowlist == ["none"]:
                policies.append(f"{feature}=()")
            elif allowlist == ["*"]:
                policies.append(f"{feature}=*")
            elif allowlist == ["self"]:
                policies.append(f"{feature}=self")
            else:
                # Specific origins
                origins = " ".join(f'"{origin}"' for origin in allowlist)
                policies.append(f"{feature}=({origins})")

        return ", ".join(policies)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers to responses."""

    def __init__(self, app, config: SecurityHeaders | None = None):
        """Initialize security headers middleware.

        Args:
            app: FastAPI application
            config: Security headers configuration
        """
        super().__init__(app)
        self.config = config or SecurityHeaders()
        logger.info("Security headers middleware initialized")

    async def dispatch(self, request: Request, call_next):
        """Process request and add security headers to response."""
        # Process the request
        response: Response = await call_next(request)

        # Add security headers
        self._add_security_headers(request, response)

        # Remove unwanted headers
        self._remove_headers(response)

        return response

    def _add_security_headers(self, request: Request, response: Response) -> None:
        """Add security headers to response."""

        # Content Security Policy
        if self.config.csp_config:
            csp_header = (
                "Content-Security-Policy-Report-Only"
                if self.config.csp_report_only
                else "Content-Security-Policy"
            )
            response.headers[csp_header] = self.config.csp_config.to_header_value()

        # X-Frame-Options
        if self.config.x_frame_options:
            response.headers["X-Frame-Options"] = self.config.x_frame_options

        # X-Content-Type-Options
        if self.config.x_content_type_options:
            response.headers["X-Content-Type-Options"] = (
                self.config.x_content_type_options
            )

        # X-XSS-Protection
        if self.config.x_xss_protection:
            response.headers["X-XSS-Protection"] = self.config.x_xss_protection

        # Referrer Policy
        if self.config.referrer_policy:
            response.headers["Referrer-Policy"] = self.config.referrer_policy

        # Strict Transport Security (only on HTTPS)
        if self._is_https(request) and self.config.hsts_enabled:
            hsts_value = self.config.get_hsts_header()
            if hsts_value:
                response.headers["Strict-Transport-Security"] = hsts_value

        # Permissions Policy
        permissions_policy = self.config.get_permissions_policy_header()
        if permissions_policy:
            response.headers["Permissions-Policy"] = permissions_policy

        # Cross-Origin Policies
        if self.config.cross_origin_embedder_policy:
            response.headers["Cross-Origin-Embedder-Policy"] = (
                self.config.cross_origin_embedder_policy
            )

        if self.config.cross_origin_opener_policy:
            response.headers["Cross-Origin-Opener-Policy"] = (
                self.config.cross_origin_opener_policy
            )

        if self.config.cross_origin_resource_policy:
            response.headers["Cross-Origin-Resource-Policy"] = (
                self.config.cross_origin_resource_policy
            )

        # Cache Control for sensitive endpoints
        if self._is_sensitive_endpoint(request):
            response.headers["Cache-Control"] = self.config.cache_control_sensitive
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        # Custom headers
        for header, value in self.config.custom_headers.items():
            response.headers[header] = value

        # Add security-related headers for debugging
        if request.url.path.startswith("/api/"):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"

    def _remove_headers(self, response: Response) -> None:
        """Remove unwanted headers from response."""
        for header in self.config.remove_headers:
            if header in response.headers:
                del response.headers[header]

    def _is_https(self, request: Request) -> bool:
        """Check if request is using HTTPS."""
        # Check scheme
        if request.url.scheme == "https":
            return True

        # Check X-Forwarded-Proto header (for load balancers)
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "").lower()
        if forwarded_proto == "https":
            return True

        # Check X-Forwarded-SSL header
        forwarded_ssl = request.headers.get("X-Forwarded-SSL", "").lower()
        if forwarded_ssl in ("on", "true", "1"):
            return True

        return False

    def _is_sensitive_endpoint(self, request: Request) -> bool:
        """Check if endpoint contains sensitive data."""
        sensitive_paths = [
            "/api/auth/",
            "/api/users/",
            "/api/admin/",
            "/health/detailed",
            "/metrics",
        ]

        path = request.url.path
        return any(
            path.startswith(sensitive_path) for sensitive_path in sensitive_paths
        )


def create_development_csp() -> CSPConfig:
    """Create a development-friendly CSP configuration."""
    return CSPConfig(
        default_src=[CSPSource.SELF],
        script_src=[
            CSPSource.SELF,
            CSPSource.UNSAFE_INLINE,
            CSPSource.UNSAFE_EVAL,
            "localhost:*",
        ],
        style_src=[CSPSource.SELF, CSPSource.UNSAFE_INLINE],
        img_src=[CSPSource.SELF, CSPSource.DATA, "localhost:*"],
        connect_src=[CSPSource.SELF, "localhost:*", "127.0.0.1:*"],
        font_src=[CSPSource.SELF, CSPSource.DATA],
        upgrade_insecure_requests=False,  # Allow HTTP in development
        block_all_mixed_content=False,
    )


def create_production_csp() -> CSPConfig:
    """Create a production-ready CSP configuration."""
    return CSPConfig(
        default_src=[CSPSource.SELF],
        script_src=[CSPSource.SELF],
        style_src=[CSPSource.SELF],
        img_src=[CSPSource.SELF, CSPSource.DATA],
        connect_src=[CSPSource.SELF],
        font_src=[CSPSource.SELF],
        object_src=[CSPSource.NONE],
        frame_ancestors=[CSPSource.NONE],
        upgrade_insecure_requests=True,
        block_all_mixed_content=True,
    )


def create_development_headers() -> SecurityHeaders:
    """Create development-friendly security headers."""
    return SecurityHeaders(
        csp_config=create_development_csp(),
        csp_report_only=True,  # Report-only mode in development
        hsts_enabled=False,  # Disable HSTS in development
        cross_origin_embedder_policy="unsafe-none",
        cross_origin_opener_policy="unsafe-none",
        cross_origin_resource_policy="cross-origin",
    )


def create_production_headers() -> SecurityHeaders:
    """Create production-ready security headers."""
    return SecurityHeaders(
        csp_config=create_production_csp(),
        csp_report_only=False,  # Enforce CSP in production
        hsts_enabled=True,
        hsts_max_age=31536000,  # 1 year
        hsts_include_subdomains=True,
        hsts_preload=True,
    )
