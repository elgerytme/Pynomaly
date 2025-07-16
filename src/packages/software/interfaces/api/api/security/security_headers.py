"""
Security headers and CORS policies for Pynomaly API.

This module provides:
- Security headers (HSTS, CSP, etc.)
- CORS policies
- Content security policies
- Security middleware
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SecurityHeaders:
    """Security headers management."""

    def __init__(self):
        self.default_headers = {
            # Strict Transport Security
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https://api.monorepo.com wss://api.monorepo.com; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            ),
            # X-Frame-Options
            "X-Frame-Options": "DENY",
            # X-Content-Type-Options
            "X-Content-Type-Options": "nosniff",
            # X-XSS-Protection
            "X-XSS-Protection": "1; mode=block",
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Permissions Policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "speaker=()"
            ),
            # Cross-Origin Embedder Policy
            "Cross-Origin-Embedder-Policy": "require-corp",
            # Cross-Origin Opener Policy
            "Cross-Origin-Opener-Policy": "same-origin",
            # Cross-Origin Resource Policy
            "Cross-Origin-Resource-Policy": "same-origin",
            # Cache Control
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            # Pragma
            "Pragma": "no-cache",
            # Expires
            "Expires": "0",
            # Server
            "Server": "Pynomaly/1.0",
        }

        self.custom_headers = {}
        self.endpoint_headers = {}

    def add_custom_header(self, name: str, value: str) -> None:
        """Add custom security header."""
        self.custom_headers[name] = value
        logger.info(f"Added custom header: {name}")

    def set_endpoint_headers(self, endpoint: str, headers: dict[str, str]) -> None:
        """Set specific headers for an endpoint."""
        self.endpoint_headers[endpoint] = headers
        logger.info(f"Set custom headers for endpoint: {endpoint}")

    def get_headers(self, endpoint: str = "") -> dict[str, str]:
        """Get security headers for response."""
        headers = self.default_headers.copy()
        headers.update(self.custom_headers)

        # Add endpoint-specific headers
        if endpoint in self.endpoint_headers:
            headers.update(self.endpoint_headers[endpoint])

        return headers

    def update_csp(self, directive: str, values: list[str]) -> None:
        """Update Content Security Policy directive."""
        current_csp = self.default_headers.get("Content-Security-Policy", "")

        # Parse current CSP
        policies = {}
        for policy in current_csp.split("; "):
            if " " in policy:
                key, value = policy.split(" ", 1)
                policies[key] = value

        # Update directive
        policies[directive] = " ".join(values)

        # Rebuild CSP
        new_csp = "; ".join(f"{key} {value}" for key, value in policies.items())
        self.default_headers["Content-Security-Policy"] = new_csp

        logger.info(f"Updated CSP directive {directive}: {values}")


class CORSPolicy:
    """CORS (Cross-Origin Resource Sharing) policy management."""

    def __init__(self):
        self.allowed_origins = set()
        self.allowed_methods = {"GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"}
        self.allowed_headers = {
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-CSRF-Token",
        }
        self.exposed_headers = {
            "Content-Length",
            "Content-Type",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        }
        self.max_age = 86400  # 24 hours
        self.allow_credentials = True

        # Default allowed origins (empty means no CORS)
        self.default_policy = "restrictive"  # restrictive, permissive, custom

    def add_allowed_origin(self, origin: str) -> None:
        """Add allowed origin."""
        self.allowed_origins.add(origin)
        logger.info(f"Added allowed origin: {origin}")

    def remove_allowed_origin(self, origin: str) -> None:
        """Remove allowed origin."""
        self.allowed_origins.discard(origin)
        logger.info(f"Removed allowed origin: {origin}")

    def set_policy(self, policy: str) -> None:
        """Set CORS policy type."""
        if policy == "restrictive":
            self.allowed_origins = {"https://monorepo.com", "https://app.monorepo.com"}
            self.allow_credentials = True
        elif policy == "permissive":
            self.allowed_origins = {"*"}
            self.allow_credentials = False

        self.default_policy = policy
        logger.info(f"Set CORS policy to: {policy}")

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if "*" in self.allowed_origins:
            return True

        return origin in self.allowed_origins

    def get_cors_headers(
        self, request_origin: str, request_method: str = "GET"
    ) -> dict[str, str]:
        """Get CORS headers for response."""
        headers = {}

        # Check if origin is allowed
        if self.is_origin_allowed(request_origin):
            headers["Access-Control-Allow-Origin"] = request_origin
        else:
            # Return restrictive headers for disallowed origins
            return headers

        # Add other CORS headers
        headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        headers["Access-Control-Expose-Headers"] = ", ".join(self.exposed_headers)
        headers["Access-Control-Max-Age"] = str(self.max_age)

        if self.allow_credentials and "*" not in self.allowed_origins:
            headers["Access-Control-Allow-Credentials"] = "true"

        return headers

    def handle_preflight(
        self, request_origin: str, request_method: str, request_headers: str
    ) -> dict[str, Any]:
        """Handle CORS preflight request."""
        result = {"allowed": False, "headers": {}, "status_code": 403}

        # Check origin
        if not self.is_origin_allowed(request_origin):
            return result

        # Check method
        if request_method not in self.allowed_methods:
            return result

        # Check headers
        if request_headers:
            requested_headers = {h.strip() for h in request_headers.split(",")}
            if not requested_headers.issubset(self.allowed_headers):
                return result

        # All checks passed
        result.update(
            {
                "allowed": True,
                "headers": self.get_cors_headers(request_origin, request_method),
                "status_code": 200,
            }
        )

        return result


class SecurityMiddleware:
    """Security middleware for request/response processing."""

    def __init__(self):
        self.security_headers = SecurityHeaders()
        self.cors_policy = CORSPolicy()
        self.content_type_validation = True
        self.request_size_limit = 100 * 1024 * 1024  # 100MB
        self.blocked_user_agents = {"curl", "wget", "python-requests", "scanner", "bot"}

    def process_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Process incoming request for security checks."""
        result = {"allowed": True, "errors": [], "warnings": []}

        # Check request size
        content_length = request_data.get("content_length", 0)
        if content_length > self.request_size_limit:
            result["allowed"] = False
            result["errors"].append(f"Request too large: {content_length} bytes")

        # Validate content type for POST/PUT requests
        method = request_data.get("method", "GET")
        if method in ["POST", "PUT", "PATCH"] and self.content_type_validation:
            content_type = request_data.get("content_type", "")
            if not self._is_valid_content_type(content_type):
                result["allowed"] = False
                result["errors"].append(f"Invalid content type: {content_type}")

        # Check user agent
        user_agent = request_data.get("user_agent", "").lower()
        if any(blocked in user_agent for blocked in self.blocked_user_agents):
            result["warnings"].append("Suspicious user agent detected")

        # Check for common attack patterns in URL
        url = request_data.get("url", "")
        if self._contains_attack_patterns(url):
            result["allowed"] = False
            result["errors"].append("Malicious URL pattern detected")

        return result

    def process_response(
        self, response_data: dict[str, Any], request_data: dict[str, Any]
    ) -> dict[str, str]:
        """Process outgoing response to add security headers."""
        endpoint = request_data.get("endpoint", "")
        origin = request_data.get("origin", "")
        method = request_data.get("method", "GET")

        # Get security headers
        headers = self.security_headers.get_headers(endpoint)

        # Add CORS headers if needed
        if origin:
            cors_headers = self.cors_policy.get_cors_headers(origin, method)
            headers.update(cors_headers)

        # Add response-specific headers
        if response_data.get("content_type"):
            headers["Content-Type"] = response_data["content_type"]

        # Remove server info for security
        if "Server" in headers and response_data.get("hide_server_info"):
            del headers["Server"]

        return headers

    def _is_valid_content_type(self, content_type: str) -> bool:
        """Validate content type."""
        valid_types = {
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
            "text/csv",
            "application/octet-stream",
        }

        # Extract main content type (ignore charset, boundary, etc.)
        main_type = content_type.split(";")[0].strip().lower()
        return main_type in valid_types

    def _contains_attack_patterns(self, url: str) -> bool:
        """Check for common attack patterns in URL."""
        attack_patterns = [
            "../",
            "..\\",
            "%2e%2e",
            "%252e",
            "etc/passwd",
            "boot.ini",
            "<script",
            "javascript:",
            "vbscript:",
            "onload=",
            "onerror=",
            "eval(",
            "alert(",
            "document.cookie",
            "document.write",
        ]

        url_lower = url.lower()
        return any(pattern in url_lower for pattern in attack_patterns)


class APISecurityConfig:
    """Centralized API security configuration."""

    def __init__(self):
        self.security_middleware = SecurityMiddleware()
        self.rate_limits = {}
        self.endpoint_permissions = {}
        self.security_policies = {}

    def configure_endpoint_security(
        self, endpoint: str, config: dict[str, Any]
    ) -> None:
        """Configure security for specific endpoint."""
        # Rate limiting
        if "rate_limit" in config:
            self.rate_limits[endpoint] = config["rate_limit"]

        # Permissions
        if "permissions" in config:
            self.endpoint_permissions[endpoint] = config["permissions"]

        # Custom headers
        if "headers" in config:
            self.security_middleware.security_headers.set_endpoint_headers(
                endpoint, config["headers"]
            )

        # CORS settings
        if "cors" in config:
            cors_config = config["cors"]
            if "origins" in cors_config:
                for origin in cors_config["origins"]:
                    self.security_middleware.cors_policy.add_allowed_origin(origin)

        logger.info(f"Configured security for endpoint: {endpoint}")

    def get_endpoint_config(self, endpoint: str) -> dict[str, Any]:
        """Get security configuration for endpoint."""
        return {
            "rate_limit": self.rate_limits.get(endpoint),
            "permissions": self.endpoint_permissions.get(endpoint),
            "headers": self.security_middleware.security_headers.endpoint_headers.get(
                endpoint, {}
            ),
            "cors_origins": list(self.security_middleware.cors_policy.allowed_origins),
        }

    def apply_security_policy(self, policy_name: str, endpoints: list[str]) -> None:
        """Apply security policy to multiple endpoints."""
        if policy_name == "high_security":
            policy = {
                "rate_limit": {"requests": 100, "window": 3600},
                "headers": {
                    "X-Frame-Options": "DENY",
                    "X-Content-Type-Options": "nosniff",
                    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
                },
                "cors": {"origins": ["https://app.monorepo.com"]},
            }
        elif policy_name == "api_endpoints":
            policy = {
                "rate_limit": {"requests": 1000, "window": 3600},
                "headers": {"Cache-Control": "no-cache, no-store, must-revalidate"},
                "cors": {"origins": ["*"]},
            }
        else:
            logger.warning(f"Unknown security policy: {policy_name}")
            return

        for endpoint in endpoints:
            self.configure_endpoint_security(endpoint, policy)

        logger.info(f"Applied {policy_name} policy to {len(endpoints)} endpoints")


def create_security_config() -> APISecurityConfig:
    """Create default security configuration."""
    config = APISecurityConfig()

    # Configure high-security endpoints
    admin_endpoints = ["/admin/*", "/config/*", "/users/*"]
    config.apply_security_policy("high_security", admin_endpoints)

    # Configure API endpoints
    api_endpoints = ["/api/v1/*", "/api/v2/*"]
    config.apply_security_policy("api_endpoints", api_endpoints)

    # Set restrictive CORS policy
    config.security_middleware.cors_policy.set_policy("restrictive")

    return config
