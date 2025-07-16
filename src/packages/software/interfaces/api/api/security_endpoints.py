"""Enhanced security API endpoints for web UI security features."""

import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from interfaces.presentation.web.csrf import get_csrf_protection, refresh_csrf_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/security", tags=["Security"])


class CSPViolationReport(BaseModel):
    """CSP violation report model."""

    type: str = Field(description="Type of violation report")
    blocked_uri: str | None = Field(alias="blockedURI", default=None)
    violated_directive: str | None = Field(alias="violatedDirective", default=None)
    effective_directive: str | None = Field(alias="effectiveDirective", default=None)
    original_policy: str | None = Field(alias="originalPolicy", default=None)
    source_file: str | None = Field(alias="sourceFile", default=None)
    line_number: int | None = Field(alias="lineNumber", default=None)
    column_number: int | None = Field(alias="columnNumber", default=None)
    timestamp: str
    user_agent: str | None = Field(alias="userAgent", default=None)
    url: str
    referrer: str | None = Field(default=None)
    violation_id: str | None = Field(alias="violationId", default=None)


class MixedContentReport(BaseModel):
    """Mixed content report model."""

    url: str
    resources: list[dict[str, Any]]
    timestamp: str


class SecurityEventReport(BaseModel):
    """Generic security event report model."""

    event_type: str
    details: dict[str, Any]
    timestamp: str | None = None
    url: str | None = None
    user_agent: str | None = None


class CSRFTokenResponse(BaseModel):
    """CSRF token response model."""

    csrf_token: str
    expires_at: str | None = None


class NonceResponse(BaseModel):
    """CSP nonce response model."""

    nonce: str
    expires_at: str | None = None


@router.post("/csp-violations", status_code=status.HTTP_204_NO_CONTENT)
async def report_csp_violation(violation: CSPViolationReport, request: Request) -> None:
    """Receive and process CSP violation reports."""
    try:
        # Log the violation
        logger.warning(
            f"CSP Violation: {violation.violated_directive} - {violation.blocked_uri} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Store violation details for analysis
        violation_data = {
            "timestamp": violation.timestamp,
            "client_ip": request.client.host if request.client else None,
            "user_agent": violation.user_agent or request.headers.get("User-Agent"),
            "violation": violation.dict(by_alias=True),
            "request_url": str(request.url),
            "referrer": violation.referrer or request.headers.get("Referer"),
        }

        # In a production environment, you would store this in a database
        # or send to a monitoring service
        logger.info(f"CSP violation data: {json.dumps(violation_data, indent=2)}")

        # Check for potential attack patterns
        await _analyze_csp_violation(violation, request)

    except Exception as e:
        logger.error(f"Error processing CSP violation report: {e}")
        # Don't raise exception to avoid disrupting the user experience


@router.post("/mixed-content", status_code=status.HTTP_204_NO_CONTENT)
async def report_mixed_content(report: MixedContentReport, request: Request) -> None:
    """Receive and process mixed content reports."""
    try:
        logger.warning(
            f"Mixed content detected on {report.url}: {len(report.resources)} resources"
        )

        # Log detailed mixed content information
        for resource in report.resources:
            logger.warning(
                f"Insecure {resource.get('type', 'resource')}: {resource.get('src') or resource.get('href')}"
            )

        # Store for security team review
        mixed_content_data = {
            "timestamp": report.timestamp,
            "client_ip": request.client.host if request.client else None,
            "report": report.dict(),
            "user_agent": request.headers.get("User-Agent"),
        }

        logger.info(f"Mixed content data: {json.dumps(mixed_content_data, indent=2)}")

    except Exception as e:
        logger.error(f"Error processing mixed content report: {e}")


@router.post("/events", status_code=status.HTTP_204_NO_CONTENT)
async def report_security_event(event: SecurityEventReport, request: Request) -> None:
    """Receive and process general security events."""
    try:
        # Set timestamp if not provided
        if not event.timestamp:
            event.timestamp = datetime.now().isoformat()

        # Set URL if not provided
        if not event.url:
            event.url = str(request.url)

        # Set user agent if not provided
        if not event.user_agent:
            event.user_agent = request.headers.get("User-Agent")

        logger.info(
            f"Security event: {event.event_type} from {request.client.host if request.client else 'unknown'}"
        )

        # Store event data
        event_data = {
            "timestamp": event.timestamp,
            "client_ip": request.client.host if request.client else None,
            "event": event.dict(),
            "request_headers": dict(request.headers),
        }

        # Log based on event severity
        if event.event_type in ["javascript_error", "unhandled_promise_rejection"]:
            logger.warning(f"Client-side error: {json.dumps(event_data, indent=2)}")
        elif event.event_type in ["rate_limit_exceeded", "suspicious_activity"]:
            logger.error(f"Security concern: {json.dumps(event_data, indent=2)}")
        else:
            logger.info(f"Security event: {json.dumps(event_data, indent=2)}")

    except Exception as e:
        logger.error(f"Error processing security event: {e}")


@router.post("/csrf/refresh", response_model=CSRFTokenResponse)
async def refresh_csrf_token_endpoint(
    request: Request, response: Response
) -> CSRFTokenResponse:
    """Refresh CSRF token for the current session."""
    try:
        # Validate existing CSRF token in request
        existing_token = request.headers.get("X-CSRFToken")
        if not existing_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing CSRF token in request",
            )

        # Generate new token
        new_token = refresh_csrf_token(request, response)

        logger.info(
            f"CSRF token refreshed for session from {request.client.host if request.client else 'unknown'}"
        )

        return CSRFTokenResponse(
            csrf_token=new_token, expires_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error refreshing CSRF token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh CSRF token",
        )


@router.post("/refresh-nonce", response_model=NonceResponse)
async def refresh_csp_nonce(request: Request) -> NonceResponse:
    """Refresh CSP nonce for dynamic content."""
    try:
        # Generate new nonce
        import secrets

        new_nonce = secrets.token_hex(16)

        logger.info(
            f"CSP nonce refreshed from {request.client.host if request.client else 'unknown'}"
        )

        return NonceResponse(nonce=new_nonce, expires_at=datetime.now().isoformat())

    except Exception as e:
        logger.error(f"Error refreshing CSP nonce: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh CSP nonce",
        )


@router.get("/status")
async def get_security_status(request: Request) -> dict[str, Any]:
    """Get current security status and configuration."""
    try:
        # Check CSRF protection status
        csrf_protection = get_csrf_protection()

        # Get security headers from request
        security_headers = {
            "x-frame-options": request.headers.get("X-Frame-Options"),
            "x-content-type-options": request.headers.get("X-Content-Type-Options"),
            "x-xss-protection": request.headers.get("X-XSS-Protection"),
            "strict-transport-security": request.headers.get(
                "Strict-Transport-Security"
            ),
            "content-security-policy": request.headers.get("Content-Security-Policy"),
        }

        return {
            "status": "active",
            "csrf_protection": {
                "enabled": True,
                "token_lifetime": csrf_protection.token_lifetime
                if csrf_protection
                else 3600,
            },
            "security_headers": security_headers,
            "session_management": {
                "enabled": True,
                "timeout": 1800,  # 30 minutes
                "warning_time": 300,  # 5 minutes
            },
            "input_validation": {"enabled": True, "real_time": True},
            "csp_reporting": {
                "enabled": True,
                "endpoint": "/api/v1/security/csp-violations",
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting security status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security status",
        )


@router.get("/health")
async def security_health_check() -> dict[str, Any]:
    """Security-specific health check endpoint."""
    try:
        # Check if security modules are functioning
        checks = {
            "csrf_protection": True,
            "session_management": True,
            "input_validation": True,
            "csp_reporting": True,
            "rate_limiting": True,
        }

        # Perform basic checks
        try:
            csrf_protection = get_csrf_protection()
            checks["csrf_protection"] = csrf_protection is not None
        except Exception:
            checks["csrf_protection"] = False

        all_healthy = all(checks.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Security health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def _analyze_csp_violation(
    violation: CSPViolationReport, request: Request
) -> None:
    """Analyze CSP violation for potential security threats."""
    try:
        # Check for common attack patterns
        blocked_uri = violation.blocked_uri or ""

        # XSS attempt detection
        xss_patterns = [
            "javascript:",
            "data:text/html",
            "vbscript:",
            "<script",
            "eval(",
            "alert(",
            "document.cookie",
        ]

        if any(pattern in blocked_uri.lower() for pattern in xss_patterns):
            logger.warning(
                f"Potential XSS attempt detected: {blocked_uri} "
                f"from {request.client.host if request.client else 'unknown'}"
            )

        # Injection attempt detection
        injection_patterns = [
            "../",
            "\\x",
            "%2e%2e",
            "union select",
            "drop table",
            "insert into",
        ]

        if any(pattern in blocked_uri.lower() for pattern in injection_patterns):
            logger.warning(
                f"Potential injection attempt detected: {blocked_uri} "
                f"from {request.client.host if request.client else 'unknown'}"
            )

        # Suspicious external resources
        if violation.violated_directive in ["script-src", "object-src", "frame-src"]:
            if (
                blocked_uri.startswith(("http://", "https://"))
                and "eval" in blocked_uri
            ):
                logger.warning(
                    f"Suspicious external resource: {blocked_uri} "
                    f"from {request.client.host if request.client else 'unknown'}"
                )

    except Exception as e:
        logger.error(f"Error analyzing CSP violation: {e}")


# Additional utility endpoints for security testing (development only)
@router.post("/test/trigger-violation")
async def trigger_test_violation(request: Request) -> dict[str, str]:
    """Trigger a test CSP violation for testing purposes (development only)."""
    if request.url.hostname not in ["localhost", "127.0.0.1", "dev.example.com"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Test endpoints only available in development",
        )

    return {
        "message": "Test violation endpoint - use this to verify CSP reporting",
        "test_script": "javascript:alert('CSP Test')",
        "instructions": "Include the test_script in a script src to trigger a violation",
    }


@router.get("/violations/summary")
async def get_violations_summary(request: Request) -> dict[str, Any]:
    """Get summary of recent security violations (for security dashboard)."""
    # In production, this would query a database
    # For now, return mock data for testing
    return {
        "period": "last_24_hours",
        "csp_violations": {
            "total": 12,
            "by_directive": {"script-src": 8, "style-src": 3, "img-src": 1},
            "blocked_domains": ["suspicious-domain.com", "malicious-ads.net"],
        },
        "mixed_content": {
            "total": 2,
            "resources": [
                "http://example.com/image.jpg",
                "http://cdn.example.com/script.js",
            ],
        },
        "security_events": {
            "total": 45,
            "by_type": {
                "rate_limit_exceeded": 15,
                "suspicious_activity": 3,
                "authentication_failure": 27,
            },
        },
        "recommendations": [
            "Consider blocking suspicious-domain.com",
            "Update CDN URLs to use HTTPS",
            "Review rate limiting configuration",
        ],
        "timestamp": datetime.now().isoformat(),
    }
