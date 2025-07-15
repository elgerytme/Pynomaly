"""CSRF token management for web UI."""

import secrets

from fastapi import Request


def generate_csrf_token() -> str:
    """Generate a secure CSRF token."""
    return secrets.token_urlsafe(32)


def get_csrf_token(request: Request) -> str:
    """Get or generate CSRF token for the current session."""
    # Check if we already have a token in the session
    if hasattr(request, "session") and request.session:
        csrf_token = request.session.get("csrf_token")
        if csrf_token:
            return csrf_token

    # Generate a new token
    csrf_token = generate_csrf_token()

    # Store in session if available
    if hasattr(request, "session") and request.session:
        request.session["csrf_token"] = csrf_token

    return csrf_token


def validate_csrf_token(request: Request, token: str) -> bool:
    """Validate a CSRF token against the session."""
    if not token:
        return False

    # Check session token
    if hasattr(request, "session") and request.session:
        session_token = request.session.get("csrf_token")
        if session_token and session_token == token:
            return True

    # Check header token
    header_token = request.headers.get("X-CSRFToken")
    if header_token and header_token == token:
        return True

    return False


def add_csrf_to_context(request: Request, context: dict) -> dict:
    """Add CSRF token to template context."""
    context["csrf_token"] = get_csrf_token(request)
    return context
