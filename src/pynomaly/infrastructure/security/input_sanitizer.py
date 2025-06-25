"""Advanced input sanitization and validation framework.

This module provides comprehensive input sanitization to prevent:
- XSS attacks
- Script injection
- Command injection
- Path traversal
- HTML/XML injection
- NoSQL injection
"""

from __future__ import annotations

import html
import logging
import re
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import Any

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SanitizationLevel(str, Enum):
    """Security sanitization levels."""

    STRICT = "strict"  # Maximum security, may break functionality
    MODERATE = "moderate"  # Balanced security and usability
    PERMISSIVE = "permissive"  # Minimal sanitization


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(
        self, message: str, field: str | None = None, value: str | None = None
    ):
        self.field = field
        self.value = value
        super().__init__(message)


class SanitizationConfig(BaseModel):
    """Configuration for input sanitization."""

    level: SanitizationLevel = SanitizationLevel.MODERATE
    max_length: int = Field(default=10000, gt=0)
    allow_html: bool = False
    allow_urls: bool = True
    allow_email: bool = True
    allow_special_chars: bool = True
    custom_patterns: list[str] = Field(default_factory=list)
    blocked_patterns: list[str] = Field(default_factory=list)

    @validator("custom_patterns", "blocked_patterns")
    def validate_patterns(cls, v):
        """Validate regex patterns."""
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        return v


class InputSanitizer:
    """Advanced input sanitization service."""

    def __init__(self, config: SanitizationConfig | None = None):
        """Initialize input sanitizer.

        Args:
            config: Sanitization configuration
        """
        self.config = config or SanitizationConfig()

        # Compile patterns for performance
        self._script_pattern = re.compile(
            r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL
        )
        self._html_pattern = re.compile(r"<[^>]+>")
        self._sql_keywords = re.compile(
            r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|SCRIPT)\b",
            re.IGNORECASE,
        )
        self._path_traversal = re.compile(r"\.\.[\\/]")
        self._command_injection = re.compile(
            r"[;&|`$(){}[\]<>*?!]|(\|\||&&)", re.IGNORECASE
        )

        # Compile custom patterns
        self._custom_patterns: list[Pattern] = []
        for pattern in self.config.custom_patterns:
            self._custom_patterns.append(re.compile(pattern, re.IGNORECASE))

        self._blocked_patterns: list[Pattern] = []
        for pattern in self.config.blocked_patterns:
            self._blocked_patterns.append(re.compile(pattern, re.IGNORECASE))

    def sanitize_string(self, value: str, field_name: str | None = None) -> str:
        """Sanitize a string input.

        Args:
            value: String to sanitize
            field_name: Optional field name for error reporting

        Returns:
            Sanitized string

        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(value, str):
            raise ValidationError(
                f"Expected string, got {type(value)}", field_name, str(value)
            )

        # Check length
        if len(value) > self.config.max_length:
            raise ValidationError(
                f"Input too long: {len(value)} > {self.config.max_length}",
                field_name,
                value[:100] + "...",
            )

        # Apply sanitization based on level
        sanitized = value

        if self.config.level == SanitizationLevel.STRICT:
            sanitized = self._strict_sanitize(sanitized)
        elif self.config.level == SanitizationLevel.MODERATE:
            sanitized = self._moderate_sanitize(sanitized)
        else:  # PERMISSIVE
            sanitized = self._permissive_sanitize(sanitized)

        # Check blocked patterns
        for pattern in self._blocked_patterns:
            if pattern.search(sanitized):
                raise ValidationError(
                    f"Input contains blocked pattern: {pattern.pattern}",
                    field_name,
                    value,
                )

        # Apply custom validation patterns
        for pattern in self._custom_patterns:
            if not pattern.search(sanitized):
                raise ValidationError(
                    f"Input doesn't match required pattern: {pattern.pattern}",
                    field_name,
                    value,
                )

        return sanitized

    def _strict_sanitize(self, value: str) -> str:
        """Apply strict sanitization."""
        # Remove all HTML
        value = self._html_pattern.sub("", value)

        # Remove script tags
        value = self._script_pattern.sub("", value)

        # Escape remaining special characters
        value = html.escape(value, quote=True)

        # Remove potential command injection chars
        if self._command_injection.search(value):
            # Replace with safe alternatives
            value = re.sub(r"[;&|`$(){}[\]<>*?!]", "", value)
            value = re.sub(r"(\|\||&&)", "", value)

        # Remove path traversal
        value = self._path_traversal.sub("", value)

        return value.strip()

    def _moderate_sanitize(self, value: str) -> str:
        """Apply moderate sanitization."""
        # Remove script tags
        value = self._script_pattern.sub("", value)

        # Remove path traversal
        value = self._path_traversal.sub("", value)

        # HTML escape if not allowed
        if not self.config.allow_html:
            value = html.escape(value, quote=False)

        # Check for SQL injection keywords in non-SQL contexts
        if self._sql_keywords.search(value) and not self._is_sql_context(value):
            logger.warning(f"Potential SQL injection attempt: {value[:100]}")

        return value.strip()

    def _permissive_sanitize(self, value: str) -> str:
        """Apply minimal sanitization."""
        # Only remove obvious script tags
        value = self._script_pattern.sub("", value)

        # Remove obvious path traversal
        value = re.sub(r"\.\.[\\/]{2,}", "", value)

        return value.strip()

    def _is_sql_context(self, value: str) -> bool:
        """Check if value appears to be legitimate SQL."""
        # Simple heuristic - could be enhanced
        return len(self._sql_keywords.findall(value)) > 2

    def sanitize_dict(
        self, data: dict[str, Any], recursive: bool = True
    ) -> dict[str, Any]:
        """Sanitize all string values in a dictionary.

        Args:
            data: Dictionary to sanitize
            recursive: Whether to recursively sanitize nested dicts

        Returns:
            Sanitized dictionary
        """
        sanitized = {}

        for key, value in data.items():
            # Sanitize the key itself
            clean_key = self.sanitize_string(key, "dict_key")

            if isinstance(value, str):
                sanitized[clean_key] = self.sanitize_string(value, key)
            elif isinstance(value, dict) and recursive:
                sanitized[clean_key] = self.sanitize_dict(value, recursive)
            elif isinstance(value, list):
                sanitized[clean_key] = self._sanitize_list(value, recursive)
            else:
                sanitized[clean_key] = value

        return sanitized

    def _sanitize_list(self, data: list[Any], recursive: bool = True) -> list[Any]:
        """Sanitize string values in a list."""
        sanitized = []

        for i, item in enumerate(data):
            if isinstance(item, str):
                sanitized.append(self.sanitize_string(item, f"list[{i}]"))
            elif isinstance(item, dict) and recursive:
                sanitized.append(self.sanitize_dict(item, recursive))
            elif isinstance(item, list) and recursive:
                sanitized.append(self._sanitize_list(item, recursive))
            else:
                sanitized.append(item)

        return sanitized

    def validate_file_path(self, path: str) -> str:
        """Validate and sanitize file path.

        Args:
            path: File path to validate

        Returns:
            Sanitized path

        Raises:
            ValidationError: If path is unsafe
        """
        # Basic sanitization
        path = self.sanitize_string(path, "file_path")

        try:
            resolved_path = Path(path).resolve()
        except (OSError, ValueError) as e:
            raise ValidationError(f"Invalid file path: {e}", "file_path", path)

        # Check for path traversal
        if ".." in path or self._path_traversal.search(path):
            raise ValidationError("Path traversal detected", "file_path", path)

        # Ensure path doesn't contain null bytes
        if "\x00" in path:
            raise ValidationError("Null byte in path", "file_path", path)

        return str(resolved_path)

    def validate_url(self, url: str) -> str:
        """Validate and sanitize URL.

        Args:
            url: URL to validate

        Returns:
            Sanitized URL

        Raises:
            ValidationError: If URL is unsafe
        """
        if not self.config.allow_urls:
            raise ValidationError("URLs not allowed", "url", url)

        # Basic sanitization
        url = self.sanitize_string(url, "url")

        # Simple URL validation
        url_pattern = re.compile(
            r"^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w._~!$&\'()*+,;=:@-]|%[0-9a-fA-F]{2})*)*(?:\?(?:[\w._~!$&\'()*+,;=:@-]|%[0-9a-fA-F]{2})*)?(?:#(?:[\w._~!$&\'()*+,;=:@-]|%[0-9a-fA-F]{2})*)?$"
        )

        if not url_pattern.match(url):
            raise ValidationError("Invalid URL format", "url", url)

        # Check for suspicious schemes
        if url.lower().startswith(("javascript:", "data:", "vbscript:", "file:")):
            raise ValidationError("Unsafe URL scheme", "url", url)

        return url

    def validate_email(self, email: str) -> str:
        """Validate and sanitize email address.

        Args:
            email: Email to validate

        Returns:
            Sanitized email

        Raises:
            ValidationError: If email is invalid
        """
        if not self.config.allow_email:
            raise ValidationError("Email addresses not allowed", "email", email)

        # Basic sanitization
        email = self.sanitize_string(email, "email")

        # Simple email validation
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

        if not email_pattern.match(email):
            raise ValidationError("Invalid email format", "email", email)

        return email.lower()


# Global sanitizer instance
_sanitizer: InputSanitizer | None = None


def get_sanitizer() -> InputSanitizer:
    """Get global sanitizer instance."""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = InputSanitizer()
    return _sanitizer


def sanitize_input(
    value: str | dict[str, Any] | list[Any],
    config: SanitizationConfig | None = None,
) -> str | dict[str, Any] | list[Any]:
    """Convenience function to sanitize input.

    Args:
        value: Value to sanitize
        config: Optional sanitization config

    Returns:
        Sanitized value
    """
    sanitizer = InputSanitizer(config) if config else get_sanitizer()

    if isinstance(value, str):
        return sanitizer.sanitize_string(value)
    elif isinstance(value, dict):
        return sanitizer.sanitize_dict(value)
    elif isinstance(value, list):
        return sanitizer._sanitize_list(value)
    else:
        return value


def validate_sql_safe(value: str) -> bool:
    """Quick check if string is SQL injection safe.

    Args:
        value: String to check

    Returns:
        True if appears safe
    """
    sanitizer = get_sanitizer()

    # Check for SQL keywords
    if sanitizer._sql_keywords.search(value):
        return False

    # Check for SQL injection patterns
    sql_injection_patterns = [
        r"'\s*(?:OR|AND)\s*'",  # ' OR ', ' AND '
        r"'\s*(?:OR|AND)\s*\d+\s*=\s*\d+",  # ' OR 1=1
        r"--",  # SQL comments
        r"/\*.*?\*/",  # SQL block comments
        r";\s*(?:DROP|DELETE|INSERT|UPDATE|CREATE|ALTER)",  # Statement termination
    ]

    for pattern in sql_injection_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            return False

    return True
