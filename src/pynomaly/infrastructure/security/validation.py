"""Input validation and sanitization utilities for API security."""

import html
import re
import urllib.parse
from typing import Any

import bleach
from fastapi import HTTPException, status
from pydantic import BaseModel, validator


class SecurityConfig:
    """Security configuration for validation."""

    # Maximum lengths
    MAX_STRING_LENGTH = 1000
    MAX_TEXT_LENGTH = 10000
    MAX_USERNAME_LENGTH = 50
    MAX_EMAIL_LENGTH = 254
    MAX_FILENAME_LENGTH = 255

    # Allowed HTML tags for rich text (very restrictive)
    ALLOWED_HTML_TAGS = ['b', 'i', 'u', 'em', 'strong', 'p', 'br']
    ALLOWED_HTML_ATTRS = {}

    # Regex patterns
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')

    # Dangerous patterns to detect
    SQL_INJECTION_PATTERNS = [
        r"'\s*(or|and)\s*'?\w",
        r"'\s*(or|and)\s*'?\d",
        r"'\s*-{2,}",
        r"(union|select|insert|update|delete|drop|create|alter)\s",
        r"'\s*(;|\||&)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
    ]


class ValidationError(Exception):
    """Custom validation error."""
    pass


class InputSanitizer:
    """Input sanitization utilities."""

    @staticmethod
    def sanitize_string(value: str, max_length: int | None = None) -> str:
        """Sanitize a general string input.

        Args:
            value: Input string
            max_length: Maximum allowed length

        Returns:
            Sanitized string

        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(value, str):
            raise ValidationError("Input must be a string")

        # Remove null bytes
        value = value.replace('\x00', '')

        # Normalize whitespace
        value = ' '.join(value.split())

        # HTML escape
        value = html.escape(value)

        # Check length
        max_len = max_length or SecurityConfig.MAX_STRING_LENGTH
        if len(value) > max_len:
            raise ValidationError(f"String too long (max {max_len} characters)")

        return value

    @staticmethod
    def sanitize_html(value: str, allowed_tags: list[str] | None = None) -> str:
        """Sanitize HTML content.

        Args:
            value: HTML string
            allowed_tags: List of allowed HTML tags

        Returns:
            Sanitized HTML
        """
        if not isinstance(value, str):
            raise ValidationError("Input must be a string")

        tags = allowed_tags or SecurityConfig.ALLOWED_HTML_TAGS

        # Use bleach to sanitize HTML
        sanitized = bleach.clean(
            value,
            tags=tags,
            attributes=SecurityConfig.ALLOWED_HTML_ATTRS,
            strip=True
        )

        # Check length after sanitization
        if len(sanitized) > SecurityConfig.MAX_TEXT_LENGTH:
            raise ValidationError(f"Text too long (max {SecurityConfig.MAX_TEXT_LENGTH} characters)")

        return sanitized

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename

        Raises:
            ValidationError: If filename is invalid
        """
        if not isinstance(filename, str):
            raise ValidationError("Filename must be a string")

        # Remove path components
        filename = filename.split('/')[-1]
        filename = filename.split('\\')[-1]

        # Remove dangerous characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

        # Ensure it's not empty or just dots
        if not filename or filename.strip('.') == '':
            raise ValidationError("Invalid filename")

        # Check length
        if len(filename) > SecurityConfig.MAX_FILENAME_LENGTH:
            raise ValidationError(f"Filename too long (max {SecurityConfig.MAX_FILENAME_LENGTH} characters)")

        return filename

    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL input.

        Args:
            url: URL string

        Returns:
            Sanitized URL

        Raises:
            ValidationError: If URL is invalid
        """
        if not isinstance(url, str):
            raise ValidationError("URL must be a string")

        # Parse and validate URL
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            raise ValidationError("Invalid URL format")

        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            raise ValidationError("Only HTTP and HTTPS URLs are allowed")

        # Reconstruct URL to normalize it
        sanitized = urllib.parse.urlunparse(parsed)

        return sanitized


class ThreatDetector:
    """Detect various security threats in input."""

    @staticmethod
    def detect_sql_injection(value: str) -> bool:
        """Detect potential SQL injection attempts.

        Args:
            value: Input string to check

        Returns:
            True if potential SQL injection detected
        """
        value_lower = value.lower()

        for pattern in SecurityConfig.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def detect_xss(value: str) -> bool:
        """Detect potential XSS attempts.

        Args:
            value: Input string to check

        Returns:
            True if potential XSS detected
        """
        value_lower = value.lower()

        for pattern in SecurityConfig.XSS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def detect_path_traversal(value: str) -> bool:
        """Detect potential path traversal attempts.

        Args:
            value: Input string to check

        Returns:
            True if potential path traversal detected
        """
        value_lower = value.lower()

        for pattern in SecurityConfig.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def detect_command_injection(value: str) -> bool:
        """Detect potential command injection attempts.

        Args:
            value: Input string to check

        Returns:
            True if potential command injection detected
        """
        dangerous_chars = ['|', '&', ';', '$', '`', '>', '<', '\n', '\r']

        return any(char in value for char in dangerous_chars)


class InputValidator:
    """High-level input validation with threat detection."""

    def __init__(self, strict_mode: bool = True):
        """Initialize validator.

        Args:
            strict_mode: Enable strict validation (reject on threat detection)
        """
        self.strict_mode = strict_mode
        self.sanitizer = InputSanitizer()
        self.threat_detector = ThreatDetector()

    def validate_and_sanitize(self,
                             value: Any,
                             field_type: str,
                             allow_html: bool = False,
                             max_length: int | None = None) -> Any:
        """Validate and sanitize input value.

        Args:
            value: Input value
            field_type: Type of field (string, email, username, filename, url, text)
            allow_html: Whether to allow HTML content
            max_length: Maximum length override

        Returns:
            Validated and sanitized value

        Raises:
            ValidationError: If validation fails
            HTTPException: If threats detected in strict mode
        """
        if value is None:
            return None

        if not isinstance(value, str):
            if field_type in ['string', 'text', 'email', 'username', 'filename', 'url']:
                raise ValidationError(f"Expected string for {field_type}, got {type(value)}")
            return value

        # Threat detection
        threats = []
        if self.threat_detector.detect_sql_injection(value):
            threats.append('SQL injection')
        if self.threat_detector.detect_xss(value):
            threats.append('XSS')
        if self.threat_detector.detect_path_traversal(value):
            threats.append('Path traversal')
        if self.threat_detector.detect_command_injection(value):
            threats.append('Command injection')

        if threats and self.strict_mode:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Security threat detected: {', '.join(threats)}"
            )

        # Field-specific validation and sanitization
        try:
            if field_type == 'email':
                return self._validate_email(value)
            elif field_type == 'username':
                return self._validate_username(value)
            elif field_type == 'filename':
                return self.sanitizer.sanitize_filename(value)
            elif field_type == 'url':
                return self.sanitizer.sanitize_url(value)
            elif field_type == 'text' and allow_html:
                return self.sanitizer.sanitize_html(value)
            else:
                return self.sanitizer.sanitize_string(value, max_length)

        except ValidationError as e:
            if self.strict_mode:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
            raise

    def _validate_email(self, email: str) -> str:
        """Validate email address.

        Args:
            email: Email string

        Returns:
            Validated email

        Raises:
            ValidationError: If email is invalid
        """
        email = email.strip().lower()

        if len(email) > SecurityConfig.MAX_EMAIL_LENGTH:
            raise ValidationError(f"Email too long (max {SecurityConfig.MAX_EMAIL_LENGTH} characters)")

        if not SecurityConfig.EMAIL_PATTERN.match(email):
            raise ValidationError("Invalid email format")

        return email

    def _validate_username(self, username: str) -> str:
        """Validate username.

        Args:
            username: Username string

        Returns:
            Validated username

        Raises:
            ValidationError: If username is invalid
        """
        username = username.strip()

        if len(username) > SecurityConfig.MAX_USERNAME_LENGTH:
            raise ValidationError(f"Username too long (max {SecurityConfig.MAX_USERNAME_LENGTH} characters)")

        if len(username) < 3:
            raise ValidationError("Username must be at least 3 characters")

        if not SecurityConfig.USERNAME_PATTERN.match(username):
            raise ValidationError("Username can only contain letters, numbers, dots, hyphens, and underscores")

        return username


class SecureBaseModel(BaseModel):
    """Base Pydantic model with automatic input validation and sanitization."""

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        str_strip_whitespace = True

    def __init__(self, **data):
        """Initialize with validation."""
        validator = InputValidator(strict_mode=True)

        # Validate and sanitize all string fields
        for field_name, field_value in data.items():
            if isinstance(field_value, str):
                field_info = self.__fields__.get(field_name)
                if field_info:
                    # Determine field type from field info
                    field_type = 'string'
                    if 'email' in field_name.lower():
                        field_type = 'email'
                    elif 'username' in field_name.lower():
                        field_type = 'username'
                    elif 'filename' in field_name.lower() or 'file_name' in field_name.lower():
                        field_type = 'filename'
                    elif 'url' in field_name.lower():
                        field_type = 'url'

                    # Get max length from field constraints
                    max_length = None
                    if hasattr(field_info, 'constraints'):
                        max_length = getattr(field_info.constraints, 'max_length', None)

                    # Validate and sanitize
                    data[field_name] = validator.validate_and_sanitize(
                        field_value,
                        field_type,
                        max_length=max_length
                    )

        super().__init__(**data)


# Global validator instance
validator = InputValidator(strict_mode=True)


def validate_input(value: Any, field_type: str = 'string', **kwargs) -> Any:
    """Convenience function for input validation.

    Args:
        value: Input value
        field_type: Type of field
        **kwargs: Additional validation options

    Returns:
        Validated value
    """
    return validator.validate_and_sanitize(value, field_type, **kwargs)
