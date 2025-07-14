"""
Input validation and sanitization for Pynomaly API.

This module provides:
- Input validation and sanitization
- SQL injection prevention
- XSS prevention
- Data type validation
- Schema validation
"""

import html
import json
import logging
import re
from datetime import datetime
from typing import Any

import bleach
from marshmallow import Schema, ValidationError, fields

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation."""

    def __init__(self):
        self.email_pattern = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
        self.phone_pattern = re.compile(r"^\+?1?\d{9,15}$")
        self.uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        self.ip_pattern = re.compile(
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        )

        # Dangerous SQL keywords
        self.sql_keywords = {
            "select",
            "insert",
            "update",
            "delete",
            "drop",
            "create",
            "alter",
            "exec",
            "execute",
            "union",
            "script",
            "declare",
            "grant",
            "revoke",
        }

        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>.*?</embed>",
            r"<link[^>]*>",
            r"<meta[^>]*>",
        ]

    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        if not email or not isinstance(email, str):
            return False
        return bool(self.email_pattern.match(email.lower()))

    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format."""
        if not phone or not isinstance(phone, str):
            return False
        return bool(self.phone_pattern.match(phone))

    def validate_uuid(self, uuid_str: str) -> bool:
        """Validate UUID format."""
        if not uuid_str or not isinstance(uuid_str, str):
            return False
        return bool(self.uuid_pattern.match(uuid_str.lower()))

    def validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format."""
        if not ip or not isinstance(ip, str):
            return False
        return bool(self.ip_pattern.match(ip))

    def validate_string_length(
        self, value: str, min_length: int = 0, max_length: int = 255
    ) -> bool:
        """Validate string length."""
        if not isinstance(value, str):
            return False
        return min_length <= len(value) <= max_length

    def validate_numeric_range(
        self,
        value: int | float,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> bool:
        """Validate numeric range."""
        if not isinstance(value, (int, float)):
            return False

        if min_val is not None and value < min_val:
            return False

        if max_val is not None and value > max_val:
            return False

        return True

    def validate_date_format(self, date_str: str, format_str: str = "%Y-%m-%d") -> bool:
        """Validate date format."""
        if not isinstance(date_str, str):
            return False

        try:
            datetime.strptime(date_str, format_str)
            return True
        except ValueError:
            return False

    def detect_sql_injection(self, input_str: str) -> bool:
        """Detect potential SQL injection attempts."""
        if not isinstance(input_str, str):
            return False

        input_lower = input_str.lower()

        # Check for SQL keywords
        for keyword in self.sql_keywords:
            if keyword in input_lower:
                return True

        # Check for common SQL injection patterns
        injection_patterns = [
            r"'\s*or\s*'",
            r"'\s*and\s*'",
            r"'\s*;",
            r"--",
            r"/\*.*\*/",
            r"union\s+select",
            r"drop\s+table",
            r"exec\s*\(",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, input_lower):
                return True

        return False

    def detect_xss(self, input_str: str) -> bool:
        """Detect potential XSS attempts."""
        if not isinstance(input_str, str):
            return False

        input_lower = input_str.lower()

        for pattern in self.xss_patterns:
            if re.search(pattern, input_lower):
                return True

        return False

    def validate_json(self, json_str: str) -> bool:
        """Validate JSON format."""
        if not isinstance(json_str, str):
            return False

        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False

    def validate_file_extension(
        self, filename: str, allowed_extensions: list[str]
    ) -> bool:
        """Validate file extension."""
        if not isinstance(filename, str):
            return False

        if "." not in filename:
            return False

        extension = filename.rsplit(".", 1)[1].lower()
        return extension in [ext.lower() for ext in allowed_extensions]

    def validate_file_size(
        self, file_size: int, max_size: int = 10 * 1024 * 1024
    ) -> bool:
        """Validate file size (default 10MB)."""
        return 0 < file_size <= max_size


class SecuritySanitizer:
    """Security-focused input sanitization."""

    def __init__(self):
        # HTML tags allowed for rich text
        self.allowed_tags = ["b", "i", "u", "em", "strong", "p", "br", "ul", "ol", "li"]
        self.allowed_attributes = {"*": ["class"], "a": ["href", "title"]}

    def sanitize_string(self, input_str: str) -> str:
        """Sanitize string input."""
        if not isinstance(input_str, str):
            return ""

        # Remove null bytes
        sanitized = input_str.replace("\x00", "")

        # Remove control characters except newlines and tabs
        sanitized = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", sanitized)

        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", sanitized).strip()

        return sanitized

    def sanitize_html(self, html_str: str) -> str:
        """Sanitize HTML input."""
        if not isinstance(html_str, str):
            return ""

        # Use bleach to sanitize HTML
        return bleach.clean(
            html_str,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True,
        )

    def sanitize_sql(self, input_str: str) -> str:
        """Sanitize SQL input (basic escaping)."""
        if not isinstance(input_str, str):
            return ""

        # Escape SQL special characters
        sanitized = input_str.replace("'", "''")
        sanitized = sanitized.replace('"', '""')
        sanitized = sanitized.replace("\\", "\\\\")

        return sanitized

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename."""
        if not isinstance(filename, str):
            return ""

        # Remove path traversal attempts
        sanitized = filename.replace("..", "").replace("/", "").replace("\\", "")

        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"|?*]', "", sanitized)

        # Limit length
        sanitized = sanitized[:255]

        # Ensure not empty
        if not sanitized.strip():
            sanitized = "file"

        return sanitized

    def sanitize_json(self, json_str: str) -> str:
        """Sanitize JSON input."""
        if not isinstance(json_str, str):
            return "{}"

        try:
            # Parse and re-serialize to remove potentially malicious content
            parsed = json.loads(json_str)
            return json.dumps(parsed, separators=(",", ":"))
        except json.JSONDecodeError:
            return "{}"

    def escape_html_entities(self, input_str: str) -> str:
        """Escape HTML entities."""
        if not isinstance(input_str, str):
            return ""

        return html.escape(input_str)

    def sanitize_url(self, url: str) -> str:
        """Sanitize URL input."""
        if not isinstance(url, str):
            return ""

        # Remove dangerous protocols
        dangerous_protocols = ["javascript:", "data:", "vbscript:", "file:", "ftp:"]
        url_lower = url.lower()

        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                return ""

        # Only allow http and https
        if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
            return ""

        return url


class SchemaValidator:
    """Schema-based validation using Marshmallow."""

    def __init__(self):
        self.schemas = {}

    def register_schema(self, name: str, schema: Schema) -> None:
        """Register a validation schema."""
        self.schemas[name] = schema

    def validate_data(self, schema_name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Validate data against schema."""
        if schema_name not in self.schemas:
            raise ValueError(f"Schema {schema_name} not found")

        schema = self.schemas[schema_name]

        try:
            return schema.load(data)
        except ValidationError as e:
            logger.warning(f"Schema validation failed: {e.messages}")
            raise ValidationError(f"Invalid data: {e.messages}")


# Common validation schemas
class UserSchema(Schema):
    """User data validation schema."""

    username = fields.Str(required=True, validate=lambda x: 3 <= len(x) <= 50)
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=lambda x: len(x) >= 8)
    first_name = fields.Str(required=True, validate=lambda x: 1 <= len(x) <= 50)
    last_name = fields.Str(required=True, validate=lambda x: 1 <= len(x) <= 50)
    phone = fields.Str(required=False)
    role = fields.Str(
        required=False, validate=lambda x: x in ["admin", "user", "analyst"]
    )


class ModelSchema(Schema):
    """Model data validation schema."""

    name = fields.Str(required=True, validate=lambda x: 1 <= len(x) <= 100)
    description = fields.Str(required=False, validate=lambda x: len(x) <= 500)
    algorithm = fields.Str(
        required=True,
        validate=lambda x: x in ["isolation_forest", "one_class_svm", "autoencoder"],
    )
    parameters = fields.Dict(required=False)
    training_data = fields.Dict(required=False)


class DatasetSchema(Schema):
    """Dataset validation schema."""

    name = fields.Str(required=True, validate=lambda x: 1 <= len(x) <= 100)
    description = fields.Str(required=False, validate=lambda x: len(x) <= 500)
    source = fields.Str(required=True)
    format = fields.Str(
        required=True, validate=lambda x: x in ["csv", "json", "parquet"]
    )
    size = fields.Int(required=False, validate=lambda x: x > 0)


class SecurityValidator:
    """Comprehensive security validation."""

    def __init__(self):
        self.input_validator = InputValidator()
        self.sanitizer = SecuritySanitizer()
        self.schema_validator = SchemaValidator()

        # Register common schemas
        self.schema_validator.register_schema("user", UserSchema())
        self.schema_validator.register_schema("model", ModelSchema())
        self.schema_validator.register_schema("dataset", DatasetSchema())

    def validate_and_sanitize(
        self, data: dict[str, Any], schema_name: str | None = None
    ) -> dict[str, Any]:
        """Validate and sanitize input data."""
        # First sanitize all string inputs
        sanitized_data = self._sanitize_recursive(data)

        # Then validate against schema if provided
        if schema_name:
            validated_data = self.schema_validator.validate_data(
                schema_name, sanitized_data
            )
            return validated_data

        return sanitized_data

    def _sanitize_recursive(self, data: Any) -> Any:
        """Recursively sanitize data structure."""
        if isinstance(data, dict):
            return {key: self._sanitize_recursive(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_recursive(item) for item in data]
        elif isinstance(data, str):
            return self.sanitizer.sanitize_string(data)
        else:
            return data

    def check_security_threats(self, data: dict[str, Any]) -> list[str]:
        """Check for security threats in input data."""
        threats = []

        def check_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_recursive(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    check_recursive(value, f"{path}[{i}]")
            elif isinstance(obj, str):
                if self.input_validator.detect_sql_injection(obj):
                    threats.append(f"SQL injection detected in {path}")
                if self.input_validator.detect_xss(obj):
                    threats.append(f"XSS attempt detected in {path}")

        check_recursive(data)
        return threats

    def validate_file_upload(
        self, filename: str, file_size: int, file_content: bytes
    ) -> dict[str, Any]:
        """Validate file upload."""
        result = {"valid": True, "errors": [], "warnings": []}

        # Validate filename
        if not self.input_validator.validate_file_extension(
            filename, ["csv", "json", "txt", "parquet"]
        ):
            result["valid"] = False
            result["errors"].append("Invalid file extension")

        # Validate file size (10MB limit)
        if not self.input_validator.validate_file_size(file_size, 10 * 1024 * 1024):
            result["valid"] = False
            result["errors"].append("File size too large")

        # Check for malicious content (basic check)
        if b"<script" in file_content or b"javascript:" in file_content:
            result["valid"] = False
            result["errors"].append("Potentially malicious content detected")

        return result
