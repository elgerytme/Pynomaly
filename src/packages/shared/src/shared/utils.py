"""
Common utilities for the shared package.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, TypeVar

from .types import ValidationResult, ValidationError
from .value_objects import Email

T = TypeVar('T')


class ValidationUtils:
    """Common validation patterns and utilities."""
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        """Validate an email address."""
        try:
            Email(email)
            return ValidationResult.success()
        except ValueError as e:
            error = ValidationError(
                field="email",
                value=email,
                message=str(e),
                code="INVALID_EMAIL"
            )
            return ValidationResult.failure([error])
    
    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any], 
        required_fields: List[str]
    ) -> ValidationResult:
        """Validate that required fields are present and not empty."""
        errors = []
        
        for field in required_fields:
            if field not in data:
                errors.append(ValidationError(
                    field=field,
                    value=None,
                    message=f"Field '{field}' is required",
                    code="MISSING_FIELD"
                ))
            elif not data[field] and data[field] != 0:  # Allow 0 but not empty string, None, etc.
                errors.append(ValidationError(
                    field=field,
                    value=data[field],
                    message=f"Field '{field}' cannot be empty",
                    code="EMPTY_FIELD"
                ))
        
        if errors:
            return ValidationResult.failure(errors)
        return ValidationResult.success()
    
    @staticmethod
    def validate_string_length(
        value: str,
        field_name: str,
        min_length: int = 0,
        max_length: int | None = None
    ) -> ValidationResult:
        """Validate string length constraints."""
        errors = []
        
        if len(value) < min_length:
            errors.append(ValidationError(
                field=field_name,
                value=value,
                message=f"Field '{field_name}' must be at least {min_length} characters",
                code="TOO_SHORT"
            ))
        
        if max_length and len(value) > max_length:
            errors.append(ValidationError(
                field=field_name,
                value=value,
                message=f"Field '{field_name}' cannot exceed {max_length} characters",
                code="TOO_LONG"
            ))
        
        if errors:
            return ValidationResult.failure(errors)
        return ValidationResult.success()
    
    @staticmethod
    def validate_numeric_range(
        value: float | int,
        field_name: str,
        min_value: float | int | None = None,
        max_value: float | int | None = None
    ) -> ValidationResult:
        """Validate numeric range constraints."""
        errors = []
        
        if min_value is not None and value < min_value:
            errors.append(ValidationError(
                field=field_name,
                value=value,
                message=f"Field '{field_name}' must be at least {min_value}",
                code="TOO_SMALL"
            ))
        
        if max_value is not None and value > max_value:
            errors.append(ValidationError(
                field=field_name,
                value=value,
                message=f"Field '{field_name}' cannot exceed {max_value}",
                code="TOO_LARGE"
            ))
        
        if errors:
            return ValidationResult.failure(errors)
        return ValidationResult.success()


class DateTimeUtils:
    """Date and time manipulation utilities."""
    
    @staticmethod
    def now_utc() -> datetime:
        """Get current UTC datetime."""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def format_iso(dt: datetime) -> str:
        """Format datetime as ISO string."""
        return dt.isoformat()
    
    @staticmethod
    def parse_iso(iso_string: str) -> datetime:
        """Parse ISO string to datetime."""
        return datetime.fromisoformat(iso_string)
    
    @staticmethod
    def to_unix_timestamp(dt: datetime) -> float:
        """Convert datetime to Unix timestamp."""
        return dt.timestamp()
    
    @staticmethod
    def from_unix_timestamp(timestamp: float) -> datetime:
        """Convert Unix timestamp to datetime."""
        return datetime.fromtimestamp(timestamp, timezone.utc)
    
    @staticmethod
    def is_within_range(
        dt: datetime,
        start: datetime,
        end: datetime
    ) -> bool:
        """Check if datetime is within a range."""
        return start <= dt <= end


class SerializationUtils:
    """Serialization and deserialization utilities."""
    
    @staticmethod
    def to_json(obj: Any, **kwargs) -> str:
        """Convert object to JSON string."""
        return json.dumps(obj, default=SerializationUtils._serialize_default, **kwargs)
    
    @staticmethod
    def from_json(json_str: str, **kwargs) -> Any:
        """Convert JSON string to object."""
        return json.loads(json_str, **kwargs)
    
    @staticmethod
    def _serialize_default(obj: Any) -> Any:
        """Default serializer for JSON encoding."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '_asdict'):  # namedtuple
            return obj._asdict()
        else:
            return str(obj)
    
    @staticmethod
    def to_dict(obj: Any) -> Dict[str, Any]:
        """Convert object to dictionary."""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '_asdict'):  # namedtuple
            return obj._asdict()
        else:
            raise ValueError(f"Cannot convert {type(obj)} to dict")


class LoggingUtils:
    """Structured logging utilities."""
    
    @staticmethod
    def create_log_context(
        operation: str,
        user_id: str | None = None,
        request_id: str | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a structured logging context."""
        context = {
            'operation': operation,
            'timestamp': DateTimeUtils.now_utc().isoformat(),
        }
        
        if user_id:
            context['user_id'] = user_id
        if request_id:
            context['request_id'] = request_id
        
        # Add any additional context
        context.update(kwargs)
        
        return context
    
    @staticmethod
    def format_error_context(
        error: Exception,
        operation: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Create error logging context."""
        context = LoggingUtils.create_log_context(operation, **kwargs)
        context.update({
            'error_type': type(error).__name__,
            'error_message': str(error),
            'error_class': error.__class__.__module__ + '.' + error.__class__.__qualname__,
        })
        
        return context


class StringUtils:
    """String manipulation utilities."""
    
    @staticmethod
    def snake_to_camel(snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split('_')
        return components[0] + ''.join(word.capitalize() for word in components[1:])
    
    @staticmethod
    def camel_to_snake(camel_str: str) -> str:
        """Convert camelCase to snake_case."""
        # Insert underscore before uppercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    @staticmethod
    def slugify(text: str) -> str:
        """Convert text to URL-friendly slug."""
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')
    
    @staticmethod
    def truncate(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to maximum length with suffix."""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def is_blank(text: str | None) -> bool:
        """Check if string is None, empty, or only whitespace."""
        return text is None or text.strip() == ""
    
    @staticmethod
    def ensure_string(value: Any) -> str:
        """Ensure value is converted to string."""
        if value is None:
            return ""
        return str(value)