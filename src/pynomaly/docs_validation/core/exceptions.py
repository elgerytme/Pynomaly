"""Exception classes for documentation validation."""


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class ConfigurationError(ValidationError):
    """Exception raised for configuration-related errors."""
    pass


class CheckerError(ValidationError):
    """Exception raised when a checker fails to execute."""
    pass


class FileNotFoundError(ValidationError):
    """Exception raised when a required file is not found."""
    pass


class ParseError(ValidationError):
    """Exception raised when a document cannot be parsed."""
    pass
