"""Comprehensive error handling utilities for Pynomaly."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PynomaliError(Exception):
    """Base exception for all Pynomaly errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict:
        """Convert error to dictionary for API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(PynomaliError):
    """Error raised when data validation fails."""

    pass


class ConfigurationError(PynomaliError):
    """Error raised when configuration is invalid."""

    pass


class ResourceNotFoundError(PynomaliError):
    """Error raised when a required resource is not found."""

    pass


class PerformanceError(PynomaliError):
    """Error raised when performance constraints are violated."""

    pass


def validate_file_exists(file_path: str | Path) -> Path:
    """Validate that a file exists and is readable.

    Args:
        file_path: Path to the file to validate

    Returns:
        Path object for the validated file

    Raises:
        ResourceNotFoundError: If file doesn't exist or isn't readable
    """
    path = Path(file_path)

    if not path.exists():
        raise ResourceNotFoundError(
            f"File not found: {file_path}",
            details={
                "file_path": str(file_path),
                "absolute_path": str(path.absolute()),
            },
        )

    if not path.is_file():
        raise ResourceNotFoundError(
            f"Path is not a file: {file_path}",
            details={
                "file_path": str(file_path),
                "path_type": "directory" if path.is_dir() else "other",
            },
        )

    try:
        with open(path) as f:
            f.read(1)  # Try to read one character
    except PermissionError:
        raise ResourceNotFoundError(
            f"File is not readable: {file_path}",
            details={"file_path": str(file_path), "error": "permission_denied"},
        )
    except Exception as e:
        raise ResourceNotFoundError(
            f"Cannot access file: {file_path}",
            details={"file_path": str(file_path), "error": str(e)},
        )

    return path


def validate_data_format(file_path: Path) -> str:
    """Validate and detect data format.

    Args:
        file_path: Path to the data file

    Returns:
        String indicating the data format ('csv', 'json', etc.)

    Raises:
        ValidationError: If format is not supported
    """
    suffix = file_path.suffix.lower()

    supported_formats = {
        ".csv": "csv",
        ".json": "json",
        ".parquet": "parquet",
        ".xlsx": "excel",
        ".xls": "excel",
    }

    if suffix not in supported_formats:
        raise ValidationError(
            f"Unsupported file format: {suffix}",
            details={
                "file_path": str(file_path),
                "detected_format": suffix,
                "supported_formats": list(supported_formats.keys()),
            },
        )

    return supported_formats[suffix]


def validate_contamination_rate(contamination: float) -> float:
    """Validate contamination rate parameter.

    Args:
        contamination: Contamination rate value

    Returns:
        Validated contamination rate

    Raises:
        ValidationError: If contamination rate is invalid
    """
    if not isinstance(contamination, (int, float)):
        raise ValidationError(
            f"Contamination rate must be a number, got {type(contamination).__name__}",
            details={"value": contamination, "type": type(contamination).__name__},
        )

    contamination = float(contamination)

    if not 0.0 < contamination < 1.0:
        raise ValidationError(
            f"Contamination rate must be between 0.0 and 1.0, got {contamination}",
            details={"value": contamination, "valid_range": "0.0 < rate < 1.0"},
        )

    return contamination


def validate_algorithm_name(algorithm: str, available_algorithms: list[str]) -> str:
    """Validate algorithm name.

    Args:
        algorithm: Algorithm name to validate
        available_algorithms: List of available algorithm names

    Returns:
        Validated algorithm name

    Raises:
        ValidationError: If algorithm is not available
    """
    if not isinstance(algorithm, str):
        raise ValidationError(
            f"Algorithm name must be a string, got {type(algorithm).__name__}",
            details={"value": algorithm, "type": type(algorithm).__name__},
        )

    if algorithm not in available_algorithms:
        # Try case-insensitive match
        algorithm_lower = algorithm.lower()
        matches = [
            alg for alg in available_algorithms if alg.lower() == algorithm_lower
        ]

        if matches:
            return matches[0]  # Return the correctly cased version

        raise ValidationError(
            f"Unknown algorithm: {algorithm}",
            details={
                "requested_algorithm": algorithm,
                "available_algorithms": available_algorithms,
                "suggestion": f"Did you mean one of: {', '.join(available_algorithms[:3])}?",
            },
        )

    return algorithm


def handle_cli_errors(func: Callable) -> Callable:
    """Decorator to handle CLI command errors gracefully.

    Args:
        func: Function to wrap with error handling

    Returns:
        Wrapped function with error handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PynomaliError as e:
            print(f"Error: {e.message}")
            if e.details:
                print("Details:")
                for key, value in e.details.items():
                    print(f"  {key}: {value}")
            return False
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            logger.exception("Unexpected error in CLI command")
            return False

    return wrapper


def handle_api_errors(func: Callable) -> Callable:
    """Decorator to handle API endpoint errors gracefully.

    Args:
        func: Function to wrap with error handling

    Returns:
        Wrapped function with error handling
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except PynomaliError as e:
            logger.error(f"API error: {e.message}", extra={"details": e.details})
            raise
        except Exception as e:
            logger.exception("Unexpected API error")
            raise PynomaliError(
                "Internal server error occurred", details={"original_error": str(e)}
            )

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PynomaliError as e:
            logger.error(f"API error: {e.message}", extra={"details": e.details})
            raise
        except Exception as e:
            logger.exception("Unexpected API error")
            raise PynomaliError(
                "Internal server error occurred", details={"original_error": str(e)}
            )

    # Return appropriate wrapper based on function type
    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def safe_import(module_name: str, error_message: str | None = None) -> Any:
    """Safely import a module with user-friendly error handling.

    Args:
        module_name: Name of the module to import
        error_message: Custom error message if import fails

    Returns:
        Imported module

    Raises:
        ConfigurationError: If module cannot be imported
    """
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError as e:
        default_message = f"Required dependency '{module_name}' is not installed"
        message = error_message or default_message

        raise ConfigurationError(
            message,
            details={
                "module_name": module_name,
                "import_error": str(e),
                "suggestion": f"Install with: pip install {module_name}",
            },
        )


def validate_data_shape(data, min_samples: int = 1, min_features: int = 1) -> None:
    """Validate data shape requirements.

    Args:
        data: Data to validate (pandas DataFrame or numpy array)
        min_samples: Minimum number of samples required
        min_features: Minimum number of features required

    Raises:
        ValidationError: If data shape is invalid
    """
    if hasattr(data, "shape"):
        n_samples, n_features = data.shape
    else:
        raise ValidationError(
            "Data must have a 'shape' attribute (pandas DataFrame or numpy array)",
            details={"data_type": type(data).__name__},
        )

    if n_samples < min_samples:
        raise ValidationError(
            f"Insufficient samples: need at least {min_samples}, got {n_samples}",
            details={
                "current_samples": n_samples,
                "minimum_required": min_samples,
                "data_shape": data.shape,
            },
        )

    if n_features < min_features:
        raise ValidationError(
            f"Insufficient features: need at least {min_features}, got {n_features}",
            details={
                "current_features": n_features,
                "minimum_required": min_features,
                "data_shape": data.shape,
            },
        )


def create_user_friendly_message(error: Exception) -> str:
    """Create user-friendly error message from exception.

    Args:
        error: Exception to convert

    Returns:
        User-friendly error message
    """
    if isinstance(error, PynomaliError):
        return error.message

    error_type = type(error).__name__
    error_message = str(error)

    # Common error patterns with user-friendly messages
    friendly_messages = {
        "FileNotFoundError": "The specified file could not be found",
        "PermissionError": "Permission denied - check file permissions",
        "MemoryError": "Not enough memory to process the data",
        "KeyError": "Required data field is missing",
        "ValueError": "Invalid data value encountered",
        "TypeError": "Data type mismatch",
        "ImportError": "Required library is not installed",
        "ConnectionError": "Network connection failed",
        "TimeoutError": "Operation timed out",
    }

    base_message = friendly_messages.get(error_type, f"{error_type}: {error_message}")

    # Add specific guidance based on error type
    if "FileNotFoundError" in error_type:
        return f"{base_message}. Please check the file path and try again."
    elif "PermissionError" in error_type:
        return f"{base_message}. Make sure you have read access to the file."
    elif "MemoryError" in error_type:
        return f"{base_message}. Try using a smaller dataset or increasing available memory."
    elif "ImportError" in error_type:
        return f"{base_message}. Install missing dependencies with pip install."

    return base_message
