"""
CLI-specific error handling utilities.
"""

import functools
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from .unified_exceptions import (
    ConfigurationError,
    DataIntegrityError,
    ValidationError,
    create_validation_error,
)

console = Console()


def handle_cli_errors(func: Callable) -> Callable:
    """
    Decorator to handle CLI errors gracefully with rich formatting.

    Args:
        func: The function to wrap with error handling

    Returns:
        Wrapped function with error handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(1)
        except ValidationError as e:
            console.print(
                Panel(
                    f"[red]Validation Error:[/red] {e.message}",
                    title="❌ Input Validation Failed",
                    border_style="red",
                )
            )
            sys.exit(1)
        except ConfigurationError as e:
            console.print(
                Panel(
                    f"[red]Configuration Error:[/red] {e.message}",
                    title="⚙️ Configuration Issue",
                    border_style="red",
                )
            )
            sys.exit(1)
        except DataIntegrityError as e:
            console.print(
                Panel(
                    f"[red]Data Error:[/red] {e.message}",
                    title="📊 Data Issue",
                    border_style="red",
                )
            )
            sys.exit(1)
        except FileNotFoundError as e:
            console.print(
                Panel(
                    f"[red]File not found:[/red] {e}",
                    title="📁 File Error",
                    border_style="red",
                )
            )
            sys.exit(1)
        except PermissionError as e:
            console.print(
                Panel(
                    f"[red]Permission denied:[/red] {e}",
                    title="🔒 Permission Error",
                    border_style="red",
                )
            )
            sys.exit(1)
        except Exception as e:
            console.print(
                Panel(
                    f"[red]Unexpected error:[/red] {e}",
                    title="💥 Internal Error",
                    border_style="red",
                )
            )
            console.print(
                "[dim]If this error persists, please report it as a bug.[/dim]"
            )
            sys.exit(1)

    return wrapper


def validate_file_exists(file_path: str | Path) -> Path:
    """
    Validate that a file exists and is readable.

    Args:
        file_path: Path to the file to validate

    Returns:
        Path object for the validated file

    Raises:
        ValidationError: If file doesn't exist or isn't readable
    """
    path = Path(file_path)

    if not path.exists():
        raise create_validation_error(
            f"File not found: {file_path}",
            details={
                "file_path": str(file_path),
                "absolute_path": str(path.absolute()),
            },
        )

    if not path.is_file():
        raise create_validation_error(
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
        raise create_validation_error(
            f"File is not readable: {file_path}",
            details={"file_path": str(file_path), "error": "permission_denied"},
        )
    except Exception as e:
        raise create_validation_error(
            f"Cannot access file: {file_path}",
            details={"file_path": str(file_path), "error": str(e)},
        )

    return path


def validate_algorithm_name(
    algorithm: str, available_algorithms: list[str] = None
) -> str:
    """
    Validate algorithm name.

    Args:
        algorithm: Algorithm name to validate
        available_algorithms: List of valid algorithms (optional, uses default if None)

    Returns:
        Validated algorithm name

    Raises:
        ValidationError: If algorithm is invalid
    """
    if not algorithm:
        raise create_validation_error("Algorithm name cannot be empty")

    if available_algorithms is None:
        available_algorithms = [
            "IsolationForest",
            "LOF",
            "OneClassSVM",
            "ABOD",
            "CBLOF",
            "HBOS",
            "KNN",
            "LODA",
            "OCSVM",
            "PCA",
            "MCD",
            "LMDD",
            "COPOD",
            "ECOD",
            "SOD",
            "LUNAR",
            "GMM",
            "INNE",
            "FB",
            "SUOD",
            "LSCP",
            "XGBOD",
            "DeepSVDD",
            "AnoGAN",
            "ALAD",
            "SO_GAAL",
            "MO_GAAL",
            "AUTO_ENCODER",
            "VAE",
            "SOGAAL",
            "MOGAAL",
            "DAGMM",
            "ANOGAN",
            "FENCE",
            "RGRAPH",
        ]

    if algorithm not in available_algorithms:
        # Try case-insensitive match
        algorithm_lower = algorithm.lower()
        matches = [
            alg for alg in available_algorithms if alg.lower() == algorithm_lower
        ]

        if matches:
            return matches[0]  # Return the correctly cased version

        raise create_validation_error(
            f"Unknown algorithm: {algorithm}",
            details={
                "requested_algorithm": algorithm,
                "available_algorithms": available_algorithms,
                "suggestion": f"Did you mean one of: {', '.join(available_algorithms[:3])}?",
            },
        )

    return algorithm


def validate_contamination_rate(contamination: str | float) -> float:
    """
    Validate contamination rate.

    Args:
        contamination: Contamination rate to validate

    Returns:
        Validated contamination rate

    Raises:
        ValidationError: If contamination rate is invalid
    """
    if not isinstance(contamination, (int, float, str)):
        raise create_validation_error(
            f"Contamination rate must be a number, got {type(contamination).__name__}",
            details={"value": contamination, "type": type(contamination).__name__},
        )

    try:
        rate = float(contamination)
    except (ValueError, TypeError):
        raise create_validation_error(
            f"Invalid contamination rate: {contamination}",
            details={"value": contamination, "type": type(contamination).__name__},
        )

    if not (0.0 < rate < 1.0):
        raise create_validation_error(
            f"Contamination rate must be between 0.0 and 1.0, got: {rate}",
            details={"value": rate, "valid_range": "0.0 < rate < 1.0"},
        )

    return rate


def validate_data_format(file_path: str | Path) -> str:
    """
    Validate data file format.

    Args:
        file_path: Path to the data file

    Returns:
        Detected file format

    Raises:
        ValidationError: If format is unsupported
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    supported_formats = {
        ".csv": "csv",
        ".json": "json",
        ".parquet": "parquet",
        ".xlsx": "excel",
        ".xls": "excel",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
    }

    if suffix not in supported_formats:
        raise create_validation_error(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: {', '.join(supported_formats.keys())}"
        )

    return supported_formats[suffix]


def validate_output_path(output_path: str | Path) -> Path:
    """
    Validate output file path.

    Args:
        output_path: Path for output file

    Returns:
        Validated output path

    Raises:
        ValidationError: If path is invalid
    """
    path = Path(output_path)

    # Check if parent directory exists
    if not path.parent.exists():
        raise create_validation_error(f"Output directory does not exist: {path.parent}")

    # Check if parent directory is writable
    if not os.access(path.parent, os.W_OK):
        raise create_validation_error(
            f"Output directory is not writable: {path.parent}"
        )

    return path


def validate_positive_integer(value: str | int, name: str) -> int:
    """
    Validate positive integer value.

    Args:
        value: Value to validate
        name: Name of the parameter for error messages

    Returns:
        Validated integer value

    Raises:
        ValidationError: If value is not a positive integer
    """
    try:
        int_value = int(value)
    except (ValueError, TypeError):
        raise create_validation_error(f"Invalid {name}: {value} (must be an integer)")

    if int_value <= 0:
        raise create_validation_error(f"Invalid {name}: {int_value} (must be positive)")

    return int_value


def validate_dataset_name(name: str) -> str:
    """
    Validate dataset name.

    Args:
        name: Dataset name to validate

    Returns:
        Validated dataset name

    Raises:
        ValidationError: If name is invalid
    """
    if not name or not name.strip():
        raise create_validation_error("Dataset name cannot be empty")

    # Check for invalid characters
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for char in invalid_chars:
        if char in name:
            raise create_validation_error(
                f"Dataset name contains invalid character: {char}"
            )

    # Check length
    if len(name) > 255:
        raise create_validation_error("Dataset name is too long (max 255 characters)")

    return name.strip()


def format_cli_error(error: Exception) -> str:
    """
    Format error message for CLI display.

    Args:
        error: Exception to format

    Returns:
        Formatted error message
    """
    if hasattr(error, "message"):
        return str(error.message)
    return str(error)


def print_success(message: str, title: str | None = None) -> None:
    """
    Print success message with rich formatting.

    Args:
        message: Success message to display
        title: Optional title for the panel
    """
    console.print(
        Panel(
            f"[green]{message}[/green]",
            title=title or "✅ Success",
            border_style="green",
        )
    )


def print_warning(message: str, title: str | None = None) -> None:
    """
    Print warning message with rich formatting.

    Args:
        message: Warning message to display
        title: Optional title for the panel
    """
    console.print(
        Panel(
            f"[yellow]{message}[/yellow]",
            title=title or "⚠️ Warning",
            border_style="yellow",
        )
    )


def validate_data_shape(data, min_samples: int = 1, min_features: int = 1) -> None:
    """
    Validate data shape requirements.

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
        raise create_validation_error(
            "Data must have a 'shape' attribute (pandas DataFrame or numpy array)"
        )

    if n_samples < min_samples:
        raise create_validation_error(
            f"Insufficient samples: need at least {min_samples}, got {n_samples}"
        )

    if n_features < min_features:
        raise create_validation_error(
            f"Insufficient features: need at least {min_features}, got {n_features}"
        )


def safe_import(module_name: str, error_message: str | None = None) -> Any:
    """
    Safely import a module with user-friendly error handling.

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


def create_user_friendly_message(error: Exception) -> str:
    """
    Create user-friendly error message from exception.

    Args:
        error: Exception to convert

    Returns:
        User-friendly error message
    """
    if hasattr(error, "message"):
        return str(error.message)

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


def print_info(message: str, title: str | None = None) -> None:
    """
    Print info message with rich formatting.

    Args:
        message: Info message to display
        title: Optional title for the panel
    """
    console.print(
        Panel(f"[blue]{message}[/blue]", title=title or "ℹ️ Info", border_style="blue")
    )
