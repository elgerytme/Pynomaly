"""Custom exceptions for data profiling operations."""

from typing import Any, Dict, Optional, List


class DataProfilingError(Exception):
    """Base exception for data profiling operations."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class DataSourceError(DataProfilingError):
    """Exception raised when there are issues with data sources."""
    pass


class FileFormatError(DataSourceError):
    """Exception raised when file format is not supported or invalid."""
    
    def __init__(self, file_path: str, format_type: Optional[str] = None, 
                 supported_formats: Optional[List[str]] = None):
        self.file_path = file_path
        self.format_type = format_type
        self.supported_formats = supported_formats or []
        
        message = f"Unsupported or invalid file format for: {file_path}"
        if format_type:
            message += f" (detected format: {format_type})"
        if supported_formats:
            message += f". Supported formats: {', '.join(supported_formats)}"
        
        context = {
            'file_path': file_path,
            'format_type': format_type,
            'supported_formats': supported_formats
        }
        super().__init__(message, context)


class DatabaseConnectionError(DataSourceError):
    """Exception raised when database connection fails."""
    
    def __init__(self, db_type: str, host: Optional[str] = None, 
                 database: Optional[str] = None, original_error: Optional[Exception] = None):
        self.db_type = db_type
        self.host = host
        self.database = database
        self.original_error = original_error
        
        message = f"Failed to connect to {db_type} database"
        if host:
            message += f" at {host}"
        if database:
            message += f", database: {database}"
        if original_error:
            message += f". Error: {str(original_error)}"
        
        context = {
            'db_type': db_type,
            'host': host,
            'database': database,
            'original_error': str(original_error) if original_error else None
        }
        super().__init__(message, context)


class DataLoadError(DataSourceError):
    """Exception raised when data loading fails."""
    
    def __init__(self, source: str, operation: str, original_error: Optional[Exception] = None):
        self.source = source
        self.operation = operation
        self.original_error = original_error
        
        message = f"Failed to {operation} data from {source}"
        if original_error:
            message += f". Error: {str(original_error)}"
        
        context = {
            'source': source,
            'operation': operation,
            'original_error': str(original_error) if original_error else None
        }
        super().__init__(message, context)


class SchemaAnalysisError(DataProfilingError):
    """Exception raised during schema analysis."""
    
    def __init__(self, column_name: Optional[str] = None, analysis_type: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        self.column_name = column_name
        self.analysis_type = analysis_type
        self.original_error = original_error
        
        message = "Schema analysis failed"
        if analysis_type:
            message += f" during {analysis_type}"
        if column_name:
            message += f" for column '{column_name}'"
        if original_error:
            message += f". Error: {str(original_error)}"
        
        context = {
            'column_name': column_name,
            'analysis_type': analysis_type,
            'original_error': str(original_error) if original_error else None
        }
        super().__init__(message, context)


class StatisticalAnalysisError(DataProfilingError):
    """Exception raised during statistical analysis."""
    
    def __init__(self, column_name: Optional[str] = None, statistic: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        self.column_name = column_name
        self.statistic = statistic
        self.original_error = original_error
        
        message = "Statistical analysis failed"
        if statistic:
            message += f" for statistic '{statistic}'"
        if column_name:
            message += f" on column '{column_name}'"
        if original_error:
            message += f". Error: {str(original_error)}"
        
        context = {
            'column_name': column_name,
            'statistic': statistic,
            'original_error': str(original_error) if original_error else None
        }
        super().__init__(message, context)


class PatternDiscoveryError(DataProfilingError):
    """Exception raised during pattern discovery."""
    
    def __init__(self, column_name: Optional[str] = None, pattern_type: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        self.column_name = column_name
        self.pattern_type = pattern_type
        self.original_error = original_error
        
        message = "Pattern discovery failed"
        if pattern_type:
            message += f" for pattern type '{pattern_type}'"
        if column_name:
            message += f" on column '{column_name}'"
        if original_error:
            message += f". Error: {str(original_error)}"
        
        context = {
            'column_name': column_name,
            'pattern_type': pattern_type,
            'original_error': str(original_error) if original_error else None
        }
        super().__init__(message, context)


class QualityAssessmentError(DataProfilingError):
    """Exception raised during quality assessment."""
    
    def __init__(self, assessment_type: Optional[str] = None, column_name: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        self.assessment_type = assessment_type
        self.column_name = column_name
        self.original_error = original_error
        
        message = "Quality assessment failed"
        if assessment_type:
            message += f" for assessment '{assessment_type}'"
        if column_name:
            message += f" on column '{column_name}'"
        if original_error:
            message += f". Error: {str(original_error)}"
        
        context = {
            'assessment_type': assessment_type,
            'column_name': column_name,
            'original_error': str(original_error) if original_error else None
        }
        super().__init__(message, context)


class ValidationError(DataProfilingError):
    """Exception raised when validation fails."""
    
    def __init__(self, field: str, value: Any, constraint: str, 
                 expected: Optional[Any] = None):
        self.field = field
        self.value = value
        self.constraint = constraint
        self.expected = expected
        
        message = f"Validation failed for field '{field}': {constraint}"
        if expected is not None:
            message += f". Expected: {expected}, Got: {value}"
        else:
            message += f". Value: {value}"
        
        context = {
            'field': field,
            'value': value,
            'constraint': constraint,
            'expected': expected
        }
        super().__init__(message, context)


class ConfigurationError(DataProfilingError):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, config_key: str, issue: str, valid_options: Optional[List[str]] = None):
        self.config_key = config_key
        self.issue = issue
        self.valid_options = valid_options or []
        
        message = f"Configuration error for '{config_key}': {issue}"
        if valid_options:
            message += f". Valid options: {', '.join(map(str, valid_options))}"
        
        context = {
            'config_key': config_key,
            'issue': issue,
            'valid_options': valid_options
        }
        super().__init__(message, context)


class ResourceLimitError(DataProfilingError):
    """Exception raised when resource limits are exceeded."""
    
    def __init__(self, resource_type: str, limit: Any, current: Any, unit: Optional[str] = None):
        self.resource_type = resource_type
        self.limit = limit
        self.current = current
        self.unit = unit or ""
        
        message = f"Resource limit exceeded for {resource_type}: {current}{unit} > {limit}{unit}"
        
        context = {
            'resource_type': resource_type,
            'limit': limit,
            'current': current,
            'unit': unit
        }
        super().__init__(message, context)


class DependencyError(DataProfilingError):
    """Exception raised when required dependencies are missing."""
    
    def __init__(self, dependency: str, feature: str, install_command: Optional[str] = None):
        self.dependency = dependency
        self.feature = feature
        self.install_command = install_command
        
        message = f"Missing dependency '{dependency}' required for {feature}"
        if install_command:
            message += f". Install with: {install_command}"
        
        context = {
            'dependency': dependency,
            'feature': feature,
            'install_command': install_command
        }
        super().__init__(message, context)


def handle_exception(func):
    """Decorator to handle and convert common exceptions to custom exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise DataLoadError(str(e.filename), "load", e)
        except PermissionError as e:
            raise DataLoadError(str(e.filename), "access", e)
        except pd.errors.EmptyDataError as e:
            raise DataLoadError("unknown", "load empty data", e)
        except pd.errors.ParserError as e:
            raise FileFormatError("unknown", "parsing error")
        except ImportError as e:
            # Try to extract dependency name from error message
            dependency = str(e).split("'")[1] if "'" in str(e) else "unknown"
            raise DependencyError(dependency, func.__name__)
        except MemoryError as e:
            raise ResourceLimitError("memory", "unknown", "exceeded")
        except Exception as e:
            # Re-raise custom exceptions as-is
            if isinstance(e, DataProfilingError):
                raise
            # Wrap other exceptions
            raise DataProfilingError(f"Unexpected error in {func.__name__}: {str(e)}", 
                                   {'function': func.__name__, 'original_error': str(e)})
    
    return wrapper


def safe_execute(operation_name: str, func, *args, **kwargs):
    """Safely execute a function and handle exceptions."""
    try:
        return func(*args, **kwargs)
    except DataProfilingError:
        # Re-raise custom exceptions
        raise
    except Exception as e:
        # Wrap unexpected exceptions
        raise DataProfilingError(
            f"Failed to execute {operation_name}: {str(e)}",
            {
                'operation': operation_name,
                'args': str(args),
                'kwargs': str(kwargs),
                'exception_type': type(e).__name__
            }
        )