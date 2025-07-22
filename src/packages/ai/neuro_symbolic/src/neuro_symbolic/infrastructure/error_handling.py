"""Comprehensive error handling and validation for neuro-symbolic AI package."""

import logging
import traceback
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from enum import Enum
import sys
import inspect
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    DATA = "data"
    MODEL = "model"
    INFERENCE = "inference"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    INTEGRATION = "integration"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for errors."""
    function_name: str
    module_name: str
    line_number: int
    timestamp: datetime
    input_parameters: Dict[str, Any]
    system_info: Dict[str, Any]


class NeuroSymbolicError(Exception):
    """Base exception class for neuro-symbolic AI package."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        remediation: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.category = category
        self.context = context
        self.cause = cause
        self.remediation = remediation
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "function": self.context.function_name if self.context else None,
                "module": self.context.module_name if self.context else None,
                "line": self.context.line_number if self.context else None,
            },
            "cause": str(self.cause) if self.cause else None,
            "remediation": self.remediation
        }


class ValidationError(NeuroSymbolicError):
    """Error for input validation failures."""
    
    def __init__(self, message: str, field_name: str = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )
        self.field_name = field_name


class DataError(NeuroSymbolicError):
    """Error for data-related issues."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA,
            **kwargs
        )


class ModelError(NeuroSymbolicError):
    """Error for model-related issues."""
    
    def __init__(self, message: str, model_id: str = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MODEL,
            **kwargs
        )
        self.model_id = model_id


class InferenceError(NeuroSymbolicError):
    """Error during model inference."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INFERENCE,
            **kwargs
        )


class ConfigurationError(NeuroSymbolicError):
    """Error in configuration settings."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )
        self.config_key = config_key


class ResourceError(NeuroSymbolicError):
    """Error related to system resources."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.RESOURCE,
            **kwargs
        )


def create_error_context(frame: inspect.FrameInfo, **kwargs) -> ErrorContext:
    """Create error context from frame information."""
    return ErrorContext(
        function_name=frame.function,
        module_name=frame.filename.split('/')[-1] if frame.filename else 'unknown',
        line_number=frame.lineno,
        timestamp=datetime.now(),
        input_parameters=kwargs.get('input_params', {}),
        system_info={
            'python_version': sys.version,
            'memory_usage': get_memory_usage()
        }
    )


def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage information."""
    try:
        import psutil
        process = psutil.Process()
        return {
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
        }
    except ImportError:
        return {'memory_mb': 'unknown', 'memory_percent': 'unknown'}


def error_handler(
    reraise: bool = True,
    log_level: int = logging.ERROR,
    fallback_return: Any = None
):
    """Decorator for comprehensive error handling."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except NeuroSymbolicError:
                # Re-raise our custom errors
                raise
            except Exception as e:
                # Convert generic exceptions to our error types
                frame = inspect.currentframe().f_back
                frame_info = inspect.getframeinfo(frame)
                
                context = ErrorContext(
                    function_name=func.__name__,
                    module_name=func.__module__,
                    line_number=frame_info.lineno,
                    timestamp=datetime.now(),
                    input_parameters={
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    },
                    system_info={'error_type': type(e).__name__}
                )
                
                # Determine error type based on exception
                if isinstance(e, (ValueError, TypeError)):
                    error_class = ValidationError
                elif isinstance(e, (FileNotFoundError, IOError)):
                    error_class = DataError
                elif isinstance(e, MemoryError):
                    error_class = ResourceError
                else:
                    error_class = NeuroSymbolicError
                
                enhanced_error = error_class(
                    message=f"Error in {func.__name__}: {str(e)}",
                    context=context,
                    cause=e
                )
                
                # Log the error
                logger = logging.getLogger(__name__)
                logger.log(log_level, enhanced_error.to_dict())
                
                if reraise:
                    raise enhanced_error
                else:
                    return fallback_return
                    
        return wrapper
    return decorator


class InputValidator:
    """Comprehensive input validation for neuro-symbolic components."""
    
    @staticmethod
    def validate_array_input(
        data: Any,
        name: str = "data",
        min_dimensions: int = 1,
        max_dimensions: int = 3,
        min_samples: int = 1,
        max_samples: Optional[int] = None,
        min_features: int = 1,
        max_features: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        allow_nan: bool = False,
        allow_inf: bool = False
    ) -> NDArray[np.floating]:
        """Validate array input with comprehensive checks."""
        
        # Type checking
        if not isinstance(data, (np.ndarray, list, tuple)):
            raise ValidationError(
                f"{name} must be numpy array, list, or tuple, got {type(data)}",
                field_name=name,
                remediation="Convert input to numpy array using np.array()"
            )
        
        # Convert to numpy array
        try:
            data_array = np.asarray(data, dtype=dtype or np.float32)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"Cannot convert {name} to numpy array: {e}",
                field_name=name,
                cause=e
            )
        
        # Dimension validation
        if data_array.ndim < min_dimensions:
            raise ValidationError(
                f"{name} must have at least {min_dimensions} dimensions, got {data_array.ndim}",
                field_name=name,
                remediation=f"Reshape data to have {min_dimensions} dimensions"
            )
        
        if data_array.ndim > max_dimensions:
            raise ValidationError(
                f"{name} must have at most {max_dimensions} dimensions, got {data_array.ndim}",
                field_name=name,
                remediation=f"Reshape or flatten data to have {max_dimensions} dimensions"
            )
        
        # Sample count validation
        if data_array.ndim >= 1:
            num_samples = data_array.shape[0]
            if num_samples < min_samples:
                raise ValidationError(
                    f"{name} must have at least {min_samples} samples, got {num_samples}",
                    field_name=name,
                    remediation="Provide more data samples"
                )
            
            if max_samples and num_samples > max_samples:
                raise ValidationError(
                    f"{name} must have at most {max_samples} samples, got {num_samples}",
                    field_name=name,
                    remediation="Reduce number of samples or process in batches"
                )
        
        # Feature count validation
        if data_array.ndim >= 2:
            num_features = data_array.shape[1]
            if num_features < min_features:
                raise ValidationError(
                    f"{name} must have at least {min_features} features, got {num_features}",
                    field_name=name,
                    remediation="Ensure input has sufficient features"
                )
            
            if max_features and num_features > max_features:
                raise ValidationError(
                    f"{name} must have at most {max_features} features, got {num_features}",
                    field_name=name,
                    remediation="Use feature selection or dimensionality reduction"
                )
        
        # NaN validation
        if not allow_nan and np.isnan(data_array).any():
            raise ValidationError(
                f"{name} contains NaN values",
                field_name=name,
                remediation="Remove or impute NaN values before processing"
            )
        
        # Infinity validation
        if not allow_inf and np.isinf(data_array).any():
            raise ValidationError(
                f"{name} contains infinite values",
                field_name=name,
                remediation="Replace infinite values with finite numbers"
            )
        
        # Empty array validation
        if data_array.size == 0:
            raise ValidationError(
                f"{name} cannot be empty",
                field_name=name,
                remediation="Provide non-empty input data"
            )
        
        return data_array
    
    @staticmethod
    def validate_model_id(model_id: str, name: str = "model_id") -> str:
        """Validate model ID."""
        if not isinstance(model_id, str):
            raise ValidationError(
                f"{name} must be a string, got {type(model_id)}",
                field_name=name
            )
        
        if not model_id.strip():
            raise ValidationError(
                f"{name} cannot be empty",
                field_name=name,
                remediation="Provide a valid model identifier"
            )
        
        if len(model_id) > 255:
            raise ValidationError(
                f"{name} must be less than 255 characters, got {len(model_id)}",
                field_name=name,
                remediation="Use a shorter model identifier"
            )
        
        # Check for invalid characters
        invalid_chars = set(model_id) - set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')
        if invalid_chars:
            raise ValidationError(
                f"{name} contains invalid characters: {invalid_chars}",
                field_name=name,
                remediation="Use only alphanumeric characters, underscores, hyphens, and periods"
            )
        
        return model_id.strip()
    
    @staticmethod
    def validate_confidence_score(score: float, name: str = "confidence") -> float:
        """Validate confidence score."""
        if not isinstance(score, (int, float, np.number)):
            raise ValidationError(
                f"{name} must be a number, got {type(score)}",
                field_name=name
            )
        
        score = float(score)
        
        if not (0.0 <= score <= 1.0):
            raise ValidationError(
                f"{name} must be between 0.0 and 1.0, got {score}",
                field_name=name,
                remediation="Normalize confidence score to [0, 1] range"
            )
        
        if np.isnan(score):
            raise ValidationError(
                f"{name} cannot be NaN",
                field_name=name,
                remediation="Provide a valid confidence score"
            )
        
        return score
    
    @staticmethod
    def validate_feature_names(
        feature_names: Optional[List[str]],
        expected_count: Optional[int] = None,
        name: str = "feature_names"
    ) -> Optional[List[str]]:
        """Validate feature names."""
        if feature_names is None:
            return None
        
        if not isinstance(feature_names, (list, tuple)):
            raise ValidationError(
                f"{name} must be a list or tuple, got {type(feature_names)}",
                field_name=name
            )
        
        feature_names = list(feature_names)
        
        if not all(isinstance(name, str) for name in feature_names):
            raise ValidationError(
                f"All {name} must be strings",
                field_name=name,
                remediation="Convert feature names to strings"
            )
        
        if expected_count and len(feature_names) != expected_count:
            raise ValidationError(
                f"{name} must have {expected_count} names, got {len(feature_names)}",
                field_name=name,
                remediation="Provide correct number of feature names"
            )
        
        # Check for duplicates
        if len(feature_names) != len(set(feature_names)):
            duplicates = [name for name in feature_names if feature_names.count(name) > 1]
            raise ValidationError(
                f"{name} contains duplicates: {list(set(duplicates))}",
                field_name=name,
                remediation="Ensure all feature names are unique"
            )
        
        # Check for empty names
        empty_names = [i for i, name in enumerate(feature_names) if not name.strip()]
        if empty_names:
            raise ValidationError(
                f"{name} contains empty names at indices: {empty_names}",
                field_name=name,
                remediation="Provide non-empty feature names"
            )
        
        return [name.strip() for name in feature_names]
    
    @staticmethod
    def validate_hyperparameters(
        params: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
        name: str = "hyperparameters"
    ) -> Dict[str, Any]:
        """Validate hyperparameters against schema."""
        if not isinstance(params, dict):
            raise ValidationError(
                f"{name} must be a dictionary, got {type(params)}",
                field_name=name
            )
        
        validated_params = {}
        
        for key, value in params.items():
            if key not in schema:
                raise ValidationError(
                    f"Unknown hyperparameter: {key}",
                    field_name=f"{name}.{key}",
                    remediation=f"Use one of: {list(schema.keys())}"
                )
            
            param_schema = schema[key]
            
            # Type validation
            expected_type = param_schema.get('type')
            if expected_type and not isinstance(value, expected_type):
                raise ValidationError(
                    f"{key} must be {expected_type.__name__}, got {type(value)}",
                    field_name=f"{name}.{key}"
                )
            
            # Range validation
            min_val = param_schema.get('min')
            if min_val is not None and value < min_val:
                raise ValidationError(
                    f"{key} must be >= {min_val}, got {value}",
                    field_name=f"{name}.{key}"
                )
            
            max_val = param_schema.get('max')
            if max_val is not None and value > max_val:
                raise ValidationError(
                    f"{key} must be <= {max_val}, got {value}",
                    field_name=f"{name}.{key}"
                )
            
            # Choices validation
            choices = param_schema.get('choices')
            if choices and value not in choices:
                raise ValidationError(
                    f"{key} must be one of {choices}, got {value}",
                    field_name=f"{name}.{key}"
                )
            
            validated_params[key] = value
        
        # Check for required parameters
        required_params = [k for k, v in schema.items() if v.get('required', False)]
        missing_params = set(required_params) - set(params.keys())
        if missing_params:
            raise ValidationError(
                f"Missing required hyperparameters: {missing_params}",
                field_name=name,
                remediation=f"Provide values for: {missing_params}"
            )
        
        return validated_params


class ErrorRecovery:
    """Error recovery strategies."""
    
    @staticmethod
    def handle_data_corruption(
        data: NDArray[np.floating],
        strategy: str = "interpolate"
    ) -> NDArray[np.floating]:
        """Handle corrupted data."""
        if strategy == "interpolate":
            # Linear interpolation for NaN values
            if np.isnan(data).any():
                mask = np.isnan(data)
                data_clean = data.copy()
                
                for i in range(data_clean.shape[1]):
                    col = data_clean[:, i]
                    if np.isnan(col).any():
                        valid_indices = ~np.isnan(col)
                        if valid_indices.any():
                            # Simple forward fill + backward fill
                            col_interp = np.interp(
                                np.arange(len(col)),
                                np.arange(len(col))[valid_indices],
                                col[valid_indices]
                            )
                            data_clean[:, i] = col_interp
                
                return data_clean
        
        elif strategy == "remove":
            # Remove rows with any NaN/inf values
            mask = np.isfinite(data).all(axis=1)
            return data[mask]
        
        elif strategy == "zero_fill":
            # Replace NaN/inf with zeros
            data_clean = data.copy()
            data_clean[~np.isfinite(data_clean)] = 0
            return data_clean
        
        return data
    
    @staticmethod
    def handle_memory_error(func: Callable, *args, **kwargs):
        """Handle memory errors by reducing batch size."""
        try:
            return func(*args, **kwargs)
        except MemoryError:
            # Try with reduced batch size if data is available
            if 'data' in kwargs and hasattr(kwargs['data'], 'shape'):
                data = kwargs['data']
                if len(data) > 1:
                    # Split data in half and process separately
                    mid = len(data) // 2
                    kwargs_1 = kwargs.copy()
                    kwargs_1['data'] = data[:mid]
                    kwargs_2 = kwargs.copy() 
                    kwargs_2['data'] = data[mid:]
                    
                    result_1 = func(*args, **kwargs_1)
                    result_2 = func(*args, **kwargs_2)
                    
                    # Combine results (strategy depends on return type)
                    if hasattr(result_1, 'predictions') and hasattr(result_2, 'predictions'):
                        # Combine reasoning results
                        combined_predictions = np.concatenate([result_1.predictions, result_2.predictions])
                        combined_scores = np.concatenate([result_1.confidence_scores, result_2.confidence_scores])
                        # Create combined result (simplified)
                        result_1.predictions = combined_predictions
                        result_1.confidence_scores = combined_scores
                        return result_1
                    
            raise ResourceError(
                "Memory error occurred and automatic recovery failed",
                remediation="Reduce batch size, use smaller model, or increase available memory"
            )


def setup_error_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Setup comprehensive error logging."""
    logger = logging.getLogger('neuro_symbolic')
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler for errors
        try:
            file_handler = logging.FileHandler('neuro_symbolic_errors.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception:
            # Fail silently if file logging not available
            pass
    
    logger.setLevel(log_level)
    return logger