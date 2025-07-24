"""Domain exceptions for machine learning package using standardized framework."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Import from shared infrastructure
sys.path.append(str(Path(__file__).parents[6] / "shared" / "infrastructure"))

try:
    from exceptions.base_exceptions import (
        BaseApplicationError,
        ValidationError,
        ModelOperationError,
        AlgorithmError,
        DataProcessingError,
        ConfigurationError,
        ResourceError,
        ErrorCategory,
        ErrorSeverity,
        create_error_handler
    )
    SHARED_EXCEPTIONS_AVAILABLE = True
except ImportError:
    # Fallback to basic exceptions if shared infrastructure is not available
    SHARED_EXCEPTIONS_AVAILABLE = False
    
    class BaseApplicationError(Exception):
        def __init__(self, message: str, **kwargs):
            super().__init__(message)
            self.message = message
            self.details = kwargs.get('details', {})
            self.recoverable = kwargs.get('recoverable', False)
    
    class ValidationError(BaseApplicationError):
        pass
    
    class ModelOperationError(BaseApplicationError):
        pass
    
    class AlgorithmError(BaseApplicationError):
        pass
    
    class DataProcessingError(BaseApplicationError):
        pass
    
    class ConfigurationError(BaseApplicationError):
        pass
    
    class ResourceError(BaseApplicationError):
        pass


# Legacy base exception for backward compatibility
class DomainError(BaseApplicationError):
    """Base domain exception for machine learning operations.
    
    This class extends the standardized BaseApplicationError to provide
    domain-specific error handling for machine learning operations.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC if SHARED_EXCEPTIONS_AVAILABLE else None,
            operation=operation,
            **kwargs
        )


class AutoMLError(ModelOperationError):
    """AutoML-specific domain exception with enhanced context."""
    
    def __init__(
        self,
        message: str,
        automl_stage: Optional[str] = None,
        model_candidates: Optional[list] = None,
        best_score: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if automl_stage:
            details['automl_stage'] = automl_stage
        if model_candidates:
            details['model_candidates'] = model_candidates
        if best_score is not None:
            details['best_score'] = best_score
            
        super().__init__(
            message=message,
            model_operation="automl",
            details=details,
            **kwargs
        )


class ModelTrainingError(ModelOperationError):
    """Model training exception with detailed context."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        training_stage: Optional[str] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if training_stage:
            details['training_stage'] = training_stage
        if epoch is not None:
            details['epoch'] = epoch
        if loss is not None:
            details['loss'] = loss
        if hyperparameters:
            details['hyperparameters'] = hyperparameters
            
        super().__init__(
            message=message,
            model_name=model_name,
            model_operation="training",
            details=details,
            severity=ErrorSeverity.HIGH if SHARED_EXCEPTIONS_AVAILABLE else None,
            **kwargs
        )


class ModelValidationError(ValidationError):
    """Model validation exception with validation context."""
    
    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        validation_metric: Optional[str] = None,
        expected_value: Any = None,
        actual_value: Any = None,
        threshold: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if validation_type:
            details['validation_type'] = validation_type
        if validation_metric:
            details['validation_metric'] = validation_metric
        if expected_value is not None:
            details['expected_value'] = expected_value
        if actual_value is not None:
            details['actual_value'] = actual_value
        if threshold is not None:
            details['threshold'] = threshold
            
        super().__init__(
            message=message,
            field_name=validation_metric,
            field_value=actual_value,
            validation_rule=validation_type,
            details=details,
            **kwargs
        )


class UncertaintyQuantificationError(AlgorithmError):
    """Uncertainty quantification exception with algorithm context."""
    
    def __init__(
        self,
        message: str,
        uncertainty_method: Optional[str] = None,
        confidence_level: Optional[float] = None,
        sample_size: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if uncertainty_method:
            details['uncertainty_method'] = uncertainty_method
        if confidence_level is not None:
            details['confidence_level'] = confidence_level
        if sample_size is not None:
            details['sample_size'] = sample_size
            
        super().__init__(
            message=message,
            algorithm_name=uncertainty_method,
            details=details,
            **kwargs
        )


class FeatureEngineeringError(DataProcessingError):
    """Feature engineering exception with feature context."""
    
    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        transformation_type: Optional[str] = None,
        feature_count: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if feature_name:
            details['feature_name'] = feature_name
        if transformation_type:
            details['transformation_type'] = transformation_type
        if feature_count is not None:
            details['feature_count'] = feature_count
            
        super().__init__(
            message=message,
            processing_step="feature_engineering",
            details=details,
            **kwargs
        )


class HyperparameterOptimizationError(AlgorithmError):
    """Hyperparameter optimization exception with optimization context."""
    
    def __init__(
        self,
        message: str,
        optimization_method: Optional[str] = None,
        search_space: Optional[Dict[str, Any]] = None,
        iteration: Optional[int] = None,
        best_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if optimization_method:
            details['optimization_method'] = optimization_method
        if search_space:
            details['search_space'] = search_space
        if iteration is not None:
            details['iteration'] = iteration
        if best_params:
            details['best_params'] = best_params
            
        super().__init__(
            message=message,
            algorithm_name=optimization_method,
            details=details,
            **kwargs
        )


class ModelInferenceError(ModelOperationError):
    """Model inference exception with inference context."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if input_shape:
            details['input_shape'] = input_shape
        if batch_size is not None:
            details['batch_size'] = batch_size
            
        super().__init__(
            message=message,
            model_name=model_name,
            model_operation="inference",
            details=details,
            **kwargs
        )


class ModelSerializationError(PersistenceError):
    """Model serialization/deserialization exception."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        serialization_format: Optional[str] = None,
        model_size_mb: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if model_name:
            details['model_name'] = model_name
        if serialization_format:
            details['serialization_format'] = serialization_format
        if model_size_mb is not None:
            details['model_size_mb'] = model_size_mb
            
        super().__init__(
            message=message,
            resource_type="model",
            operation_type="serialization",
            details=details,
            **kwargs
        )


class ModelConfigurationError(ConfigurationError):
    """Model configuration exception with configuration context."""
    
    def __init__(
        self,
        message: str,
        model_type: Optional[str] = None,
        config_section: Optional[str] = None,
        invalid_params: Optional[list] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if model_type:
            details['model_type'] = model_type
        if config_section:
            details['config_section'] = config_section
        if invalid_params:
            details['invalid_params'] = invalid_params
            
        super().__init__(
            message=message,
            config_key=config_section,
            details=details,
            **kwargs
        )


class ComputeResourceError(ResourceError):
    """Compute resource exception for ML operations."""
    
    def __init__(
        self,
        message: str,
        resource_type: str = "compute",
        gpu_available: Optional[bool] = None,
        memory_required_gb: Optional[float] = None,
        memory_available_gb: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if gpu_available is not None:
            details['gpu_available'] = gpu_available
        if memory_required_gb is not None:
            details['memory_required_gb'] = memory_required_gb
        if memory_available_gb is not None:
            details['memory_available_gb'] = memory_available_gb
            
        super().__init__(
            message=message,
            resource_type=resource_type,
            details=details,
            **kwargs
        )


# Create package-specific error handler
if SHARED_EXCEPTIONS_AVAILABLE:
    ml_error_handler = create_error_handler("machine_learning")
else:
    ml_error_handler = None


def handle_ml_error(
    error: Exception,
    operation: str = "unknown",
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
) -> Optional[BaseApplicationError]:
    """Handle machine learning specific errors with context."""
    if ml_error_handler:
        return ml_error_handler.handle_error(error, context, operation, reraise)
    else:
        # Fallback behavior
        if reraise:
            raise error
        return BaseApplicationError(str(error))


# Backward compatibility aliases
MLError = DomainError
TrainingError = ModelTrainingError
ValidationError = ModelValidationError
InferenceError = ModelInferenceError
ConfigError = ModelConfigurationError


__all__ = [
    # New standardized exceptions
    "AutoMLError",
    "ModelTrainingError", 
    "ModelValidationError",
    "UncertaintyQuantificationError",
    "FeatureEngineeringError",
    "HyperparameterOptimizationError", 
    "ModelInferenceError",
    "ModelSerializationError",
    "ModelConfigurationError",
    "ComputeResourceError",
    
    # Legacy compatibility
    "DomainError",
    "MLError",
    "TrainingError",
    "ValidationError", 
    "InferenceError",
    "ConfigError",
    
    # Error handling
    "handle_ml_error",
    "ml_error_handler",
    
    # Flag for feature detection
    "SHARED_EXCEPTIONS_AVAILABLE"
]