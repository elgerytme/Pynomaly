"""Comprehensive input validation utilities for anomaly detection API."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, validator
from enum import Enum
import structlog

from ..logging.error_handler import InputValidationError

logger = structlog.get_logger(__name__)


class ValidationResult:
    """Result of validation operation."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        """Initialize validation result.
        
        Args:
            is_valid: Whether validation passed
            errors: List of validation errors
            warnings: List of validation warnings
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def __bool__(self) -> bool:
        """Boolean conversion returns is_valid."""
        return self.is_valid
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings
        }


class DataValidator:
    """Comprehensive data validation for anomaly detection."""
    
    def __init__(self):
        """Initialize data validator."""
        self.min_samples = 10
        self.max_samples = 100000
        self.min_features = 1
        self.max_features = 1000
        self.max_feature_value = 1e10
        self.min_feature_value = -1e10
    
    def validate_detection_data(
        self, 
        data: Union[List[List[float]], npt.NDArray[np.floating]], 
        algorithm: str = "iforest",
        contamination: float = 0.1
    ) -> ValidationResult:
        """Validate data for anomaly detection.
        
        Args:
            data: Input data for detection
            algorithm: Detection algorithm name
            contamination: Expected contamination rate
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(True)
        
        try:
            # Convert to numpy array if needed
            if isinstance(data, list):
                try:
                    data_array = np.array(data, dtype=np.float64)
                except (ValueError, TypeError) as e:
                    result.add_error(f"Cannot convert data to numeric array: {str(e)}")
                    return result
            else:
                data_array = data
            
            # Basic shape validation
            if data_array.ndim != 2:
                result.add_error(f"Data must be 2-dimensional, got {data_array.ndim} dimensions")
                return result
            
            n_samples, n_features = data_array.shape
            
            # Sample count validation
            if n_samples < self.min_samples:
                result.add_error(f"Too few samples: {n_samples} < {self.min_samples}")
            elif n_samples > self.max_samples:
                result.add_error(f"Too many samples: {n_samples} > {self.max_samples}")
            
            # Feature count validation
            if n_features < self.min_features:
                result.add_error(f"Too few features: {n_features} < {self.min_features}")
            elif n_features > self.max_features:
                result.add_error(f"Too many features: {n_features} > {self.max_features}")
            
            # Data quality checks
            self._validate_data_quality(data_array, result)
            
            # Algorithm-specific validation
            self._validate_algorithm_compatibility(data_array, algorithm, result)
            
            # Contamination validation
            self._validate_contamination(contamination, n_samples, result)
            
            # Performance warnings
            self._add_performance_warnings(n_samples, n_features, result)
            
        except Exception as e:
            result.add_error(f"Validation failed with error: {str(e)}")
        
        return result
    
    def _validate_data_quality(self, data: npt.NDArray[np.floating], result: ValidationResult) -> None:
        """Validate data quality aspects."""
        # Check for infinite values
        if np.isinf(data).any():
            inf_count = np.isinf(data).sum()
            result.add_error(f"Data contains {inf_count} infinite values")
        
        # Check for NaN values
        if np.isnan(data).any():
            nan_count = np.isnan(data).sum()
            result.add_error(f"Data contains {nan_count} NaN values")
        
        # Check for extreme values
        max_val = np.max(data)
        min_val = np.min(data)
        
        if max_val > self.max_feature_value:
            result.add_warning(f"Data contains very large values (max: {max_val:.2e})")
        
        if min_val < self.min_feature_value:
            result.add_warning(f"Data contains very small values (min: {min_val:.2e})")
        
        # Check for constant features
        constant_features = []
        for i in range(data.shape[1]):
            if np.var(data[:, i]) < 1e-12:
                constant_features.append(i)
        
        if constant_features:
            result.add_warning(f"Features with no variance detected: {constant_features}")
        
        # Check for duplicate rows
        unique_rows = np.unique(data, axis=0)
        if len(unique_rows) < data.shape[0]:
            duplicate_count = data.shape[0] - len(unique_rows)
            result.add_warning(f"Data contains {duplicate_count} duplicate rows")
    
    def _validate_algorithm_compatibility(
        self, 
        data: npt.NDArray[np.floating], 
        algorithm: str, 
        result: ValidationResult
    ) -> None:
        """Validate algorithm-specific requirements."""
        n_samples, n_features = data.shape
        
        if algorithm.lower() in ['lof', 'local_outlier_factor']:
            # LOF needs sufficient neighbors
            min_neighbors = 20  # Rule of thumb
            if n_samples < min_neighbors:
                result.add_warning(f"LOF algorithm may not work well with only {n_samples} samples (recommended: >{min_neighbors})")
        
        elif algorithm.lower() in ['ocsvm', 'one_class_svm']:
            # One-Class SVM can be slow with many samples/features
            if n_samples > 10000:
                result.add_warning(f"One-Class SVM may be slow with {n_samples} samples")
            if n_features > 100:
                result.add_warning(f"One-Class SVM may be slow with {n_features} features")
        
        elif algorithm.lower() in ['iforest', 'isolation_forest']:
            # Isolation Forest works well with moderate dimensionality
            if n_features > 50:
                result.add_warning(f"Isolation Forest may be less effective with {n_features} features (recommended: <50)")
    
    def _validate_contamination(self, contamination: float, n_samples: int, result: ValidationResult) -> None:
        """Validate contamination parameter."""
        if not 0.001 <= contamination <= 0.5:
            result.add_error(f"Contamination must be between 0.001 and 0.5, got {contamination}")
        
        # Check if contamination makes sense with sample count
        expected_anomalies = int(contamination * n_samples)
        if expected_anomalies < 1:
            result.add_warning(f"With contamination {contamination} and {n_samples} samples, expected anomalies: {expected_anomalies}")
        
        if contamination > 0.3:
            result.add_warning(f"High contamination rate ({contamination}) may indicate data quality issues")
    
    def _add_performance_warnings(self, n_samples: int, n_features: int, result: ValidationResult) -> None:
        """Add performance-related warnings."""
        # Large dataset warnings
        if n_samples > 50000:
            result.add_warning(f"Large dataset ({n_samples} samples) may require significant processing time")
        
        if n_features > 100:
            result.add_warning(f"High-dimensional data ({n_features} features) may impact performance")
        
        # Memory usage estimation
        estimated_memory_mb = (n_samples * n_features * 8) / (1024 * 1024)  # float64 = 8 bytes
        if estimated_memory_mb > 100:
            result.add_warning(f"Estimated memory usage: {estimated_memory_mb:.1f}MB")


class ParameterValidator:
    """Validator for algorithm parameters."""
    
    def __init__(self):
        """Initialize parameter validator."""
        self.algorithm_params = {
            'iforest': {
                'n_estimators': {'type': int, 'min': 10, 'max': 1000, 'default': 100},
                'max_samples': {'type': (int, float, str), 'min': 1, 'default': 'auto'},
                'contamination': {'type': float, 'min': 0.001, 'max': 0.5, 'default': 0.1},
                'random_state': {'type': int, 'min': 0, 'default': 42},
                'bootstrap': {'type': bool, 'default': False},
                'n_jobs': {'type': int, 'min': -1, 'default': 1}
            },
            'lof': {
                'n_neighbors': {'type': int, 'min': 1, 'max': 1000, 'default': 20},
                'algorithm': {'type': str, 'choices': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'default': 'auto'},
                'leaf_size': {'type': int, 'min': 1, 'max': 100, 'default': 30},
                'metric': {'type': str, 'choices': ['minkowski', 'euclidean', 'manhattan'], 'default': 'minkowski'},
                'p': {'type': int, 'min': 1, 'max': 10, 'default': 2},
                'contamination': {'type': float, 'min': 0.001, 'max': 0.5, 'default': 0.1},
                'n_jobs': {'type': int, 'min': -1, 'default': 1}
            },
            'ocsvm': {
                'kernel': {'type': str, 'choices': ['linear', 'poly', 'rbf', 'sigmoid'], 'default': 'rbf'},
                'degree': {'type': int, 'min': 1, 'max': 10, 'default': 3},
                'gamma': {'type': (float, str), 'min': 0.0001, 'choices': ['scale', 'auto'], 'default': 'scale'},
                'coef0': {'type': float, 'default': 0.0},
                'tol': {'type': float, 'min': 1e-6, 'max': 1e-2, 'default': 1e-3},
                'nu': {'type': float, 'min': 0.001, 'max': 1.0, 'default': 0.5},
                'shrinking': {'type': bool, 'default': True},
                'cache_size': {'type': float, 'min': 1, 'max': 10000, 'default': 200},
                'max_iter': {'type': int, 'min': -1, 'default': -1}
            }
        }
    
    def validate_parameters(self, algorithm: str, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate algorithm parameters.
        
        Args:
            algorithm: Algorithm name
            parameters: Dictionary of parameters to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(True)
        
        # Check if algorithm is supported
        if algorithm not in self.algorithm_params:
            result.add_error(f"Unsupported algorithm: {algorithm}")
            return result
        
        param_specs = self.algorithm_params[algorithm]
        
        # Validate each parameter
        for param_name, param_value in parameters.items():
            if param_name not in param_specs:
                result.add_warning(f"Unknown parameter '{param_name}' for algorithm '{algorithm}'")
                continue
            
            spec = param_specs[param_name]
            validation_error = self._validate_single_parameter(param_name, param_value, spec)
            
            if validation_error:
                result.add_error(validation_error)
        
        # Check for missing required parameters and add defaults
        self._add_default_warnings(algorithm, parameters, param_specs, result)
        
        return result
    
    def _validate_single_parameter(self, name: str, value: Any, spec: Dict[str, Any]) -> Optional[str]:
        """Validate a single parameter against its specification."""
        # Type validation
        expected_type = spec['type']
        if isinstance(expected_type, tuple):
            if not any(isinstance(value, t) for t in expected_type):
                return f"Parameter '{name}' must be one of types {expected_type}, got {type(value)}"
        else:
            if not isinstance(value, expected_type):
                return f"Parameter '{name}' must be of type {expected_type.__name__}, got {type(value)}"
        
        # Choice validation
        if 'choices' in spec:
            if isinstance(value, str) and value not in spec['choices']:
                return f"Parameter '{name}' must be one of {spec['choices']}, got '{value}'"
        
        # Range validation for numeric types
        if isinstance(value, (int, float)):
            if 'min' in spec and value < spec['min']:
                return f"Parameter '{name}' must be >= {spec['min']}, got {value}"
            if 'max' in spec and value > spec['max']:
                return f"Parameter '{name}' must be <= {spec['max']}, got {value}"
        
        return None
    
    def _add_default_warnings(
        self, 
        algorithm: str, 
        parameters: Dict[str, Any], 
        param_specs: Dict[str, Dict[str, Any]], 
        result: ValidationResult
    ) -> None:
        """Add warnings for parameters using defaults."""
        for param_name, spec in param_specs.items():
            if param_name not in parameters and 'default' in spec:
                result.add_warning(f"Parameter '{param_name}' not specified, using default: {spec['default']}")


class EnsembleValidator:
    """Validator for ensemble detection parameters."""
    
    def __init__(self):
        """Initialize ensemble validator."""
        self.supported_methods = [
            'majority', 'average', 'weighted_average', 'max', 'min', 'median'
        ]
        self.min_algorithms = 2
        self.max_algorithms = 10
    
    def validate_ensemble_request(
        self, 
        algorithms: List[str], 
        method: str, 
        weights: Optional[List[float]] = None
    ) -> ValidationResult:
        """Validate ensemble detection request.
        
        Args:
            algorithms: List of algorithms to use
            method: Ensemble combination method
            weights: Optional weights for weighted methods
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(True)
        
        # Validate algorithm count
        if len(algorithms) < self.min_algorithms:
            result.add_error(f"Ensemble requires at least {self.min_algorithms} algorithms, got {len(algorithms)}")
        
        if len(algorithms) > self.max_algorithms:
            result.add_error(f"Too many algorithms: {len(algorithms)} > {self.max_algorithms}")
        
        # Check for duplicate algorithms
        if len(set(algorithms)) != len(algorithms):
            duplicates = [alg for alg in set(algorithms) if algorithms.count(alg) > 1]
            result.add_warning(f"Duplicate algorithms detected: {duplicates}")
        
        # Validate method
        if method not in self.supported_methods:
            result.add_error(f"Unsupported ensemble method: {method}. Supported: {self.supported_methods}")
        
        # Validate weights for weighted methods
        if method == 'weighted_average':
            if weights is None:
                result.add_error("Weights required for weighted_average method")
            elif len(weights) != len(algorithms):
                result.add_error(f"Number of weights ({len(weights)}) must match number of algorithms ({len(algorithms)})")
            elif any(w <= 0 for w in weights):
                result.add_error("All weights must be positive")
            elif abs(sum(weights) - 1.0) > 1e-6:
                result.add_warning(f"Weights sum to {sum(weights):.6f}, normalizing to 1.0")
        
        return result


class ModelValidator:
    """Validator for model-related operations."""
    
    def __init__(self):
        """Initialize model validator."""
        self.valid_model_id_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
        self.max_model_id_length = 100
        self.max_metadata_size = 10000  # bytes
    
    def validate_model_id(self, model_id: str) -> ValidationResult:
        """Validate model ID format.
        
        Args:
            model_id: Model identifier to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(True)
        
        if not model_id:
            result.add_error("Model ID cannot be empty")
            return result
        
        if len(model_id) > self.max_model_id_length:
            result.add_error(f"Model ID too long: {len(model_id)} > {self.max_model_id_length}")
        
        if not self.valid_model_id_pattern.match(model_id):
            result.add_error("Model ID can only contain letters, numbers, underscores, and hyphens")
        
        return result
    
    def validate_model_metadata(self, metadata: Dict[str, Any]) -> ValidationResult:
        """Validate model metadata.
        
        Args:
            metadata: Model metadata dictionary
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(True)
        
        # Check metadata size
        import json
        try:
            metadata_json = json.dumps(metadata)
            size_bytes = len(metadata_json.encode('utf-8'))
            
            if size_bytes > self.max_metadata_size:
                result.add_error(f"Metadata too large: {size_bytes} > {self.max_metadata_size} bytes")
        except (TypeError, ValueError) as e:
            result.add_error(f"Metadata is not JSON serializable: {str(e)}")
        
        # Check required fields
        required_fields = ['algorithm', 'created_at', 'version']
        for field in required_fields:
            if field not in metadata:
                result.add_warning(f"Recommended metadata field missing: {field}")
        
        # Validate timestamp fields
        timestamp_fields = ['created_at', 'updated_at', 'last_used_at']
        for field in timestamp_fields:
            if field in metadata:
                try:
                    if isinstance(metadata[field], str):
                        datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                except ValueError:
                    result.add_error(f"Invalid timestamp format for field '{field}': {metadata[field]}")
        
        return result


# Comprehensive validator factory
class ComprehensiveValidator:
    """Factory class for all validators."""
    
    def __init__(self):
        """Initialize comprehensive validator."""
        self.data_validator = DataValidator()
        self.parameter_validator = ParameterValidator()
        self.ensemble_validator = EnsembleValidator()
        self.model_validator = ModelValidator()
    
    def validate_detection_request(
        self, 
        data: Union[List[List[float]], npt.NDArray[np.floating]], 
        algorithm: str,
        contamination: float,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Comprehensive validation for detection requests."""
        combined_result = ValidationResult(True)
        
        # Data validation
        data_result = self.data_validator.validate_detection_data(data, algorithm, contamination)
        combined_result.errors.extend(data_result.errors)
        combined_result.warnings.extend(data_result.warnings)
        
        # Parameter validation
        if parameters:
            param_result = self.parameter_validator.validate_parameters(algorithm, parameters)
            combined_result.errors.extend(param_result.errors)
            combined_result.warnings.extend(param_result.warnings)
        
        # Update validity
        combined_result.is_valid = combined_result.is_valid and data_result.is_valid and (not parameters or param_result.is_valid)
        
        return combined_result
    
    def validate_ensemble_request(
        self,
        data: Union[List[List[float]], npt.NDArray[np.floating]],
        algorithms: List[str],
        method: str,
        contamination: float,
        weights: Optional[List[float]] = None,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> ValidationResult:
        """Comprehensive validation for ensemble requests."""
        combined_result = ValidationResult(True)
        
        # Data validation (using first algorithm for compatibility check)
        data_result = self.data_validator.validate_detection_data(data, algorithms[0] if algorithms else 'iforest', contamination)
        combined_result.errors.extend(data_result.errors)
        combined_result.warnings.extend(data_result.warnings)
        
        # Ensemble validation
        ensemble_result = self.ensemble_validator.validate_ensemble_request(algorithms, method, weights)
        combined_result.errors.extend(ensemble_result.errors)
        combined_result.warnings.extend(ensemble_result.warnings)
        
        # Parameter validation for each algorithm
        if parameters:
            for algorithm, params in parameters.items():
                if algorithm in algorithms:
                    param_result = self.parameter_validator.validate_parameters(algorithm, params)
                    combined_result.errors.extend([f"Algorithm '{algorithm}': {error}" for error in param_result.errors])
                    combined_result.warnings.extend([f"Algorithm '{algorithm}': {warning}" for warning in param_result.warnings])
        
        # Update validity
        param_errors = []
        if parameters:
            param_errors = [
                self.parameter_validator.validate_parameters(alg, parameters.get(alg, {}))
                for alg in algorithms
            ]
        
        combined_result.is_valid = all([
            data_result.is_valid,
            ensemble_result.is_valid,
            not any(param_result.errors for param_result in param_errors)
        ])
        
        return combined_result