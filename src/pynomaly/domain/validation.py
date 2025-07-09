"""Enhanced domain validation system for PyNomaly."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ValidationSeverity(Enum):
    """Severity levels for validation errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationError(Exception):
    """Enhanced validation error with severity and context."""
    
    def __init__(self, message: str, field: str = None, severity: ValidationSeverity = ValidationSeverity.ERROR, **kwargs):
        super().__init__(message)
        self.message = message
        self.field = field
        self.severity = severity
        self.context = kwargs


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool = True):
        self.is_valid = is_valid
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.timestamp = datetime.utcnow()
    
    def add_error(self, message: str, field: str = None, **kwargs):
        """Add an error to the validation result."""
        self.errors.append({
            "message": message,
            "field": field,
            "severity": ValidationSeverity.ERROR.value,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        })
        self.is_valid = False
    
    def add_warning(self, message: str, field: str = None, **kwargs):
        """Add a warning to the validation result."""
        self.warnings.append({
            "message": message,
            "field": field,
            "severity": ValidationSeverity.WARNING.value,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        })
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


class DomainValidator:
    """Enhanced domain validation utilities."""
    
    @staticmethod
    def validate_anomaly_score(score: Union[float, int]) -> ValidationResult:
        """Validate anomaly score value."""
        result = ValidationResult()
        
        if not isinstance(score, (int, float)):
            result.add_error(f"Score must be numeric, got {type(score)}", field="score")
            return result
        
        if math.isnan(score) or math.isinf(score):
            result.add_error("Score cannot be NaN or infinity", field="score")
            return result
        
        if score < 0.0 or score > 1.0:
            result.add_error(f"Score must be between 0.0 and 1.0, got {score}", field="score")
            return result
        
        # Warning for edge cases
        if score > 0.8:
            result.add_warning("High anomaly score detected", field="score", score=score)
        
        return result
    
    @staticmethod
    def validate_detector_name(name: str) -> ValidationResult:
        """Validate detector name format."""
        result = ValidationResult()
        
        if not isinstance(name, str):
            result.add_error(f"Detector name must be string, got {type(name)}", field="detector_name")
            return result
        
        if not name.strip():
            result.add_error("Detector name cannot be empty", field="detector_name")
            return result
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9_.-]+$', name.strip()):
            result.add_error(
                "Detector name can only contain alphanumeric characters, underscores, hyphens, and dots",
                field="detector_name"
            )
        
        return result
    
    @staticmethod
    def validate_data_point(data_point: Dict[str, Any]) -> ValidationResult:
        """Validate data point structure."""
        result = ValidationResult()
        
        if not isinstance(data_point, dict):
            result.add_error(f"Data point must be dictionary, got {type(data_point)}", field="data_point")
            return result
        
        if not data_point:
            result.add_error("Data point cannot be empty", field="data_point")
            return result
        
        # Check for reserved keys
        reserved_keys = {"_id", "_score", "_detector", "_timestamp"}
        if any(key in reserved_keys for key in data_point.keys()):
            result.add_error(f"Data point cannot contain reserved keys: {reserved_keys}", field="data_point")
        
        # Validate JSON serializable
        try:
            json.dumps(data_point)
        except (TypeError, ValueError) as e:
            result.add_error(f"Data point must be JSON serializable: {e}", field="data_point")
        
        return result
    
    @staticmethod
    def validate_contamination_rate(rate: Union[float, int]) -> ValidationResult:
        """Validate contamination rate value."""
        result = ValidationResult()
        
        if not isinstance(rate, (int, float)):
            result.add_error(f"Contamination rate must be numeric, got {type(rate)}", field="contamination_rate")
            return result
        
        if math.isnan(rate) or math.isinf(rate):
            result.add_error("Contamination rate cannot be NaN or infinity", field="contamination_rate")
            return result
        
        if rate < 0.0 or rate > 0.5:
            result.add_error(f"Contamination rate must be between 0.0 and 0.5, got {rate}", field="contamination_rate")
            return result
        
        # Warning for very high rates
        if rate > 0.3:
            result.add_warning("Very high contamination rate detected", field="contamination_rate", rate=rate)
        
        return result
    
    @staticmethod
    def validate_dataset_name(name: str) -> ValidationResult:
        """Validate dataset name format."""
        result = ValidationResult()
        
        if not isinstance(name, str):
            result.add_error(f"Dataset name must be string, got {type(name)}", field="name")
            return result
        
        if not name.strip():
            result.add_error("Dataset name cannot be empty", field="name")
            return result
        
        if len(name) > 255:
            result.add_error(f"Dataset name cannot exceed 255 characters, got {len(name)}", field="name")
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9_.-]+$', name.strip()):
            result.add_error(
                "Dataset name can only contain alphanumeric characters, underscores, hyphens, and dots",
                field="name"
            )
        
        return result
    
    @staticmethod
    def validate_timestamp(timestamp: datetime) -> ValidationResult:
        """Validate timestamp is reasonable."""
        result = ValidationResult()
        
        if not isinstance(timestamp, datetime):
            result.add_error(f"Timestamp must be datetime, got {type(timestamp)}", field="timestamp")
            return result
        
        now = datetime.utcnow()
        if timestamp > now + timedelta(seconds=60):  # 1 minute tolerance
            result.add_error("Timestamp cannot be in the future", field="timestamp")
        
        if timestamp < datetime(2000, 1, 1):
            result.add_error("Timestamp cannot be before year 2000", field="timestamp")
        
        return result
    
    @staticmethod
    def validate_explanation(explanation: Optional[str]) -> ValidationResult:
        """Validate explanation content."""
        result = ValidationResult()
        
        if explanation is None:
            return result
        
        if not isinstance(explanation, str):
            result.add_error(f"Explanation must be string, got {type(explanation)}", field="explanation")
            return result
        
        if len(explanation) > 2048:
            result.add_error(f"Explanation cannot exceed 2048 characters, got {len(explanation)}", field="explanation")
        
        return result


class ValidationStrategy:
    """Base validation strategy."""
    
    def __init__(self, strict: bool = True):
        self.strict = strict
    
    def validate_anomaly(self, score: float, data_point: Dict[str, Any], detector_name: str, 
                        explanation: Optional[str] = None) -> ValidationResult:
        """Validate anomaly data."""
        result = ValidationResult()
        
        # Validate individual fields
        score_result = DomainValidator.validate_anomaly_score(score)
        result.errors.extend(score_result.errors)
        result.warnings.extend(score_result.warnings)
        
        data_result = DomainValidator.validate_data_point(data_point)
        result.errors.extend(data_result.errors)
        result.warnings.extend(data_result.warnings)
        
        detector_result = DomainValidator.validate_detector_name(detector_name)
        result.errors.extend(detector_result.errors)
        result.warnings.extend(detector_result.warnings)
        
        if explanation:
            exp_result = DomainValidator.validate_explanation(explanation)
            result.errors.extend(exp_result.errors)
            result.warnings.extend(exp_result.warnings)
        
        # Business rule validation
        if score > 0.8 and not explanation:
            if self.strict:
                result.add_error("High-confidence anomaly should have explanation", field="explanation")
            else:
                result.add_warning("High-confidence anomaly should have explanation", field="explanation")
        
        result.is_valid = not result.has_errors()
        return result
    
    def validate_dataset(self, name: str, data: Any, target_column: Optional[str] = None) -> ValidationResult:
        """Validate dataset data."""
        result = ValidationResult()
        
        # Validate name
        name_result = DomainValidator.validate_dataset_name(name)
        result.errors.extend(name_result.errors)
        result.warnings.extend(name_result.warnings)
        
        # Basic data validation
        if hasattr(data, 'empty') and data.empty:
            result.add_error("Dataset cannot be empty", field="data")
        
        if hasattr(data, 'shape') and data.shape[0] < 2:
            result.add_error("Dataset must have at least 2 rows", field="data")
        
        # Target column validation
        if target_column and hasattr(data, 'columns') and target_column not in data.columns:
            result.add_error(f"Target column '{target_column}' not found in dataset", field="target_column")
        
        result.is_valid = not result.has_errors()
        return result


# Global validation instance
_default_validator = ValidationStrategy(strict=True)


def validate_anomaly(score: float, data_point: Dict[str, Any], detector_name: str, 
                    explanation: Optional[str] = None) -> ValidationResult:
    """Validate anomaly using default strategy."""
    return _default_validator.validate_anomaly(score, data_point, detector_name, explanation)


def validate_dataset(name: str, data: Any, target_column: Optional[str] = None) -> ValidationResult:
    """Validate dataset using default strategy."""
    return _default_validator.validate_dataset(name, data, target_column)


def set_validation_strategy(strategy: ValidationStrategy):
    """Set the global validation strategy."""
    global _default_validator
    _default_validator = strategy
