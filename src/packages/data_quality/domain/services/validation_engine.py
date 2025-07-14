"""Comprehensive data validation engine with advanced rule processing capabilities."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Callable, Optional, Set, Union, Tuple
import asyncio
import re
import logging
import time
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    """Validation rule categories."""
    DATA_TYPE = "data_type"
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    CONSISTENCY = "consistency"
    BUSINESS_RULE = "business_rule"
    STATISTICAL = "statistical"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    FORMAT = "format"
    RANGE = "range"
    CUSTOM = "custom"


class ValidationContext(BaseModel):
    """Validation execution context."""
    dataset_name: str
    dataset_id: Optional[UUID] = None
    execution_id: UUID = Field(default_factory=uuid4)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ValidationError(BaseModel):
    """Individual validation error with detailed context."""
    error_id: UUID = Field(default_factory=uuid4)
    rule_id: str
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    invalid_value: Any = None
    expected_value: Any = None
    error_message: str
    error_code: Optional[str] = None
    severity: ValidationSeverity
    category: ValidationCategory
    context: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True


class ValidationMetrics(BaseModel):
    """Validation execution metrics."""
    total_records: int
    records_processed: int
    records_passed: int
    records_failed: int
    pass_rate: float
    execution_time_ms: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    @validator('pass_rate')
    def validate_pass_rate(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Pass rate must be between 0.0 and 1.0')
        return v


class ValidationResult(BaseModel):
    """Comprehensive validation result."""
    validation_id: UUID = Field(default_factory=uuid4)
    rule_id: str
    rule_name: str
    category: ValidationCategory
    severity: ValidationSeverity
    passed: bool
    
    # Metrics
    metrics: ValidationMetrics
    
    # Error details
    errors: List[ValidationError] = Field(default_factory=list)
    
    # Execution context
    context: ValidationContext
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Statistical information
    statistics: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True

class ValidationRule(ABC):
    """Abstract base class for validation rules with advanced capabilities."""
    
    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str = '',
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        category: ValidationCategory = ValidationCategory.CUSTOM,
        enabled: bool = True,
        fail_fast: bool = False,
        sample_errors: int = 100
    ):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.severity = severity
        self.category = category
        self.enabled = enabled
        self.fail_fast = fail_fast
        self.sample_errors = sample_errors
        self._errors: List[ValidationError] = []
    
    @abstractmethod
    def validate_record(self, record: Dict[str, Any], row_index: int) -> bool:
        """Validate a single record."""
        pass
    
    @abstractmethod
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Validate entire dataset (for rules that need full context)."""
        pass
    
    def get_applicable_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns this rule applies to."""
        return list(df.columns)
    
    def can_run_parallel(self) -> bool:
        """Whether this rule can be executed in parallel."""
        return True
    
    def reset(self) -> None:
        """Reset rule state for new validation run."""
        self._errors.clear()
    
    def add_error(
        self,
        row_index: Optional[int],
        column_name: Optional[str],
        invalid_value: Any,
        message: str,
        error_code: Optional[str] = None,
        expected_value: Any = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add validation error."""
        if len(self._errors) >= self.sample_errors:
            return
            
        error = ValidationError(
            rule_id=self.rule_id,
            row_index=row_index,
            column_name=column_name,
            invalid_value=invalid_value,
            expected_value=expected_value,
            error_message=message,
            error_code=error_code,
            severity=self.severity,
            category=self.category,
            context=context or {}
        )
        self._errors.append(error)
    
    def get_errors(self) -> List[ValidationError]:
        """Get collected validation errors."""
        return self._errors.copy()

class RangeRule(ValidationRule):
    """Validate that a numeric field falls within a specified range."""
    def __init__(self, rule_id: str, field: str, min_value: float = None, max_value: float = None, description: str = '', severity: str = 'medium'):
        super().__init__(rule_id, description, severity)
        self.field = field
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, record: Dict[str, Any]) -> bool:
        value = record.get(self.field)
        if value is None:
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        if self.min_value is not None and v < self.min_value:
            return False
        if self.max_value is not None and v > self.max_value:
            return False
        return True

class FormatRule(ValidationRule):
    """Validate that a string field matches a regular expression."""
    def __init__(self, rule_id: str, field: str, pattern: str, description: str = '', severity: str = 'medium'):
        super().__init__(rule_id, description, severity)
        self.field = field
        self.pattern = re.compile(pattern)

    def validate(self, record: Dict[str, Any]) -> bool:
        value = record.get(self.field)
        if value is None:
            return False
        return bool(self.pattern.match(str(value)))

class CompletenessRule(ValidationRule):
    """Validate that a field is not null or empty."""
    def __init__(self, rule_id: str, field: str, description: str = '', severity: str = 'medium'):
        super().__init__(rule_id, description, severity)
        self.field = field

    def validate(self, record: Dict[str, Any]) -> bool:
        value = record.get(self.field)
        return value is not None and value != ''

class UniquenessRule(ValidationRule):
    """Validate that field values are unique across records."""
    def __init__(self, rule_id: str, field: str, description: str = '', severity: str = 'medium'):
        super().__init__(rule_id, description, severity)
        self.field = field
        self._seen = set()

    def validate(self, record: Dict[str, Any]) -> bool:
        value = record.get(self.field)
        if value in self._seen:
            return False
        self._seen.add(value)
        return True

class ConsistencyRule(ValidationRule):
    """Validate consistency between two fields via a comparator function."""
    def __init__(self, rule_id: str, field_a: str, field_b: str, comparator: Callable[[Any, Any], bool], description: str = '', severity: str = 'medium'):
        super().__init__(rule_id, description, severity)
        self.field_a = field_a
        self.field_b = field_b
        self.comparator = comparator

    def validate(self, record: Dict[str, Any]) -> bool:
        a = record.get(self.field_a)
        b = record.get(self.field_b)
        if a is None or b is None:
            return False
        try:
            return bool(self.comparator(a, b))
        except Exception:
            return False

class ValidationEngine:
    """Engine to apply multiple validation rules to a dataset."""
    def __init__(self, rules: List[ValidationRule]):
        self.rules = rules

    def run(self, records: List[Dict[str, Any]]) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        for rule in self.rules:
            failed_details: List[Dict[str, Any]] = []
            if isinstance(rule, UniquenessRule):
                rule._seen.clear()
            for record in records:
                if not rule.validate(record):
                    failed_details.append(record)
            results.append(ValidationResult(
                rule_id=rule.rule_id,
                passed=(len(failed_details) == 0),
                failed_records=len(failed_details),
                error_details=failed_details
            ))
        return results