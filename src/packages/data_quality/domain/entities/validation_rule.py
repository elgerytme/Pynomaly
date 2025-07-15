"""Validation Rule Domain Entities.

Contains entities for quality rules, validation logic, and validation results.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable
from enum import Enum
import uuid


# Value Objects and Enums
@dataclass(frozen=True)
class RuleId:
    """Rule identifier value object."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ValidationId:
    """Validation identifier value object."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class UserId:
    """User identifier value object."""
    value: str
    
    def __str__(self) -> str:
        return self.value


class RuleType(Enum):
    """Types of validation rules."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    BUSINESS_LOGIC = "business_logic"
    FORMAT = "format"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"


class LogicType(Enum):
    """Types of validation logic."""
    SQL = "sql"
    PYTHON = "python"
    REGEX = "regex"
    STATISTICAL = "statistical"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    LOOKUP = "lookup"
    CONDITIONAL = "conditional"
    EXPRESSION = "expression"


class ValidationStatus(Enum):
    """Status of validation execution."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class Severity(Enum):
    """Severity levels for validation rules."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class QualityCategory(Enum):
    """Categories of quality rules."""
    DATA_INTEGRITY = "data_integrity"
    BUSINESS_RULES = "business_rules"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    PERFORMANCE = "performance"
    SECURITY = "security"
    METADATA = "metadata"
    LINEAGE = "lineage"


@dataclass(frozen=True)
class SuccessCriteria:
    """Success criteria for validation rules."""
    min_pass_rate: float = 1.0  # Minimum pass rate (0.0 to 1.0)
    max_failure_count: Optional[int] = None  # Maximum number of failures allowed
    warning_threshold: float = 0.95  # Threshold for warnings
    
    def __post_init__(self):
        """Validate success criteria."""
        if not (0.0 <= self.min_pass_rate <= 1.0):
            raise ValueError("Min pass rate must be between 0 and 1")
        if not (0.0 <= self.warning_threshold <= 1.0):
            raise ValueError("Warning threshold must be between 0 and 1")
        if self.max_failure_count is not None and self.max_failure_count < 0:
            raise ValueError("Max failure count must be non-negative")


@dataclass(frozen=True)
class ValidationLogic:
    """Validation logic specification."""
    logic_type: LogicType
    expression: str
    parameters: Dict[str, Any]
    error_message: str
    success_criteria: SuccessCriteria
    
    # Optional metadata
    description: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate logic specification."""
        if not self.expression.strip():
            raise ValueError("Expression cannot be empty")
        if not self.error_message.strip():
            raise ValueError("Error message cannot be empty")
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get parameter value with optional default."""
        return self.parameters.get(key, default)
    
    def has_parameter(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in self.parameters
    
    def with_parameter(self, key: str, value: Any) -> 'ValidationLogic':
        """Create new logic with additional parameter."""
        new_params = self.parameters.copy()
        new_params[key] = value
        return ValidationLogic(
            logic_type=self.logic_type,
            expression=self.expression,
            parameters=new_params,
            error_message=self.error_message,
            success_criteria=self.success_criteria,
            description=self.description,
            examples=self.examples,
            dependencies=self.dependencies
        )


@dataclass(frozen=True)
class ValidationError:
    """Individual validation error details."""
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    field_value: Optional[str] = None
    error_message: str = ""
    error_code: Optional[str] = None
    severity: Severity = Severity.MEDIUM
    
    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None
    
    def __post_init__(self):
        """Validate error details."""
        if not self.error_message.strip():
            raise ValueError("Error message cannot be empty")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of validation error."""
        return {
            'row_index': self.row_index,
            'column_name': self.column_name,
            'field_value': self.field_value,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'severity': self.severity.value,
            'has_suggested_fix': bool(self.suggested_fix),
            'context_keys': list(self.context.keys()) if self.context else []
        }


# Main Domain Entities
@dataclass(frozen=True)
class QualityRule:
    """Quality validation rule entity."""
    
    rule_id: RuleId
    rule_name: str
    rule_type: RuleType
    description: str
    validation_logic: ValidationLogic
    severity: Severity
    category: QualityCategory
    is_active: bool
    created_by: UserId
    created_at: datetime
    last_modified: datetime
    
    # Optional metadata
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    business_justification: Optional[str] = None
    
    # Scope and targeting
    target_tables: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)
    target_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Execution settings
    execution_timeout_seconds: int = 300
    execution_mode: str = "batch"  # "batch", "streaming", "scheduled"
    execution_schedule: Optional[str] = None
    
    # Performance and monitoring
    average_execution_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    execution_count: int = 0
    success_rate: float = 0.0
    
    def __post_init__(self):
        """Validate rule entity."""
        if not self.rule_name.strip():
            raise ValueError("Rule name cannot be empty")
        if not self.description.strip():
            raise ValueError("Rule description cannot be empty")
        if self.last_modified < self.created_at:
            raise ValueError("Last modified cannot be before created date")
        if self.execution_timeout_seconds <= 0:
            raise ValueError("Execution timeout must be positive")
    
    def is_applicable_to_table(self, table_name: str) -> bool:
        """Check if rule applies to specific table."""
        return not self.target_tables or table_name in self.target_tables
    
    def is_applicable_to_column(self, column_name: str) -> bool:
        """Check if rule applies to specific column."""
        return not self.target_columns or column_name in self.target_columns
    
    def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of rule configuration."""
        return {
            'rule_id': str(self.rule_id),
            'rule_name': self.rule_name,
            'rule_type': self.rule_type.value,
            'severity': self.severity.value,
            'category': self.category.value,
            'is_active': self.is_active,
            'version': self.version,
            'target_tables': self.target_tables,
            'target_columns': self.target_columns,
            'execution_mode': self.execution_mode,
            'success_rate': self.success_rate,
            'execution_count': self.execution_count,
            'last_execution': self.last_execution_time.isoformat() if self.last_execution_time else None,
            'created_at': self.created_at.isoformat()
        }
    
    def update_execution_stats(self, execution_time: float, success: bool) -> 'QualityRule':
        """Update rule execution statistics."""
        new_count = self.execution_count + 1
        new_avg_time = ((self.average_execution_time * self.execution_count) + execution_time) / new_count
        new_success_rate = ((self.success_rate * self.execution_count) + (1.0 if success else 0.0)) / new_count
        
        return QualityRule(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            rule_type=self.rule_type,
            description=self.description,
            validation_logic=self.validation_logic,
            severity=self.severity,
            category=self.category,
            is_active=self.is_active,
            created_by=self.created_by,
            created_at=self.created_at,
            last_modified=self.last_modified,
            version=self.version,
            tags=self.tags,
            business_justification=self.business_justification,
            target_tables=self.target_tables,
            target_columns=self.target_columns,
            target_conditions=self.target_conditions,
            execution_timeout_seconds=self.execution_timeout_seconds,
            execution_mode=self.execution_mode,
            execution_schedule=self.execution_schedule,
            average_execution_time=new_avg_time,
            last_execution_time=datetime.now(),
            execution_count=new_count,
            success_rate=new_success_rate
        )
    
    def deactivate(self) -> 'QualityRule':
        """Create deactivated version of rule."""
        return QualityRule(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            rule_type=self.rule_type,
            description=self.description,
            validation_logic=self.validation_logic,
            severity=self.severity,
            category=self.category,
            is_active=False,
            created_by=self.created_by,
            created_at=self.created_at,
            last_modified=datetime.now(),
            version=self.version,
            tags=self.tags,
            business_justification=self.business_justification,
            target_tables=self.target_tables,
            target_columns=self.target_columns,
            target_conditions=self.target_conditions,
            execution_timeout_seconds=self.execution_timeout_seconds,
            execution_mode=self.execution_mode,
            execution_schedule=self.execution_schedule,
            average_execution_time=self.average_execution_time,
            last_execution_time=self.last_execution_time,
            execution_count=self.execution_count,
            success_rate=self.success_rate
        )


@dataclass(frozen=True)
class ValidationResult:
    """Result of validation rule execution."""
    
    validation_id: ValidationId
    rule_id: RuleId
    dataset_id: 'DatasetId'
    status: ValidationStatus
    passed_records: int
    failed_records: int
    failure_rate: float
    error_details: List[ValidationError]
    execution_time: timedelta
    validated_at: datetime
    
    # Additional metrics
    warning_count: int = 0
    skipped_count: int = 0
    total_records: int = 0
    
    # Execution context
    execution_context: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result entity."""
        if self.passed_records < 0:
            raise ValueError("Passed records cannot be negative")
        if self.failed_records < 0:
            raise ValueError("Failed records cannot be negative")
        if not (0.0 <= self.failure_rate <= 1.0):
            raise ValueError("Failure rate must be between 0 and 1")
        if self.execution_time.total_seconds() < 0:
            raise ValueError("Execution time cannot be negative")
        
        # Calculate total records if not provided
        if self.total_records == 0:
            calculated_total = self.passed_records + self.failed_records + self.warning_count + self.skipped_count
            object.__setattr__(self, 'total_records', calculated_total)
    
    def get_success_rate(self) -> float:
        """Get validation success rate."""
        return 1.0 - self.failure_rate
    
    def is_successful(self) -> bool:
        """Check if validation was successful."""
        return self.status == ValidationStatus.PASSED
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return self.warning_count > 0
    
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.error_details) > 0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of validation errors."""
        error_by_severity = {}
        error_by_column = {}
        
        for error in self.error_details:
            # Group by severity
            severity = error.severity.value
            if severity not in error_by_severity:
                error_by_severity[severity] = 0
            error_by_severity[severity] += 1
            
            # Group by column
            if error.column_name:
                if error.column_name not in error_by_column:
                    error_by_column[error.column_name] = 0
                error_by_column[error.column_name] += 1
        
        return {
            'total_errors': len(self.error_details),
            'errors_by_severity': error_by_severity,
            'errors_by_column': error_by_column,
            'most_common_error': self._get_most_common_error(),
            'unique_error_messages': len(set(error.error_message for error in self.error_details))
        }
    
    def _get_most_common_error(self) -> Optional[str]:
        """Get the most common error message."""
        if not self.error_details:
            return None
        
        error_counts = {}
        for error in self.error_details:
            message = error.error_message
            error_counts[message] = error_counts.get(message, 0) + 1
        
        return max(error_counts, key=error_counts.get)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        return {
            'validation_id': str(self.validation_id),
            'rule_id': str(self.rule_id),
            'dataset_id': str(self.dataset_id),
            'status': self.status.value,
            'success_rate': self.get_success_rate(),
            'failure_rate': self.failure_rate,
            'total_records': self.total_records,
            'passed_records': self.passed_records,
            'failed_records': self.failed_records,
            'warning_count': self.warning_count,
            'skipped_count': self.skipped_count,
            'execution_time_seconds': self.execution_time.total_seconds(),
            'validated_at': self.validated_at.isoformat(),
            'has_errors': self.has_errors(),
            'has_warnings': self.has_warnings(),
            'error_summary': self.get_error_summary()
        }
    
    def get_critical_errors(self) -> List[ValidationError]:
        """Get all critical errors."""
        return [error for error in self.error_details 
                if error.severity == Severity.CRITICAL]
    
    def get_high_severity_errors(self) -> List[ValidationError]:
        """Get all high severity errors."""
        return [error for error in self.error_details 
                if error.severity in [Severity.CRITICAL, Severity.HIGH]]
    
    def get_errors_by_column(self, column_name: str) -> List[ValidationError]:
        """Get errors for specific column."""
        return [error for error in self.error_details 
                if error.column_name == column_name]


# Import statements for forward references
from .quality_profile import DatasetId