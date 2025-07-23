"""Data Quality Rule domain entity for rule-based quality management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


class RuleType(str, Enum):
    """Type of data quality rule."""
    
    RANGE = "range"
    PATTERN = "pattern"
    LIST = "list"
    REFERENCE = "reference"
    UNIQUENESS = "uniqueness"
    NOT_NULL = "not_null"
    CUSTOM = "custom"
    STATISTICAL = "statistical"
    BUSINESS = "business"


class RuleSeverity(str, Enum):
    """Severity level of rule violations."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleOperator(str, Enum):
    """Operators for rule conditions."""
    
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    GREATER_EQUAL = "greater_equal"
    LESS_THAN = "less_than"
    LESS_EQUAL = "less_equal"
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"
    NOT_MATCHES = "not_matches"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


@dataclass
class RuleCondition:
    """Individual condition within a data quality rule."""
    
    id: UUID = field(default_factory=uuid4)
    column_name: str = ""
    operator: RuleOperator = RuleOperator.EQUALS
    value: Optional[Union[str, int, float, List[Any]]] = None
    case_sensitive: bool = True
    
    # Advanced conditions
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    reference_table: Optional[str] = None
    reference_column: Optional[str] = None
    
    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate condition after initialization."""
        if not self.column_name:
            raise ValueError("Column name cannot be empty")
        
        # Validate operator-specific requirements
        if self.operator in [RuleOperator.BETWEEN] and (
            self.min_value is None or self.max_value is None
        ):
            raise ValueError("BETWEEN operator requires min_value and max_value")
        
        if self.operator in [RuleOperator.IN, RuleOperator.NOT_IN] and (
            not isinstance(self.value, list) or not self.value
        ):
            raise ValueError("IN/NOT_IN operators require a non-empty list value")
        
        if self.operator in [RuleOperator.MATCHES, RuleOperator.NOT_MATCHES] and not self.pattern:
            raise ValueError("MATCHES operators require a pattern")
    
    def evaluate(self, column_value: Any) -> bool:
        """Evaluate the condition against a column value."""
        if column_value is None:
            return self.operator in [RuleOperator.IS_NULL]
        
        if self.operator == RuleOperator.IS_NOT_NULL:
            return column_value is not None
        
        if self.operator == RuleOperator.IS_NULL:
            return column_value is None
        
        # String operations
        if isinstance(column_value, str):
            test_value = column_value if self.case_sensitive else column_value.lower()
            comparison_value = self.value
            
            if isinstance(comparison_value, str) and not self.case_sensitive:
                comparison_value = comparison_value.lower()
            
            if self.operator == RuleOperator.EQUALS:
                return test_value == comparison_value
            elif self.operator == RuleOperator.NOT_EQUALS:
                return test_value != comparison_value
            elif self.operator == RuleOperator.CONTAINS:
                return str(comparison_value) in test_value
            elif self.operator == RuleOperator.NOT_CONTAINS:
                return str(comparison_value) not in test_value
            elif self.operator == RuleOperator.STARTS_WITH:
                return test_value.startswith(str(comparison_value))
            elif self.operator == RuleOperator.ENDS_WITH:
                return test_value.endswith(str(comparison_value))
            elif self.operator == RuleOperator.MATCHES:
                import re
                return bool(re.match(self.pattern or "", test_value))
            elif self.operator == RuleOperator.NOT_MATCHES:
                import re
                return not bool(re.match(self.pattern or "", test_value))
        
        # Numeric operations
        if isinstance(column_value, (int, float)):
            if self.operator == RuleOperator.EQUALS:
                return column_value == self.value
            elif self.operator == RuleOperator.NOT_EQUALS:
                return column_value != self.value
            elif self.operator == RuleOperator.GREATER_THAN:
                return column_value > self.value
            elif self.operator == RuleOperator.GREATER_EQUAL:
                return column_value >= self.value
            elif self.operator == RuleOperator.LESS_THAN:
                return column_value < self.value
            elif self.operator == RuleOperator.LESS_EQUAL:
                return column_value <= self.value
            elif self.operator == RuleOperator.BETWEEN:
                return self.min_value <= column_value <= self.max_value
        
        # List operations
        if self.operator == RuleOperator.IN:
            return column_value in (self.value or [])
        elif self.operator == RuleOperator.NOT_IN:
            return column_value not in (self.value or [])
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert condition to dictionary."""
        return {
            "id": str(self.id),
            "column_name": self.column_name,
            "operator": self.operator.value,
            "value": self.value,
            "case_sensitive": self.case_sensitive,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "pattern": self.pattern,
            "reference_table": self.reference_table,
            "reference_column": self.reference_column,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DataQualityRule:
    """Data quality rule domain entity for defining quality constraints."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    rule_type: RuleType = RuleType.NOT_NULL
    severity: RuleSeverity = RuleSeverity.ERROR
    
    # Target specification
    dataset_name: str = ""
    table_name: Optional[str] = None
    schema_name: Optional[str] = None
    
    # Rule definition
    conditions: List[RuleCondition] = field(default_factory=list)
    logical_operator: str = "AND"  # AND, OR
    expression: Optional[str] = None  # Custom SQL expression
    
    # Enforcement settings
    is_active: bool = True
    is_blocking: bool = False  # Blocks data pipeline if violated
    auto_fix: bool = False
    fix_action: Optional[str] = None
    
    # Thresholds
    violation_threshold: float = 0.0  # Max allowed violation rate (0.0 to 1.0)
    sample_size: Optional[int] = None  # Limit evaluation to sample
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = ""
    
    # Usage tracking
    last_evaluated_at: Optional[datetime] = None
    evaluation_count: int = 0
    violation_count: int = 0
    last_violation_at: Optional[datetime] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Dependencies
    depends_on: List[UUID] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate rule after initialization."""
        if not self.name:
            raise ValueError("Rule name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Rule name cannot exceed 100 characters")
        
        if not self.dataset_name:
            raise ValueError("Dataset name cannot be empty")
        
        if not (0.0 <= self.violation_threshold <= 1.0):
            raise ValueError("Violation threshold must be between 0.0 and 1.0")
        
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("Sample size must be positive")
        
        if self.logical_operator not in ["AND", "OR"]:
            raise ValueError("Logical operator must be 'AND' or 'OR'")
    
    @property
    def is_simple_rule(self) -> bool:
        """Check if this is a simple rule (single condition)."""
        return len(self.conditions) <= 1
    
    @property
    def is_complex_rule(self) -> bool:
        """Check if this is a complex rule (multiple conditions)."""
        return len(self.conditions) > 1
    
    @property
    def violation_rate(self) -> float:
        """Get current violation rate."""
        if self.evaluation_count == 0:
            return 0.0
        return self.violation_count / self.evaluation_count
    
    @property
    def is_healthy(self) -> bool:
        """Check if the rule is in a healthy state."""
        return (
            self.is_active and
            self.violation_rate <= self.violation_threshold
        )
    
    @property
    def has_recent_violations(self) -> bool:
        """Check if there were recent violations."""
        if not self.last_violation_at:
            return False
        
        # Consider violations within last 24 hours as recent
        time_diff = datetime.utcnow() - self.last_violation_at
        return time_diff.total_seconds() < 86400  # 24 hours
    
    def add_condition(self, condition: RuleCondition) -> None:
        """Add a condition to the rule."""
        # Check for duplicate column conditions
        existing_columns = [c.column_name for c in self.conditions]
        if condition.column_name in existing_columns:
            raise ValueError(f"Condition for column '{condition.column_name}' already exists")
        
        self.conditions.append(condition)
        self.updated_at = datetime.utcnow()
    
    def remove_condition(self, condition_id: UUID) -> bool:
        """Remove a condition from the rule."""
        for i, condition in enumerate(self.conditions):
            if condition.id == condition_id:
                self.conditions.pop(i)
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def get_condition(self, condition_id: UUID) -> Optional[RuleCondition]:
        """Get a condition by ID."""
        for condition in self.conditions:
            if condition.id == condition_id:
                return condition
        return None
    
    def get_condition_by_column(self, column_name: str) -> Optional[RuleCondition]:
        """Get a condition by column name."""
        for condition in self.conditions:
            if condition.column_name == column_name:
                return condition
        return None
    
    def evaluate_record(self, record: Dict[str, Any]) -> bool:
        """Evaluate the rule against a single record."""
        if not self.conditions:
            return True  # No conditions means rule passes
        
        results = []
        for condition in self.conditions:
            column_value = record.get(condition.column_name)
            result = condition.evaluate(column_value)
            results.append(result)
        
        # Apply logical operator
        if self.logical_operator == "AND":
            return all(results)
        else:  # OR
            return any(results)
    
    def validate_expression(self) -> bool:
        """Validate the rule expression syntax."""
        if not self.expression:
            return True
        
        # Basic validation - would be more comprehensive in real implementation
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        expression_upper = self.expression.upper()
        
        for keyword in forbidden_keywords:
            if keyword in expression_upper:
                return False
        
        return True
    
    def activate(self) -> None:
        """Activate the rule."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate the rule."""
        self.is_active = False
        self.updated_at = datetime.utcnow()
    
    def record_evaluation(self, had_violation: bool) -> None:
        """Record the result of a rule evaluation."""
        self.evaluation_count += 1
        self.last_evaluated_at = datetime.utcnow()
        
        if had_violation:
            self.violation_count += 1
            self.last_violation_at = self.last_evaluated_at
        
        self.updated_at = self.last_evaluated_at
    
    def reset_statistics(self) -> None:
        """Reset evaluation statistics."""
        self.evaluation_count = 0
        self.violation_count = 0
        self.last_evaluated_at = None
        self.last_violation_at = None
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the rule."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the rule."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if rule has a specific tag."""
        return tag in self.tags
    
    def add_dependency(self, rule_id: UUID) -> None:
        """Add a dependency on another rule."""
        if rule_id not in self.depends_on:
            self.depends_on.append(rule_id)
            self.updated_at = datetime.utcnow()
    
    def remove_dependency(self, rule_id: UUID) -> None:
        """Remove a dependency."""
        if rule_id in self.depends_on:
            self.depends_on.remove(rule_id)
            self.updated_at = datetime.utcnow()
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration key-value pair."""
        self.config[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_target_name(self) -> str:
        """Get the full target name for the rule."""
        parts = []
        if self.schema_name:
            parts.append(self.schema_name)
        if self.table_name:
            parts.append(self.table_name)
        elif self.dataset_name:
            parts.append(self.dataset_name)
        
        return ".".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type.value,
            "severity": self.severity.value,
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
            "schema_name": self.schema_name,
            "conditions": [c.to_dict() for c in self.conditions],
            "logical_operator": self.logical_operator,
            "expression": self.expression,
            "is_active": self.is_active,
            "is_blocking": self.is_blocking,
            "auto_fix": self.auto_fix,
            "fix_action": self.fix_action,
            "violation_threshold": self.violation_threshold,
            "sample_size": self.sample_size,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
            "last_evaluated_at": (
                self.last_evaluated_at.isoformat()
                if self.last_evaluated_at else None
            ),
            "evaluation_count": self.evaluation_count,
            "violation_count": self.violation_count,
            "last_violation_at": (
                self.last_violation_at.isoformat()
                if self.last_violation_at else None
            ),
            "config": self.config,
            "tags": self.tags,
            "depends_on": [str(dep_id) for dep_id in self.depends_on],
            "is_simple_rule": self.is_simple_rule,
            "is_complex_rule": self.is_complex_rule,
            "violation_rate": self.violation_rate,
            "is_healthy": self.is_healthy,
            "has_recent_violations": self.has_recent_violations,
            "target_name": self.get_target_name(),
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        target = self.get_target_name()
        return f"DataQualityRule('{self.name}', {self.rule_type.value}, target={target})"
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return (
            f"DataQualityRule(id={self.id}, name='{self.name}', "
            f"type={self.rule_type.value}, severity={self.severity.value})"
        )