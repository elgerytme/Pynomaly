"""Data Quality Check domain entity for data quality assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class CheckType(str, Enum):
    """Type of data quality check."""
    
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"  
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    INTEGRITY = "integrity"
    CONFORMITY = "conformity"
    PRECISION = "precision"
    CUSTOM = "custom"


class CheckStatus(str, Enum):
    """Status of data quality check execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class CheckSeverity(str, Enum):
    """Severity level of check results."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CheckResult:
    """Result of a data quality check execution."""
    
    id: UUID = field(default_factory=uuid4)
    check_id: UUID = field(default_factory=uuid4)
    dataset_name: str = ""
    column_name: Optional[str] = None
    
    # Result metrics
    passed: bool = False
    score: float = 0.0  # 0.0 to 1.0
    total_records: int = 0
    passed_records: int = 0
    failed_records: int = 0
    
    # Execution details
    executed_at: datetime = field(default_factory=datetime.utcnow)
    execution_time_ms: float = 0.0
    severity: CheckSeverity = CheckSeverity.INFO
    
    # Result details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    failed_values: List[Any] = field(default_factory=list)
    sample_failures: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate check result after initialization."""
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("Score must be between 0.0 and 1.0")
        
        if self.total_records < 0:
            raise ValueError("Total records cannot be negative")
        
        if self.passed_records < 0:
            raise ValueError("Passed records cannot be negative")
        
        if self.failed_records < 0:
            raise ValueError("Failed records cannot be negative")
        
        if self.passed_records + self.failed_records > self.total_records:
            raise ValueError("Sum of passed and failed records cannot exceed total")
    
    @property
    def pass_rate(self) -> float:
        """Get pass rate as percentage."""
        if self.total_records == 0:
            return 0.0
        return (self.passed_records / self.total_records) * 100
    
    @property
    def fail_rate(self) -> float:
        """Get failure rate as percentage."""
        if self.total_records == 0:
            return 0.0
        return (self.failed_records / self.total_records) * 100
    
    @property
    def is_passed(self) -> bool:
        """Check if the result indicates a pass."""
        return self.passed
    
    @property
    def is_critical(self) -> bool:
        """Check if the result has critical severity."""
        return self.severity == CheckSeverity.CRITICAL
    
    def add_failed_value(self, value: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Add a failed value to the result."""
        self.failed_values.append(value)
        
        if context and len(self.sample_failures) < 100:  # Limit sample size
            failure_info = {"value": value, **context}
            self.sample_failures.append(failure_info)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the result."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata key-value pair."""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "id": str(self.id),
            "check_id": str(self.check_id),
            "dataset_name": self.dataset_name,
            "column_name": self.column_name,
            "passed": self.passed,
            "score": self.score,
            "total_records": self.total_records,
            "passed_records": self.passed_records,
            "failed_records": self.failed_records,
            "pass_rate": self.pass_rate,
            "fail_rate": self.fail_rate,
            "executed_at": self.executed_at.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "failed_values": self.failed_values[:10],  # Limit for serialization
            "sample_failures": self.sample_failures[:10],
            "metadata": self.metadata,
            "tags": self.tags,
            "is_passed": self.is_passed,
            "is_critical": self.is_critical,
        }


@dataclass
class DataQualityCheck:
    """Data quality check domain entity representing a quality assessment rule."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    check_type: CheckType = CheckType.COMPLETENESS
    
    # Target specification
    dataset_name: str = ""
    column_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    
    # Check configuration
    query: Optional[str] = None
    expression: Optional[str] = None
    expected_value: Optional[Any] = None
    threshold: float = 0.95  # Pass threshold (0.0 to 1.0)
    tolerance: float = 0.0  # Acceptable variance
    
    # Execution settings
    is_active: bool = True
    schedule_cron: Optional[str] = None
    timeout_seconds: int = 300
    retry_attempts: int = 3
    
    # Status and tracking
    status: CheckStatus = CheckStatus.PENDING
    last_executed_at: Optional[datetime] = None
    next_execution_at: Optional[datetime] = None
    execution_count: int = 0
    
    # Results tracking
    last_result: Optional[CheckResult] = None
    consecutive_failures: int = 0
    success_rate: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = ""
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Dependencies
    depends_on: List[UUID] = field(default_factory=list)
    blocks: List[UUID] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate check after initialization."""
        if not self.name:
            raise ValueError("Check name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Check name cannot exceed 100 characters")
        
        if not self.dataset_name:
            raise ValueError("Dataset name cannot be empty")
        
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        if self.tolerance < 0.0:
            raise ValueError("Tolerance cannot be negative")
        
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.retry_attempts < 0:
            raise ValueError("Retry attempts cannot be negative")
    
    @property
    def is_column_level(self) -> bool:
        """Check if this is a column-level check."""
        return self.column_name is not None
    
    @property
    def is_table_level(self) -> bool:
        """Check if this is a table-level check."""
        return self.column_name is None
    
    @property
    def is_overdue(self) -> bool:
        """Check if the check is overdue for execution."""
        if not self.next_execution_at:
            return False
        return datetime.utcnow() > self.next_execution_at
    
    @property
    def has_recent_failure(self) -> bool:
        """Check if there was a recent failure."""
        return self.consecutive_failures > 0
    
    @property
    def is_healthy(self) -> bool:
        """Check if the check is in a healthy state."""
        return (
            self.is_active and
            not self.has_recent_failure and
            self.success_rate >= 0.8
        )
    
    def execute(self) -> CheckResult:
        """Execute the data quality check."""
        if not self.is_active:
            raise ValueError("Cannot execute inactive check")
        
        self.status = CheckStatus.RUNNING
        self.last_executed_at = datetime.utcnow()
        self.execution_count += 1
        self.updated_at = self.last_executed_at
        
        # This would be implemented by infrastructure layer
        # For now, simulate a check execution
        result = CheckResult(
            check_id=self.id,
            dataset_name=self.dataset_name,
            column_name=self.column_name,
            executed_at=self.last_executed_at
        )
        
        # Simulate execution logic based on check type
        result.total_records = 1000
        result.passed_records = int(1000 * self.threshold)
        result.failed_records = result.total_records - result.passed_records
        result.score = self.threshold
        result.passed = result.score >= self.threshold
        result.message = f"Check {self.name} executed successfully"
        
        self.last_result = result
        self.status = CheckStatus.COMPLETED
        
        # Update success tracking
        if result.passed:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        
        # Update success rate (simple moving average)
        if self.execution_count == 1:
            self.success_rate = 1.0 if result.passed else 0.0
        else:
            weight = 0.1  # Give more weight to recent results
            current_success = 1.0 if result.passed else 0.0
            self.success_rate = (1 - weight) * self.success_rate + weight * current_success
        
        return result
    
    def validate_expression(self) -> bool:
        """Validate the check expression syntax."""
        if not self.expression:
            return True
        
        # Basic validation - would be more comprehensive in real implementation
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
        expression_upper = self.expression.upper()
        
        for keyword in forbidden_keywords:
            if keyword in expression_upper:
                return False
        
        return True
    
    def activate(self) -> None:
        """Activate the check."""
        self.is_active = True
        self.status = CheckStatus.PENDING
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate the check."""
        self.is_active = False
        self.status = CheckStatus.CANCELLED
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the check."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the check."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if check has a specific tag."""
        return tag in self.tags
    
    def add_dependency(self, check_id: UUID) -> None:
        """Add a dependency on another check."""
        if check_id not in self.depends_on:
            self.depends_on.append(check_id)
            self.updated_at = datetime.utcnow()
    
    def remove_dependency(self, check_id: UUID) -> None:
        """Remove a dependency."""
        if check_id in self.depends_on:
            self.depends_on.remove(check_id)
            self.updated_at = datetime.utcnow()
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration key-value pair."""
        self.config[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_full_target_name(self) -> str:
        """Get the full target name for the check."""
        parts = []
        if self.schema_name:
            parts.append(self.schema_name)
        if self.table_name:
            parts.append(self.table_name)
        elif self.dataset_name:
            parts.append(self.dataset_name)
        if self.column_name:
            parts.append(self.column_name)
        
        return ".".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert check to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "check_type": self.check_type.value,
            "dataset_name": self.dataset_name,
            "column_name": self.column_name,
            "schema_name": self.schema_name,
            "table_name": self.table_name,
            "query": self.query,
            "expression": self.expression,
            "expected_value": self.expected_value,
            "threshold": self.threshold,
            "tolerance": self.tolerance,
            "is_active": self.is_active,
            "schedule_cron": self.schedule_cron,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
            "status": self.status.value,
            "last_executed_at": (
                self.last_executed_at.isoformat() 
                if self.last_executed_at else None
            ),
            "next_execution_at": (
                self.next_execution_at.isoformat()
                if self.next_execution_at else None
            ),
            "execution_count": self.execution_count,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
            "config": self.config,
            "environment_vars": self.environment_vars,
            "tags": self.tags,
            "depends_on": [str(dep_id) for dep_id in self.depends_on],
            "blocks": [str(block_id) for block_id in self.blocks],
            "is_column_level": self.is_column_level,
            "is_table_level": self.is_table_level,
            "is_overdue": self.is_overdue,
            "has_recent_failure": self.has_recent_failure,
            "is_healthy": self.is_healthy,
            "full_target_name": self.get_full_target_name(),
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        target = self.get_full_target_name()
        return f"DataQualityCheck('{self.name}', {self.check_type.value}, target={target})"
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return (
            f"DataQualityCheck(id={self.id}, name='{self.name}', "
            f"type={self.check_type.value}, status={self.status.value})"
        )