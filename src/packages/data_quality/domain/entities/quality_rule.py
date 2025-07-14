"""Quality Rule entity for data validation and quality management."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from packages.core.domain.abstractions import BaseEntity


@dataclass(frozen=True)
class RuleId:
    """Quality Rule identifier."""
    value: UUID = Field(default_factory=uuid4)


@dataclass(frozen=True)
class DatasetId:
    """Dataset identifier."""
    value: UUID = Field(default_factory=uuid4)


@dataclass(frozen=True)
class UserId:
    """User identifier."""
    value: UUID = Field(default_factory=uuid4)


class RuleType(str, Enum):
    """Quality rule type enumeration."""
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    CUSTOM = "custom"


class Severity(str, Enum):
    """Rule violation severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RuleStatus(str, Enum):
    """Rule status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"


class ValidationStatus(str, Enum):
    """Validation execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class LogicType(str, Enum):
    """Validation logic type enumeration."""
    SQL = "sql"
    PYTHON = "python"
    REGEX = "regex"
    RANGE = "range"
    LIST = "list"
    CUSTOM_FUNCTION = "custom_function"


class QualityCategory(str, Enum):
    """Quality category enumeration."""
    DATA_INTEGRITY = "data_integrity"
    BUSINESS_RULES = "business_rules"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    FORMAT_VALIDATION = "format_validation"
    STATISTICAL_VALIDATION = "statistical_validation"


class ValidationLogic(BaseModel):
    """Validation logic configuration."""
    logic_type: LogicType
    expression: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    error_message_template: str
    success_criteria: Optional[str] = None
    
    class Config:
        frozen = True


class RuleSchedule(BaseModel):
    """Rule execution schedule."""
    enabled: bool = True
    cron_expression: Optional[str] = None
    frequency: Optional[str] = None  # "hourly", "daily", "weekly", "monthly"
    next_execution: Optional[datetime] = None
    
    class Config:
        frozen = True


class ValidationError(BaseModel):
    """Individual validation error."""
    error_id: UUID = Field(default_factory=uuid4)
    row_identifier: Optional[str] = None
    column_name: Optional[str] = None
    invalid_value: Optional[str] = None
    error_message: str
    error_code: Optional[str] = None
    
    class Config:
        frozen = True


class ValidationResult(BaseModel):
    """Result of rule validation execution."""
    validation_id: UUID = Field(default_factory=uuid4)
    rule_id: RuleId
    dataset_id: DatasetId
    status: ValidationStatus
    
    # Execution metrics
    total_records: int
    records_passed: int
    records_failed: int
    pass_rate: float
    
    # Error details
    validation_errors: List[ValidationError] = Field(default_factory=list)
    
    # Performance metrics
    execution_time_seconds: float
    memory_usage_mb: Optional[float] = None
    
    # Execution context
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    executed_by: Optional[UserId] = None
    execution_environment: Optional[str] = None
    
    class Config:
        frozen = True


class QualityThreshold(BaseModel):
    """Quality threshold configuration."""
    pass_rate_threshold: float = 0.95  # 95% pass rate required
    warning_threshold: float = 0.90    # Warning below 90%
    critical_threshold: float = 0.80   # Critical below 80%
    
    class Config:
        frozen = True


class RuleMetadata(BaseModel):
    """Rule metadata and documentation."""
    description: str
    business_justification: str
    data_owner: Optional[str] = None
    business_glossary_terms: List[str] = Field(default_factory=list)
    related_regulations: List[str] = Field(default_factory=list)
    documentation_url: Optional[str] = None
    
    class Config:
        frozen = True


class QualityRule(BaseEntity):
    """Quality rule aggregate root."""
    
    rule_id: RuleId
    rule_name: str
    rule_type: RuleType
    category: QualityCategory
    severity: Severity
    status: RuleStatus = RuleStatus.DRAFT
    
    # Rule definition
    validation_logic: ValidationLogic
    target_datasets: List[DatasetId] = Field(default_factory=list)
    target_columns: List[str] = Field(default_factory=list)
    
    # Quality thresholds
    thresholds: QualityThreshold = Field(default_factory=QualityThreshold)
    
    # Scheduling
    schedule: Optional[RuleSchedule] = None
    
    # Metadata
    rule_metadata: RuleMetadata
    
    # Execution history (last N results)
    recent_results: List[ValidationResult] = Field(default_factory=list)
    
    # Ownership and governance
    created_by: UserId
    approved_by: Optional[UserId] = None
    approved_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        
    def activate(self, approved_by: UserId) -> None:
        """Activate the rule after approval."""
        if self.status != RuleStatus.DRAFT:
            raise ValueError("Only draft rules can be activated")
            
        self.status = RuleStatus.ACTIVE
        self.approved_by = approved_by
        self.approved_at = datetime.utcnow()
        self.mark_as_updated()
        
    def deactivate(self) -> None:
        """Deactivate the rule."""
        if self.status not in [RuleStatus.ACTIVE, RuleStatus.DRAFT]:
            raise ValueError("Cannot deactivate rule in current status")
            
        self.status = RuleStatus.INACTIVE
        self.mark_as_updated()
        
    def deprecate(self) -> None:
        """Deprecate the rule."""
        self.status = RuleStatus.DEPRECATED
        self.mark_as_updated()
        
    def update_validation_logic(self, logic: ValidationLogic) -> None:
        """Update the validation logic."""
        if self.status == RuleStatus.ACTIVE:
            # For active rules, create new version
            self.status = RuleStatus.DRAFT
            
        self.validation_logic = logic
        self.mark_as_updated()
        
    def add_validation_result(self, result: ValidationResult) -> None:
        """Add a new validation result."""
        # Keep only the most recent 10 results
        self.recent_results.append(result)
        if len(self.recent_results) > 10:
            self.recent_results.pop(0)
            
        self.mark_as_updated()
        
    def get_latest_result(self) -> Optional[ValidationResult]:
        """Get the most recent validation result."""
        return self.recent_results[-1] if self.recent_results else None
        
    def get_current_pass_rate(self) -> Optional[float]:
        """Get the current pass rate from latest result."""
        latest_result = self.get_latest_result()
        return latest_result.pass_rate if latest_result else None
        
    def is_threshold_violated(self) -> bool:
        """Check if quality thresholds are violated."""
        current_pass_rate = self.get_current_pass_rate()
        if current_pass_rate is None:
            return False
            
        return current_pass_rate < self.thresholds.pass_rate_threshold
        
    def is_critical_violation(self) -> bool:
        """Check if violation is critical."""
        current_pass_rate = self.get_current_pass_rate()
        if current_pass_rate is None:
            return False
            
        return current_pass_rate < self.thresholds.critical_threshold
        
    def get_violation_severity_level(self) -> Optional[str]:
        """Get the current violation severity level."""
        current_pass_rate = self.get_current_pass_rate()
        if current_pass_rate is None:
            return None
            
        if current_pass_rate < self.thresholds.critical_threshold:
            return "critical"
        elif current_pass_rate < self.thresholds.warning_threshold:
            return "warning"
        elif current_pass_rate < self.thresholds.pass_rate_threshold:
            return "minor"
        else:
            return "passing"
            
    def is_active(self) -> bool:
        """Check if rule is active."""
        return self.status == RuleStatus.ACTIVE
        
    def is_scheduled(self) -> bool:
        """Check if rule has scheduling enabled."""
        return self.schedule is not None and self.schedule.enabled
        
    def should_execute_now(self) -> bool:
        """Check if rule should execute based on schedule."""
        if not self.is_scheduled() or not self.schedule:
            return False
            
        if self.schedule.next_execution is None:
            return True
            
        return datetime.utcnow() >= self.schedule.next_execution
        
    def calculate_next_execution(self) -> None:
        """Calculate next execution time based on schedule."""
        if not self.schedule or not self.schedule.enabled:
            return
            
        # Simple frequency-based calculation
        # In production, use proper cron expression parsing
        if self.schedule.frequency == "hourly":
            next_exec = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            next_exec = next_exec.replace(hour=next_exec.hour + 1)
        elif self.schedule.frequency == "daily":
            next_exec = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            from datetime import timedelta
            next_exec = next_exec + timedelta(days=1)
        else:
            return
            
        self.schedule.next_execution = next_exec