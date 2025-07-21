"""
Service Level Agreement (SLA) domain entities for enterprise governance.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class SLAStatus(str, Enum):
    """SLA status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    PENDING_APPROVAL = "pending_approval"
    UNDER_REVIEW = "under_review"


class SLAType(str, Enum):
    """Types of SLAs."""
    AVAILABILITY = "availability"
    PERFORMANCE = "performance"
    RESPONSE_TIME = "response_time"
    RESOLUTION_TIME = "resolution_time"
    SUPPORT = "support"
    DATA_PROCESSING = "data_processing"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    CUSTOM = "custom"


class SLAMetricType(str, Enum):
    """SLA metric measurement types."""
    PERCENTAGE = "percentage"
    TIME_DURATION = "time_duration"
    COUNT = "count"
    RATE = "rate"
    BOOLEAN = "boolean"


class SLAViolationSeverity(str, Enum):
    """SLA violation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SLAMetric(BaseModel):
    """
    Individual SLA metric definition and measurement.
    
    Defines specific measurable criteria for service level agreements
    including thresholds, measurement methods, and performance targets.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Metric identifier")
    
    # Metric identification
    name: str = Field(..., description="Metric name")
    description: str = Field(..., description="Metric description")
    metric_type: SLAMetricType = Field(..., description="Type of metric measurement")
    
    # Measurement configuration
    target_value: float = Field(..., description="Target performance value")
    minimum_acceptable: float = Field(..., description="Minimum acceptable performance")
    measurement_unit: str = Field(..., description="Unit of measurement")
    measurement_frequency: str = Field(..., description="How often to measure")
    
    # Thresholds
    warning_threshold: Optional[float] = Field(None, description="Warning threshold")
    critical_threshold: Optional[float] = Field(None, description="Critical threshold")
    
    # Calculation method
    calculation_method: str = Field(..., description="How the metric is calculated")
    data_source: str = Field(..., description="Source of measurement data")
    aggregation_period: str = Field(..., description="Period for aggregation")
    
    # Current performance
    current_value: Optional[float] = Field(None, description="Current measured value")
    last_measured_at: Optional[datetime] = Field(None, description="Last measurement time")
    measurement_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Status
    is_meeting_target: bool = Field(default=True, description="Whether metric meets target")
    consecutive_failures: int = Field(default=0, description="Consecutive failures count")
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def record_measurement(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Record a new measurement for this metric."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Store measurement history
        self.measurement_history.append({
            "value": value,
            "timestamp": timestamp.isoformat(),
            "meets_target": value >= self.target_value if self.metric_type != SLAMetricType.TIME_DURATION else value <= self.target_value
        })
        
        # Update current values
        self.current_value = value
        self.last_measured_at = timestamp
        
        # Check if meeting target
        is_meeting = self._check_target_compliance(value)
        if is_meeting:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        
        self.is_meeting_target = is_meeting
        self.updated_at = timestamp
    
    def _check_target_compliance(self, value: float) -> bool:
        """Check if a value meets the target."""
        if self.metric_type == SLAMetricType.TIME_DURATION:
            # For time duration, lower is better
            return value <= self.target_value
        elif self.metric_type == SLAMetricType.PERCENTAGE:
            # For percentage, higher is usually better (e.g., availability)
            return value >= self.target_value
        else:
            # For other metrics, check against target
            return value >= self.target_value
    
    def get_compliance_percentage(self, days: int = 30) -> float:
        """Calculate compliance percentage over specified days."""
        if not self.measurement_history:
            return 100.0
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        recent_measurements = [
            m for m in self.measurement_history
            if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
        ]
        
        if not recent_measurements:
            return 100.0
        
        compliant_count = sum(1 for m in recent_measurements if m["meets_target"])
        return (compliant_count / len(recent_measurements)) * 100.0
    
    def is_in_violation(self) -> bool:
        """Check if metric is currently in violation."""
        if not self.current_value:
            return False
        
        return not self._check_target_compliance(self.current_value)
    
    def get_violation_severity(self) -> Optional[SLAViolationSeverity]:
        """Get the severity of current violation, if any."""
        if self.is_meeting_target:
            return None
        
        if self.critical_threshold and self.current_value <= self.critical_threshold:
            return SLAViolationSeverity.CRITICAL
        elif self.warning_threshold and self.current_value <= self.warning_threshold:
            return SLAViolationSeverity.HIGH
        else:
            return SLAViolationSeverity.MEDIUM


class ServiceLevelAgreement(BaseModel):
    """
    Service Level Agreement defining performance commitments and obligations.
    
    Comprehensive SLA management including metrics, targets, penalties,
    and performance tracking for enterprise service commitments.
    """
    
    id: UUID = Field(default_factory=uuid4, description="SLA identifier")
    
    # SLA identification
    name: str = Field(..., description="SLA name")
    version: str = Field(default="1.0", description="SLA version")
    description: str = Field(..., description="SLA description")
    sla_type: SLAType = Field(..., description="Type of SLA")
    
    # Parties involved
    service_provider: str = Field(..., description="Service provider")
    service_consumer: str = Field(..., description="Service consumer")
    tenant_id: UUID = Field(..., description="Associated tenant")
    
    # SLA scope and coverage
    services_covered: List[str] = Field(..., description="Services covered by this SLA")
    service_hours: str = Field(..., description="Service hours (e.g., 24x7, business hours)")
    exclusions: List[str] = Field(default_factory=list, description="SLA exclusions")
    
    # Performance metrics
    metrics: List[UUID] = Field(..., description="Associated SLA metric IDs")
    overall_target: float = Field(..., description="Overall SLA target")
    measurement_period: str = Field(..., description="SLA measurement period")
    
    # Contract details
    effective_date: datetime = Field(..., description="SLA effective date")
    expiry_date: Optional[datetime] = Field(None, description="SLA expiry date")
    auto_renewal: bool = Field(default=False, description="Auto-renewal enabled")
    renewal_period: Optional[str] = Field(None, description="Auto-renewal period")
    
    # Status and tracking
    status: SLAStatus = Field(default=SLAStatus.ACTIVE)
    current_compliance: float = Field(default=100.0, ge=0.0, le=100.0, description="Current compliance %")
    last_compliance_check: Optional[datetime] = Field(None, description="Last compliance check")
    
    # Penalties and remedies
    penalties_enabled: bool = Field(default=False, description="Penalties for violations")
    penalty_structure: List[Dict[str, Any]] = Field(default_factory=list, description="Penalty definitions")
    credits_earned: float = Field(default=0.0, description="Service credits earned")
    
    # Reporting and notifications
    reporting_frequency: str = Field(..., description="Reporting frequency")
    notification_contacts: List[str] = Field(default_factory=list, description="Notification contacts")
    escalation_matrix: List[Dict[str, Any]] = Field(default_factory=list, description="Escalation procedures")
    
    # Review and governance
    review_schedule: str = Field(..., description="SLA review schedule")
    next_review_date: Optional[datetime] = Field(None, description="Next review date")
    stakeholders: List[str] = Field(default_factory=list, description="SLA stakeholders")
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    attachments: List[str] = Field(default_factory=list, description="SLA documents")
    external_references: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('expiry_date')
    def expiry_after_effective(cls, v, values):
        """Ensure expiry date is after effective date."""
        if v and 'effective_date' in values and v <= values['effective_date']:
            raise ValueError('Expiry date must be after effective date')
        return v
    
    def is_active(self) -> bool:
        """Check if SLA is currently active."""
        now = datetime.utcnow()
        return (
            self.status == SLAStatus.ACTIVE and
            self.effective_date <= now and
            (not self.expiry_date or self.expiry_date > now)
        )
    
    def is_expired(self) -> bool:
        """Check if SLA has expired."""
        if not self.expiry_date:
            return False
        return datetime.utcnow() > self.expiry_date
    
    def update_compliance(self, compliance_percentage: float) -> None:
        """Update current compliance percentage."""
        self.current_compliance = max(0.0, min(100.0, compliance_percentage))
        self.last_compliance_check = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def add_penalty(self, violation_type: str, penalty_amount: float, description: str) -> None:
        """Add a penalty structure."""
        penalty = {
            "id": str(uuid4()),
            "violation_type": violation_type,
            "penalty_amount": penalty_amount,
            "description": description,
            "created_at": datetime.utcnow().isoformat()
        }
        self.penalty_structure.append(penalty)
        self.updated_at = datetime.utcnow()
    
    def calculate_service_credits(self, violation_duration: float, violation_type: str) -> float:
        """Calculate service credits for SLA violations."""
        applicable_penalties = [
            p for p in self.penalty_structure
            if p["violation_type"] == violation_type
        ]
        
        if not applicable_penalties:
            return 0.0
        
        # Use the first matching penalty (could be enhanced for complex rules)
        penalty = applicable_penalties[0]
        credits = penalty["penalty_amount"] * violation_duration
        
        self.credits_earned += credits
        self.updated_at = datetime.utcnow()
        
        return credits
    
    def schedule_next_review(self, months_ahead: int = 12) -> None:
        """Schedule next SLA review."""
        from dateutil.relativedelta import relativedelta
        self.next_review_date = datetime.utcnow() + relativedelta(months=months_ahead)
        self.updated_at = datetime.utcnow()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get SLA performance summary."""
        return {
            "sla_name": self.name,
            "status": self.status,
            "current_compliance": self.current_compliance,
            "overall_target": self.overall_target,
            "is_meeting_target": self.current_compliance >= self.overall_target,
            "credits_earned": self.credits_earned,
            "is_active": self.is_active(),
            "is_expired": self.is_expired(),
            "days_until_expiry": (self.expiry_date - datetime.utcnow()).days if self.expiry_date else None
        }


class SLAViolation(BaseModel):
    """
    SLA violation incident tracking and management.
    
    Records and tracks SLA violations including impact assessment,
    root cause analysis, and remediation actions.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Violation identifier")
    
    # Violation identification
    sla_id: UUID = Field(..., description="Associated SLA ID")
    metric_id: UUID = Field(..., description="Associated metric ID")
    tenant_id: UUID = Field(..., description="Tenant ID")
    
    # Violation details
    violation_type: str = Field(..., description="Type of violation")
    severity: SLAViolationSeverity = Field(..., description="Violation severity")
    description: str = Field(..., description="Violation description")
    
    # Timing
    start_time: datetime = Field(..., description="Violation start time")
    end_time: Optional[datetime] = Field(None, description="Violation end time")
    duration_minutes: Optional[float] = Field(None, description="Violation duration")
    
    # Impact assessment
    affected_services: List[str] = Field(default_factory=list, description="Affected services")
    affected_users: Optional[int] = Field(None, description="Number of affected users")
    business_impact: str = Field(default="", description="Business impact description")
    financial_impact: Optional[float] = Field(None, description="Estimated financial impact")
    
    # Performance data
    target_value: float = Field(..., description="SLA target value")
    actual_value: float = Field(..., description="Actual measured value")
    deviation_percentage: float = Field(..., description="Deviation from target")
    
    # Root cause and resolution
    root_cause: str = Field(default="", description="Root cause analysis")
    resolution_actions: List[str] = Field(default_factory=list, description="Resolution actions taken")
    preventive_measures: List[str] = Field(default_factory=list, description="Preventive measures")
    
    # Notifications and escalations
    notifications_sent: List[Dict[str, Any]] = Field(default_factory=list)
    escalated_to: List[str] = Field(default_factory=list, description="Escalated to parties")
    escalation_level: int = Field(default=0, description="Current escalation level")
    
    # Status tracking
    is_resolved: bool = Field(default=False, description="Violation resolved")
    requires_follow_up: bool = Field(default=False, description="Requires follow-up")
    follow_up_date: Optional[datetime] = Field(None, description="Follow-up due date")
    
    # Service credits and penalties
    service_credits_due: float = Field(default=0.0, description="Service credits owed")
    penalty_applied: float = Field(default=0.0, description="Penalty amount")
    credit_applied: bool = Field(default=False, description="Credits applied to account")
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    attachments: List[str] = Field(default_factory=list, description="Supporting documents")
    
    # Timestamps
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="When violation was detected")
    acknowledged_at: Optional[datetime] = Field(None, description="When violation was acknowledged")
    resolved_at: Optional[datetime] = Field(None, description="When violation was resolved")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('deviation_percentage', pre=True, always=True)
    def calculate_deviation(cls, v, values):
        """Calculate deviation percentage from target and actual values."""
        if 'target_value' in values and 'actual_value' in values:
            target = values['target_value']
            actual = values['actual_value']
            if target != 0:
                return abs((actual - target) / target) * 100
        return v or 0.0
    
    def acknowledge(self, acknowledged_by: str) -> None:
        """Acknowledge the violation."""
        self.acknowledged_at = datetime.utcnow()
        self.add_notification("acknowledged", acknowledged_by, "Violation acknowledged")
        self.updated_at = datetime.utcnow()
    
    def resolve(self, resolved_by: str, resolution_notes: str) -> None:
        """Mark violation as resolved."""
        self.is_resolved = True
        self.resolved_at = datetime.utcnow()
        
        if self.start_time:
            self.end_time = self.resolved_at
            duration = (self.resolved_at - self.start_time).total_seconds() / 60.0
            self.duration_minutes = duration
        
        self.resolution_actions.append(f"[{self.resolved_at.isoformat()}] {resolution_notes}")
        self.add_notification("resolved", resolved_by, "Violation resolved")
        self.updated_at = datetime.utcnow()
    
    def escalate(self, escalated_to: str, escalation_reason: str) -> None:
        """Escalate the violation to higher level."""
        self.escalation_level += 1
        self.escalated_to.append(escalated_to)
        
        escalation_info = {
            "level": self.escalation_level,
            "escalated_to": escalated_to,
            "reason": escalation_reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.add_notification("escalated", escalated_to, f"Escalated to level {self.escalation_level}: {escalation_reason}")
        self.updated_at = datetime.utcnow()
    
    def add_notification(self, notification_type: str, recipient: str, message: str) -> None:
        """Add a notification record."""
        notification = {
            "id": str(uuid4()),
            "type": notification_type,
            "recipient": recipient,
            "message": message,
            "sent_at": datetime.utcnow().isoformat()
        }
        self.notifications_sent.append(notification)
    
    def calculate_service_credits(self, credit_rate: float) -> float:
        """Calculate service credits for this violation."""
        if not self.duration_minutes:
            return 0.0
        
        # Simple calculation based on duration and rate
        credits = self.duration_minutes * credit_rate * (self.severity_multiplier())
        self.service_credits_due = credits
        self.updated_at = datetime.utcnow()
        
        return credits
    
    def severity_multiplier(self) -> float:
        """Get severity multiplier for calculations."""
        multipliers = {
            SLAViolationSeverity.CRITICAL: 3.0,
            SLAViolationSeverity.HIGH: 2.0,
            SLAViolationSeverity.MEDIUM: 1.5,
            SLAViolationSeverity.LOW: 1.0,
            SLAViolationSeverity.INFO: 0.5
        }
        return multipliers.get(self.severity, 1.0)
    
    def is_ongoing(self) -> bool:
        """Check if violation is still ongoing."""
        return not self.is_resolved and not self.end_time
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get violation summary information."""
        return {
            "id": str(self.id),
            "severity": self.severity,
            "violation_type": self.violation_type,
            "start_time": self.start_time.isoformat(),
            "duration_minutes": self.duration_minutes,
            "is_resolved": self.is_resolved,
            "is_ongoing": self.is_ongoing(),
            "deviation_percentage": self.deviation_percentage,
            "service_credits_due": self.service_credits_due,
            "escalation_level": self.escalation_level,
            "affected_services": self.affected_services,
            "business_impact": self.business_impact
        }