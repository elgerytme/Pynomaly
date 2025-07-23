"""Pipeline Schedule domain entity for pipeline scheduling and execution timing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class ScheduleType(str, Enum):
    """Type of pipeline schedule."""
    
    CRON = "cron"
    INTERVAL = "interval"
    EVENT_DRIVEN = "event_driven"
    MANUAL = "manual"
    DEPENDENCY = "dependency"
    CONDITIONAL = "conditional"


class ScheduleStatus(str, Enum):
    """Status of pipeline schedule."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    EXPIRED = "expired"
    ERROR = "error"


class TriggerType(str, Enum):
    """Type of schedule trigger."""
    
    TIME_BASED = "time_based"
    FILE_ARRIVAL = "file_arrival"
    DATA_CHANGE = "data_change"
    API_WEBHOOK = "api_webhook"
    EXTERNAL_SYSTEM = "external_system"
    PIPELINE_COMPLETION = "pipeline_completion"


@dataclass
class ScheduleTrigger:
    """Individual trigger for pipeline schedule."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    trigger_type: TriggerType = TriggerType.TIME_BASED
    
    # Trigger configuration
    config: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    # Time-based trigger settings
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    timezone: str = "UTC"
    
    # Event-based trigger settings
    event_source: Optional[str] = None
    event_pattern: Optional[str] = None
    webhook_url: Optional[str] = None
    
    # File-based trigger settings
    watch_path: Optional[str] = None
    file_pattern: Optional[str] = None
    
    # Dependency trigger settings
    depends_on_pipeline: Optional[UUID] = None
    dependency_status: Optional[str] = None
    
    # Execution tracking
    last_triggered_at: Optional[datetime] = None
    trigger_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate trigger after initialization."""
        if not self.name:
            raise ValueError("Trigger name cannot be empty")
        
        if self.trigger_type == TriggerType.TIME_BASED:
            if not self.cron_expression and not self.interval_seconds:
                raise ValueError("Time-based trigger requires cron expression or interval")
        
        if self.trigger_type == TriggerType.FILE_ARRIVAL:
            if not self.watch_path:
                raise ValueError("File arrival trigger requires watch path")
        
        if self.trigger_type == TriggerType.PIPELINE_COMPLETION:
            if not self.depends_on_pipeline:
                raise ValueError("Pipeline completion trigger requires dependency pipeline")
        
        if self.interval_seconds is not None and self.interval_seconds <= 0:
            raise ValueError("Interval must be positive")
    
    @property
    def success_rate(self) -> float:
        """Get trigger success rate."""
        if self.trigger_count == 0:
            return 0.0
        return ((self.trigger_count - self.error_count) / self.trigger_count) * 100
    
    @property
    def is_healthy(self) -> bool:
        """Check if trigger is healthy."""
        return self.is_active and self.error_count == 0
    
    def validate_cron_expression(self) -> bool:
        """Validate cron expression syntax."""
        if not self.cron_expression:
            return True
        
        # Basic cron validation - would be more comprehensive in real implementation
        parts = self.cron_expression.split()
        if len(parts) not in [5, 6]:  # Standard 5-part or 6-part with seconds
            return False
        
        # Check for common invalid patterns
        invalid_patterns = ["* * * * * *", ""]
        if self.cron_expression in invalid_patterns:
            return False
        
        return True
    
    def calculate_next_execution(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """Calculate next execution time based on trigger configuration."""
        if not self.is_active:
            return None
        
        base_time = from_time or datetime.utcnow()
        
        if self.trigger_type == TriggerType.TIME_BASED:
            if self.interval_seconds:
                return base_time + timedelta(seconds=self.interval_seconds)
            elif self.cron_expression and self.validate_cron_expression():
                # Simplified cron calculation - would use proper cron library
                # For demo, assume daily execution at midnight
                next_day = base_time.replace(hour=0, minute=0, second=0, microsecond=0)
                next_day += timedelta(days=1)
                return next_day
        
        return None
    
    def record_trigger(self, success: bool, error_message: Optional[str] = None) -> None:
        """Record a trigger execution."""
        self.trigger_count += 1
        self.last_triggered_at = datetime.utcnow()
        
        if not success:
            self.error_count += 1
            self.last_error = error_message
        else:
            self.last_error = None
        
        self.updated_at = self.last_triggered_at
    
    def activate(self) -> None:
        """Activate the trigger."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate the trigger."""
        self.is_active = False
        self.updated_at = datetime.utcnow()
    
    def reset_error_count(self) -> None:
        """Reset error count and last error."""
        self.error_count = 0
        self.last_error = None
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trigger to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "trigger_type": self.trigger_type.value,
            "config": self.config,
            "is_active": self.is_active,
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "timezone": self.timezone,
            "event_source": self.event_source,
            "event_pattern": self.event_pattern,
            "webhook_url": self.webhook_url,
            "watch_path": self.watch_path,
            "file_pattern": self.file_pattern,
            "depends_on_pipeline": str(self.depends_on_pipeline) if self.depends_on_pipeline else None,
            "dependency_status": self.dependency_status,
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "trigger_count": self.trigger_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "success_rate": self.success_rate,
            "is_healthy": self.is_healthy,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class PipelineSchedule:
    """Pipeline schedule domain entity for managing pipeline execution timing."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    pipeline_id: UUID = field(default_factory=uuid4)
    
    # Schedule configuration
    schedule_type: ScheduleType = ScheduleType.CRON
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    
    # Timing configuration
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    timezone: str = "UTC"
    
    # Execution settings
    max_concurrent_runs: int = 1
    allow_overlap: bool = False
    catchup: bool = True  # Run missed executions
    max_catchup_runs: int = 10
    
    # Retry configuration
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "retry_delay_seconds": 300,
        "exponential_backoff": True,
        "retry_on_failure_only": True
    })
    
    # Triggers
    triggers: List[ScheduleTrigger] = field(default_factory=list)
    
    # Execution tracking
    next_run_time: Optional[datetime] = None
    last_run_time: Optional[datetime] = None
    last_successful_run: Optional[datetime] = None
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    
    # Active executions
    active_runs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = ""
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate schedule after initialization."""
        if not self.name:
            raise ValueError("Schedule name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Schedule name cannot exceed 100 characters")
        
        if self.max_concurrent_runs <= 0:
            raise ValueError("Max concurrent runs must be positive")
        
        if self.max_catchup_runs <= 0:
            raise ValueError("Max catchup runs must be positive")
        
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
    
    @property
    def is_active(self) -> bool:
        """Check if schedule is active."""
        return self.status == ScheduleStatus.ACTIVE
    
    @property
    def is_expired(self) -> bool:
        """Check if schedule has expired."""
        if not self.end_date:
            return False
        return datetime.utcnow() > self.end_date
    
    @property
    def is_ready_to_run(self) -> bool:
        """Check if schedule is ready for execution."""
        if not self.is_active:
            return False
        
        if self.is_expired:
            return False
        
        if not self.allow_overlap and len(self.active_runs) >= self.max_concurrent_runs:
            return False
        
        return True
    
    @property
    def success_rate(self) -> float:
        """Get execution success rate."""
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100
    
    @property
    def current_active_runs(self) -> int:
        """Get current number of active runs."""
        return len(self.active_runs)
    
    @property
    def is_overdue(self) -> bool:
        """Check if schedule is overdue for execution."""
        if not self.next_run_time:
            return False
        return datetime.utcnow() > self.next_run_time
    
    def add_trigger(self, trigger: ScheduleTrigger) -> None:
        """Add a trigger to the schedule."""
        # Check for duplicate trigger names
        existing_names = [t.name for t in self.triggers]
        if trigger.name in existing_names:
            raise ValueError(f"Trigger with name '{trigger.name}' already exists")
        
        self.triggers.append(trigger)
        self.updated_at = datetime.utcnow()
    
    def remove_trigger(self, trigger_id: UUID) -> bool:
        """Remove a trigger from the schedule."""
        for i, trigger in enumerate(self.triggers):
            if trigger.id == trigger_id:
                self.triggers.pop(i)
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def get_trigger(self, trigger_id: UUID) -> Optional[ScheduleTrigger]:
        """Get a trigger by ID."""
        for trigger in self.triggers:
            if trigger.id == trigger_id:
                return trigger
        return None
    
    def get_trigger_by_name(self, name: str) -> Optional[ScheduleTrigger]:
        """Get a trigger by name."""
        for trigger in self.triggers:
            if trigger.name == name:
                return trigger
        return None
    
    def get_active_triggers(self) -> List[ScheduleTrigger]:
        """Get all active triggers."""
        return [t for t in self.triggers if t.is_active]
    
    def calculate_next_run_time(self) -> Optional[datetime]:
        """Calculate the next run time based on active triggers."""
        if not self.is_active:
            return None
        
        active_triggers = self.get_active_triggers()
        if not active_triggers:
            return None
        
        next_times = []
        for trigger in active_triggers:
            next_time = trigger.calculate_next_execution()
            if next_time:
                next_times.append(next_time)
        
        if not next_times:
            return None
        
        # Return the earliest next execution time
        earliest_time = min(next_times)
        
        # Check against schedule boundaries
        if self.start_date and earliest_time < self.start_date:
            earliest_time = self.start_date
        
        if self.end_date and earliest_time > self.end_date:
            return None
        
        return earliest_time
    
    def should_run_now(self) -> bool:
        """Check if the schedule should run now."""
        if not self.is_ready_to_run:
            return False
        
        if not self.next_run_time:
            return False
        
        return datetime.utcnow() >= self.next_run_time
    
    def start_run(self, execution_id: str, trigger_id: Optional[UUID] = None) -> None:
        """Start a new run execution."""
        if not self.is_ready_to_run:
            raise ValueError("Schedule is not ready to run")
        
        if not self.allow_overlap and self.current_active_runs >= self.max_concurrent_runs:
            raise ValueError("Maximum concurrent runs exceeded")
        
        run_info = {
            "execution_id": execution_id,
            "started_at": datetime.utcnow(),
            "trigger_id": str(trigger_id) if trigger_id else None,
            "status": "running"
        }
        
        self.active_runs.append(run_info)
        self.total_runs += 1
        self.last_run_time = run_info["started_at"]
        self.updated_at = run_info["started_at"]
        
        # Update next run time
        self.next_run_time = self.calculate_next_run_time()
    
    def complete_run(self, execution_id: str, success: bool, duration_seconds: float) -> None:
        """Complete a run execution."""
        # Find and remove the active run
        run_info = None
        for i, run in enumerate(self.active_runs):
            if run["execution_id"] == execution_id:
                run_info = self.active_runs.pop(i)
                break
        
        if not run_info:
            raise ValueError(f"Active run with execution_id {execution_id} not found")
        
        completion_time = datetime.utcnow()
        
        if success:
            self.successful_runs += 1
            self.last_successful_run = completion_time
        else:
            self.failed_runs += 1
        
        # Store completion record
        if "completed_runs" not in self.config:
            self.config["completed_runs"] = []
        
        completion_record = {
            "execution_id": execution_id,
            "started_at": run_info["started_at"].isoformat(),
            "completed_at": completion_time.isoformat(),
            "duration_seconds": duration_seconds,
            "success": success,
            "trigger_id": run_info.get("trigger_id")
        }
        
        self.config["completed_runs"].append(completion_record)
        
        # Keep only last 100 completion records
        if len(self.config["completed_runs"]) > 100:
            self.config["completed_runs"] = self.config["completed_runs"][-100:]
        
        self.updated_at = completion_time
    
    def cancel_run(self, execution_id: str) -> bool:
        """Cancel an active run."""
        for i, run in enumerate(self.active_runs):
            if run["execution_id"] == execution_id:
                cancelled_run = self.active_runs.pop(i)
                
                # Store cancellation record
                if "cancelled_runs" not in self.config:
                    self.config["cancelled_runs"] = []
                
                cancellation_record = {
                    "execution_id": execution_id,
                    "started_at": cancelled_run["started_at"].isoformat(),
                    "cancelled_at": datetime.utcnow().isoformat(),
                    "trigger_id": cancelled_run.get("trigger_id")
                }
                
                self.config["cancelled_runs"].append(cancellation_record)
                
                # Keep only last 50 cancellation records
                if len(self.config["cancelled_runs"]) > 50:
                    self.config["cancelled_runs"] = self.config["cancelled_runs"][-50:]
                
                self.updated_at = datetime.utcnow()
                return True
        
        return False
    
    def activate(self) -> None:
        """Activate the schedule."""
        self.status = ScheduleStatus.ACTIVE
        self.next_run_time = self.calculate_next_run_time()
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate the schedule."""
        self.status = ScheduleStatus.INACTIVE
        self.next_run_time = None
        self.updated_at = datetime.utcnow()
    
    def pause(self) -> None:
        """Pause the schedule."""
        if self.status != ScheduleStatus.ACTIVE:
            raise ValueError("Can only pause an active schedule")
        
        self.status = ScheduleStatus.PAUSED
        self.updated_at = datetime.utcnow()
    
    def resume(self) -> None:
        """Resume the schedule."""
        if self.status != ScheduleStatus.PAUSED:
            raise ValueError("Can only resume a paused schedule")
        
        self.status = ScheduleStatus.ACTIVE
        self.next_run_time = self.calculate_next_run_time()
        self.updated_at = datetime.utcnow()
    
    def mark_expired(self) -> None:
        """Mark the schedule as expired."""
        self.status = ScheduleStatus.EXPIRED
        self.next_run_time = None
        self.updated_at = datetime.utcnow()
    
    def extend_end_date(self, new_end_date: datetime) -> None:
        """Extend the schedule end date."""
        if new_end_date <= datetime.utcnow():
            raise ValueError("New end date must be in the future")
        
        if self.start_date and new_end_date <= self.start_date:
            raise ValueError("New end date must be after start date")
        
        self.end_date = new_end_date
        
        # Reactivate if was expired
        if self.status == ScheduleStatus.EXPIRED:
            self.status = ScheduleStatus.ACTIVE
            self.next_run_time = self.calculate_next_run_time()
        
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the schedule."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the schedule."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if schedule has a specific tag."""
        return tag in self.tags
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration key-value pair."""
        self.config[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get schedule summary information."""
        return {
            "id": str(self.id),
            "name": self.name,
            "pipeline_id": str(self.pipeline_id),
            "status": self.status.value,
            "schedule_type": self.schedule_type.value,
            "is_active": self.is_active,
            "is_ready_to_run": self.is_ready_to_run,
            "is_overdue": self.is_overdue,
            "next_run_time": self.next_run_time.isoformat() if self.next_run_time else None,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "active_runs": self.current_active_runs,
            "total_runs": self.total_runs,
            "success_rate": self.success_rate,
            "trigger_count": len(self.triggers),
            "active_trigger_count": len(self.get_active_triggers()),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schedule to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "pipeline_id": str(self.pipeline_id),
            "schedule_type": self.schedule_type.value,
            "status": self.status.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "timezone": self.timezone,
            "max_concurrent_runs": self.max_concurrent_runs,
            "allow_overlap": self.allow_overlap,
            "catchup": self.catchup,
            "max_catchup_runs": self.max_catchup_runs,
            "retry_policy": self.retry_policy,
            "triggers": [t.to_dict() for t in self.triggers],
            "next_run_time": self.next_run_time.isoformat() if self.next_run_time else None,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "last_successful_run": self.last_successful_run.isoformat() if self.last_successful_run else None,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "active_runs": self.current_active_runs,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
            "config": self.config,
            "tags": self.tags,
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "is_ready_to_run": self.is_ready_to_run,
            "success_rate": self.success_rate,
            "is_overdue": self.is_overdue,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"PipelineSchedule('{self.name}', {self.schedule_type.value}, status={self.status.value})"
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return (
            f"PipelineSchedule(id={self.id}, name='{self.name}', "
            f"pipeline_id={self.pipeline_id}, status={self.status.value})"
        )