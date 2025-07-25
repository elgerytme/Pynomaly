"""
Advanced architectural patterns for sophisticated package interactions.

This module provides CQRS (Command Query Responsibility Segregation),
Event Sourcing, and other advanced patterns for complex business scenarios
that require more sophisticated interaction patterns beyond basic events.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Protocol, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from .events import DomainEvent, EventPriority


T = TypeVar('T')
TAggregate = TypeVar('TAggregate')
TCommand = TypeVar('TCommand')
TQuery = TypeVar('TQuery')
TResult = TypeVar('TResult')


class CommandResult(Enum):
    """Result status of command execution."""
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass
class Command:
    """Base class for commands in CQRS pattern."""
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.command_id:
            self.command_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()


@dataclass
class Query:
    """Base class for queries in CQRS pattern."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    pagination: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.query_id:
            self.query_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()


@dataclass
class CommandResponse:
    """Response from command execution."""
    command_id: str
    result: CommandResult
    aggregate_id: Optional[str] = None
    version: Optional[int] = None
    events: List[DomainEvent] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None


@dataclass
class QueryResponse(Generic[T]):
    """Response from query execution."""
    query_id: str
    data: Optional[T] = None
    total_count: Optional[int] = None
    page_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None


# =============================================================================
# CQRS Pattern Interfaces
# =============================================================================

class CommandHandler(Protocol[TCommand]):
    """Protocol for command handlers in CQRS pattern."""
    
    async def handle(self, command: TCommand) -> CommandResponse:
        """Handle a command and return response."""
        ...
    
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle the given command."""
        ...


class QueryHandler(Protocol[TQuery, TResult]):
    """Protocol for query handlers in CQRS pattern."""
    
    async def handle(self, query: TQuery) -> QueryResponse[TResult]:
        """Handle a query and return response."""
        ...
    
    def can_handle(self, query: Query) -> bool:
        """Check if this handler can handle the given query."""
        ...


class CommandBus(ABC):
    """Command bus for CQRS pattern."""
    
    @abstractmethod
    async def send(self, command: Command) -> CommandResponse:
        """Send a command for processing."""
        pass
    
    @abstractmethod
    def register_handler(self, command_type: type, handler: CommandHandler) -> None:
        """Register a command handler."""
        pass


class QueryBus(ABC):
    """Query bus for CQRS pattern."""
    
    @abstractmethod
    async def ask(self, query: Query) -> QueryResponse:
        """Execute a query and get response."""
        pass
    
    @abstractmethod
    def register_handler(self, query_type: type, handler: QueryHandler) -> None:
        """Register a query handler."""
        pass


# =============================================================================
# Event Sourcing Pattern
# =============================================================================

@dataclass
class EventStoreEvent:
    """Event as stored in event store."""
    event_id: str
    aggregate_id: str
    aggregate_type: str
    event_type: str
    event_data: Dict[str, Any]
    metadata: Dict[str, Any]
    version: int
    timestamp: datetime
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

    @classmethod
    def from_domain_event(
        cls, 
        event: DomainEvent, 
        aggregate_type: str, 
        version: int
    ) -> 'EventStoreEvent':
        """Create event store event from domain event."""
        return cls(
            event_id=event.event_id,
            aggregate_id=event.aggregate_id,
            aggregate_type=aggregate_type,
            event_type=event.event_type,
            event_data=event.__dict__.copy(),
            metadata=getattr(event, 'metadata', {}),
            version=version,
            timestamp=event.occurred_at,
            correlation_id=getattr(event, 'correlation_id', None),
            causation_id=getattr(event, 'causation_id', None)
        )

    def to_domain_event(self) -> DomainEvent:
        """Convert back to domain event."""
        # This would need to be implemented based on event type registry
        # For now, return a generic domain event
        return DomainEvent(
            event_id=self.event_id,
            event_type=self.event_type,
            aggregate_id=self.aggregate_id,
            occurred_at=self.timestamp,
            metadata=self.metadata
        )


class EventStore(ABC):
    """Event store interface for event sourcing."""
    
    @abstractmethod
    async def append_events(
        self, 
        aggregate_id: str, 
        events: List[DomainEvent], 
        expected_version: int
    ) -> None:
        """Append events to the store."""
        pass
    
    @abstractmethod
    async def get_events(
        self, 
        aggregate_id: str, 
        from_version: int = 0
    ) -> List[EventStoreEvent]:
        """Get events for an aggregate."""
        pass
    
    @abstractmethod
    async def get_events_by_type(
        self, 
        event_type: str, 
        from_timestamp: Optional[datetime] = None
    ) -> List[EventStoreEvent]:
        """Get all events of a specific type."""
        pass


class Aggregate(ABC):
    """Base class for aggregates in event sourcing."""
    
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self.uncommitted_events: List[DomainEvent] = []
    
    def apply_event(self, event: DomainEvent) -> None:
        """Apply an event to the aggregate."""
        self._when(event)
        self.version += 1
    
    def raise_event(self, event: DomainEvent) -> None:
        """Raise a new event."""
        event.aggregate_id = self.aggregate_id
        self.apply_event(event)
        self.uncommitted_events.append(event)
    
    def mark_events_as_committed(self) -> None:
        """Mark uncommitted events as committed."""
        self.uncommitted_events.clear()
    
    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Get uncommitted events."""
        return self.uncommitted_events.copy()
    
    @classmethod
    def from_history(cls, aggregate_id: str, events: List[DomainEvent]) -> 'Aggregate':
        """Recreate aggregate from event history."""
        instance = cls(aggregate_id)
        for event in events:
            instance.apply_event(event)
        instance.mark_events_as_committed()
        return instance
    
    @abstractmethod
    def _when(self, event: DomainEvent) -> None:
        """Apply event to aggregate state (to be implemented by subclasses)."""
        pass


class Repository(ABC, Generic[TAggregate]):
    """Repository interface for event-sourced aggregates."""
    
    @abstractmethod
    async def get_by_id(self, aggregate_id: str) -> Optional[TAggregate]:
        """Get aggregate by ID."""
        pass
    
    @abstractmethod
    async def save(self, aggregate: TAggregate) -> None:
        """Save aggregate."""
        pass


# =============================================================================
# Saga Pattern for Distributed Transactions
# =============================================================================

class SagaStatus(Enum):
    """Status of saga execution."""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    """Individual step in a saga."""
    step_id: str
    command: Command
    compensation_command: Optional[Command] = None
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class SagaState:
    """State of saga execution."""
    saga_id: str
    status: SagaStatus
    current_step: int
    steps: List[SagaStep]
    completed_steps: List[str] = field(default_factory=list)
    failed_step: Optional[str] = None
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SagaOrchestrator(ABC):
    """Orchestrator for saga pattern."""
    
    @abstractmethod
    async def start_saga(self, saga_id: str, steps: List[SagaStep]) -> SagaState:
        """Start a new saga."""
        pass
    
    @abstractmethod
    async def handle_step_completion(
        self, 
        saga_id: str, 
        step_id: str, 
        result: CommandResponse
    ) -> SagaState:
        """Handle completion of a saga step."""
        pass
    
    @abstractmethod
    async def handle_step_failure(
        self, 
        saga_id: str, 
        step_id: str, 
        error: str
    ) -> SagaState:
        """Handle failure of a saga step."""
        pass
    
    @abstractmethod
    async def get_saga_state(self, saga_id: str) -> Optional[SagaState]:
        """Get current saga state."""
        pass


# =============================================================================
# Projection and Read Model Patterns
# =============================================================================

class ReadModel(ABC):
    """Base class for read models in CQRS."""
    
    @abstractmethod
    async def handle_event(self, event: DomainEvent) -> None:
        """Handle domain event to update read model."""
        pass
    
    @abstractmethod
    def get_supported_events(self) -> List[type]:
        """Get list of event types this read model handles."""
        pass


class ProjectionManager(ABC):
    """Manager for projections and read models."""
    
    @abstractmethod
    async def register_projection(self, read_model: ReadModel) -> None:
        """Register a read model projection."""
        pass
    
    @abstractmethod
    async def rebuild_projection(self, read_model_type: type) -> None:
        """Rebuild a projection from event store."""
        pass
    
    @abstractmethod
    async def get_projection_status(self, read_model_type: type) -> Dict[str, Any]:
        """Get status of a projection."""
        pass


# =============================================================================
# Specific CQRS Commands and Queries for Package Interactions
# =============================================================================

@dataclass
class ProcessDataQualityCommand(Command):
    """Command to process data quality assessment."""
    dataset_id: str
    quality_rules: List[str]
    priority: str = "normal"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunAnomalyDetectionCommand(Command):
    """Command to run anomaly detection."""
    dataset_id: str
    algorithm: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_version: Optional[str] = None


@dataclass
class TrainModelCommand(Command):
    """Command to train a machine learning model."""
    dataset_id: str
    model_type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    target_accuracy: Optional[float] = None


@dataclass
class GetDataQualityReportQuery(Query):
    """Query to get data quality report."""
    dataset_id: str
    include_details: bool = True
    date_range: Optional[Dict[str, datetime]] = None


@dataclass
class GetAnomalyDetectionResultsQuery(Query):
    """Query to get anomaly detection results."""
    dataset_id: str
    algorithm: Optional[str] = None
    confidence_threshold: Optional[float] = None
    limit: int = 100


@dataclass
class GetModelPerformanceQuery(Query):
    """Query to get model performance metrics."""
    model_id: str
    metric_types: List[str] = field(default_factory=list)
    time_period: Optional[str] = None


# =============================================================================
# Advanced Event Types for Complex Workflows
# =============================================================================

@dataclass
class WorkflowStarted(DomainEvent):
    """Event indicating workflow has started."""
    workflow_id: str
    workflow_type: str
    initiator: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStepCompleted(DomainEvent):
    """Event indicating workflow step completion."""
    workflow_id: str
    step_id: str
    step_type: str
    output_data: Dict[str, Any] = field(default_factory=dict)
    next_step: Optional[str] = None


@dataclass
class WorkflowFailed(DomainEvent):
    """Event indicating workflow failure."""
    workflow_id: str
    failed_step: str
    error_message: str
    error_code: Optional[str] = None
    retry_count: int = 0


@dataclass
class WorkflowCompleted(DomainEvent):
    """Event indicating workflow completion."""
    workflow_id: str
    final_output: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float
    success_rate: float = 1.0


@dataclass
class CompensationRequired(DomainEvent):
    """Event indicating compensation is required (saga pattern)."""
    saga_id: str
    failed_step: str
    compensation_steps: List[str]
    reason: str


@dataclass
class CompensationCompleted(DomainEvent):
    """Event indicating compensation has been completed."""
    saga_id: str
    compensated_steps: List[str]
    final_state: str


# =============================================================================
# Advanced Configuration and Metadata
# =============================================================================

@dataclass
class CQRSConfiguration:
    """Configuration for CQRS pattern implementation."""
    enable_command_validation: bool = True
    enable_query_caching: bool = True
    command_timeout_seconds: int = 30
    query_timeout_seconds: int = 10
    max_retry_attempts: int = 3
    enable_distributed_tracing: bool = True
    command_store_type: str = "memory"  # memory, database, eventstore
    query_store_type: str = "memory"
    
    # Performance optimizations
    enable_command_batching: bool = False
    batch_size: int = 100
    batch_timeout_ms: int = 1000
    
    # Advanced features
    enable_saga_orchestration: bool = False
    enable_event_sourcing: bool = False
    projection_update_mode: str = "async"  # sync, async, eventual


@dataclass
class EventSourcingConfiguration:
    """Configuration for event sourcing implementation."""
    event_store_type: str = "memory"  # memory, file, database
    snapshot_frequency: int = 100  # Create snapshot every N events
    enable_snapshots: bool = True
    compression_enabled: bool = False
    encryption_enabled: bool = False
    
    # Performance settings
    batch_size: int = 1000
    connection_pool_size: int = 10
    read_timeout_seconds: int = 30
    write_timeout_seconds: int = 10
    
    # Consistency settings
    consistency_level: str = "strong"  # strong, eventual, session
    conflict_resolution: str = "last_write_wins"  # last_write_wins, merge, reject


# =============================================================================
# Integration Helpers
# =============================================================================

def create_workflow_command(
    workflow_type: str,
    parameters: Dict[str, Any],
    user_id: Optional[str] = None
) -> Command:
    """Helper to create workflow commands."""
    return Command(
        command_id=str(uuid.uuid4()),
        user_id=user_id,
        metadata={
            "workflow_type": workflow_type,
            "parameters": parameters,
            "created_via": "helper_function"
        }
    )


def create_data_query(
    entity_type: str,
    filters: Dict[str, Any],
    pagination: Optional[Dict[str, Any]] = None
) -> Query:
    """Helper to create data queries."""
    return Query(
        query_id=str(uuid.uuid4()),
        filters=filters,
        pagination=pagination or {"page": 1, "size": 50},
        metadata={
            "entity_type": entity_type,
            "created_via": "helper_function"
        }
    )


def create_saga_steps(commands_and_compensations: List[tuple]) -> List[SagaStep]:
    """Helper to create saga steps from command pairs."""
    steps = []
    for i, (command, compensation) in enumerate(commands_and_compensations):
        step = SagaStep(
            step_id=f"step_{i + 1}",
            command=command,
            compensation_command=compensation,
            timeout_seconds=30
        )
        steps.append(step)
    return steps