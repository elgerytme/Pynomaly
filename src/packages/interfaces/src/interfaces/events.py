"""
Domain Events for cross-package communication.

This module defines stable events that enable different domains to
communicate asynchronously while maintaining proper boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import UUID, uuid4

from .dto import (
    DetectionStatus, 
    DataQualityStatus, 
    ModelStatus,
    DetectionResult,
    DataQualityResult,
    ModelTrainingResult,
)


class EventPriority(Enum):
    """Priority levels for domain events."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DomainEvent(ABC):
    """Base class for all domain events."""
    event_id: str
    event_type: str
    aggregate_id: str
    occurred_at: datetime
    version: int = 1
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid4())
        if not self.occurred_at:
            self.occurred_at = datetime.utcnow()
        if not self.event_type:
            self.event_type = self.__class__.__name__


# Anomaly Detection Events
@dataclass
class AnomalyDetectionStarted(DomainEvent):
    """Event fired when anomaly detection begins."""
    dataset_id: str
    algorithm: str
    parameters: Dict[str, Any]
    request_id: str
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.dataset_id


@dataclass
class AnomalyDetected(DomainEvent):
    """Event fired when anomalies are detected."""
    dataset_id: str
    anomaly_count: int
    severity: str
    detection_result: DetectionResult
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.dataset_id
        if self.severity in ["high", "critical"]:
            self.priority = EventPriority.HIGH


@dataclass
class AnomalyDetectionCompleted(DomainEvent):
    """Event fired when anomaly detection completes."""
    dataset_id: str
    status: DetectionStatus
    anomaly_count: int
    execution_time_ms: int
    detection_result: DetectionResult
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.dataset_id


@dataclass
class AnomalyDetectionFailed(DomainEvent):
    """Event fired when anomaly detection fails."""
    dataset_id: str
    error_message: str
    error_code: str
    request_id: str
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.dataset_id
        self.priority = EventPriority.HIGH


# Data Quality Events
@dataclass
class DataQualityCheckStarted(DomainEvent):
    """Event fired when data quality check begins."""
    dataset_id: str
    quality_rules: List[str]
    request_id: str
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.dataset_id


@dataclass
class DataQualityCheckCompleted(DomainEvent):
    """Event fired when data quality check completes."""
    dataset_id: str
    status: DataQualityStatus
    overall_score: float
    issues_count: int
    quality_result: DataQualityResult
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.dataset_id
        if self.status == DataQualityStatus.FAILED or self.overall_score < 0.5:
            self.priority = EventPriority.HIGH


@dataclass
class DataQualityIssueFound(DomainEvent):
    """Event fired when data quality issues are found."""
    dataset_id: str
    issue_type: str
    severity: str
    affected_columns: List[str]
    issue_count: int
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.dataset_id
        if self.severity in ["high", "critical"]:
            self.priority = EventPriority.HIGH


# ML Model Events
@dataclass
class ModelTrainingStarted(DomainEvent):
    """Event fired when model training begins."""
    model_id: str
    model_type: str
    dataset_id: str
    experiment_id: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.model_id


@dataclass
class ModelTrainingCompleted(DomainEvent):
    """Event fired when model training completes."""
    model_id: str
    status: ModelStatus
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    training_result: ModelTrainingResult
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.model_id


@dataclass
class ModelDeployed(DomainEvent):
    """Event fired when model is deployed."""
    model_id: str
    deployment_id: str
    environment: str
    endpoint_url: str
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.model_id


@dataclass
class ModelPerformanceDegraded(DomainEvent):
    """Event fired when model performance degrades."""
    model_id: str
    deployment_id: str
    current_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    degradation_percentage: float
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.model_id
        self.priority = EventPriority.HIGH


# System Events
@dataclass
class DatasetUpdated(DomainEvent):
    """Event fired when dataset is updated."""
    dataset_id: str
    schema_changed: bool
    record_count: int
    previous_record_count: int
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.dataset_id


@dataclass
class SystemHealthChanged(DomainEvent):
    """Event fired when system health status changes."""
    component: str
    status: str
    previous_status: str
    metrics: Dict[str, float]
    
    def __post_init__(self):
        super().__post_init__()
        self.aggregate_id = self.component
        if self.status in ["unhealthy", "critical"]:
            self.priority = EventPriority.CRITICAL


# Event Bus Interface
T = TypeVar('T', bound=DomainEvent)


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle a domain event."""
        pass


class EventBus(ABC):
    """Abstract event bus for publishing and subscribing to events."""
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to the bus."""
        pass
    
    @abstractmethod
    def subscribe(self, event_type: Type[T], handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: Type[T], handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        pass


class InMemoryEventBus(EventBus):
    """Simple in-memory event bus implementation."""
    
    def __init__(self):
        self._handlers: Dict[Type[DomainEvent], List[EventHandler]] = {}
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to all registered handlers."""
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                # Log error but don't stop other handlers
                print(f"Error handling event {event.event_id}: {e}")
    
    def subscribe(self, event_type: Type[T], handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def unsubscribe(self, event_type: Type[T], handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass  # Handler not found