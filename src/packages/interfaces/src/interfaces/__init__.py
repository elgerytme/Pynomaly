"""
Interfaces package for domain contracts and cross-domain communication.

This package defines stable interfaces that enable different domains to
communicate while maintaining proper boundaries and loose coupling.
"""

__version__ = "0.1.0"

from .dto import (
    # Base classes
    BaseDTO,
    DetectionStatus,
    DataQualityStatus,
    ModelStatus,
    
    # Anomaly Detection DTOs
    DetectionRequest,
    DetectionResult,
    
    # Data Quality DTOs
    DataQualityRequest,
    DataQualityResult,
    
    # ML Model DTOs
    ModelTrainingRequest,
    ModelTrainingResult,
    
    # Analytics DTOs
    AnalyticsRequest,
    AnalyticsResult,
    
    # System DTOs
    HealthCheckRequest,
    HealthCheckResult,
)

from .events import (
    # Base classes
    DomainEvent,
    EventHandler,
    EventBus,
    InMemoryEventBus,
    EventPriority,
    
    # Anomaly Detection Events
    AnomalyDetectionStarted,
    AnomalyDetected,
    AnomalyDetectionCompleted,
    AnomalyDetectionFailed,
    
    # Data Quality Events
    DataQualityCheckStarted,
    DataQualityCheckCompleted,
    DataQualityIssueFound,
    
    # ML Model Events
    ModelTrainingStarted,
    ModelTrainingCompleted,
    ModelDeployed,
    ModelPerformanceDegraded,
    
    # System Events
    DatasetUpdated,
    SystemHealthChanged,
)

from .patterns import (
    # Repository Pattern
    Repository,
    
    # Service Patterns
    Service,
    AntiCorruptionLayer,
    
    # CQRS Patterns
    QueryHandler,
    CommandHandler,
    MessageBus,
    
    # Transaction Pattern
    UnitOfWork,
    
    # Infrastructure Patterns
    Cache,
    HealthCheck,
    MetricsCollector,
    ConfigurationProvider,
)

__all__ = [
    # DTOs
    "DetectionRequest",
    "DetectionResult", 
    "ModelTrainingRequest",
    "ModelTrainingResult",
    "DataQualityRequest",
    "DataQualityResult",
    # Events
    "DomainEvent",
    "ModelTrainingCompleted", 
    "DataQualityCheckCompleted",
    "AnomalyDetected",
    # Patterns
    "Repository",
    "Service",
    "EventBus",
    "AntiCorruptionLayer",
]