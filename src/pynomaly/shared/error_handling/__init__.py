"""Comprehensive error handling system for Pynomaly."""

from .unified_exceptions import (
    # Base exceptions
    PynamolyError,
    DomainError,
    ValidationError,
    InfrastructureError,
    ExternalServiceError,
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    PerformanceError,
    TimeoutError,
    ResourceExhaustionError,
    NetworkError,
    DataIntegrityError,
    
    # Enums
    ErrorSeverity,
    ErrorCategory,
    
    # Data classes
    ErrorContext,
    ErrorDetails,
    
    # Error codes
    ErrorCodes,
    
    # Factory functions
    create_validation_error,
    create_business_error,
    create_infrastructure_error,
    create_external_service_error,
    create_timeout_error,
)

from .resilience import (
    # Circuit breaker
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    
    # Retry
    RetryHandler,
    RetryConfig,
    
    # Bulkhead
    Bulkhead,
    BulkheadConfig,
    
    # Manager
    ResilienceManager,
    get_resilience_manager,
    
    # Decorators
    circuit_breaker,
    retry,
    bulkhead,
)

from .monitoring import (
    # Monitoring
    ErrorMonitor,
    ErrorMetrics,
    Alert,
    AlertLevel,
    AlertRule,
    
    # Functions
    get_error_monitor,
    track_error,
    start_error_monitoring,
    stop_error_monitoring,
)

from .recovery import (
    # Recovery
    RecoveryManager,
    RecoveryHandler,
    RecoveryConfig,
    RecoveryStrategy,
    RecoveryStatus,
    RecoveryAttempt,
    
    # Handlers
    CacheRecoveryHandler,
    FallbackRecoveryHandler,
    DegradeRecoveryHandler,
    DefaultRecoveryHandler,
    
    # Functions
    get_recovery_manager,
    attempt_recovery,
    recovery_context,
    recovery_decorator,
)

__all__ = [
    # Unified exceptions
    "PynamolyError",
    "DomainError",
    "ValidationError",
    "InfrastructureError",
    "ExternalServiceError",
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "PerformanceError",
    "TimeoutError",
    "ResourceExhaustionError",
    "NetworkError",
    "DataIntegrityError",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    "ErrorDetails",
    "ErrorCodes",
    "create_validation_error",
    "create_business_error",
    "create_infrastructure_error",
    "create_external_service_error",
    "create_timeout_error",
    
    # Resilience
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "RetryHandler",
    "RetryConfig",
    "Bulkhead",
    "BulkheadConfig",
    "ResilienceManager",
    "get_resilience_manager",
    "circuit_breaker",
    "retry",
    "bulkhead",
    
    # Monitoring
    "ErrorMonitor",
    "ErrorMetrics",
    "Alert",
    "AlertLevel",
    "AlertRule",
    "get_error_monitor",
    "track_error",
    "start_error_monitoring",
    "stop_error_monitoring",
    
    # Recovery
    "RecoveryManager",
    "RecoveryHandler",
    "RecoveryConfig",
    "RecoveryStrategy",
    "RecoveryStatus",
    "RecoveryAttempt",
    "CacheRecoveryHandler",
    "FallbackRecoveryHandler",
    "DegradeRecoveryHandler",
    "DefaultRecoveryHandler",
    "get_recovery_manager",
    "attempt_recovery",
    "recovery_context",
    "recovery_decorator",
]