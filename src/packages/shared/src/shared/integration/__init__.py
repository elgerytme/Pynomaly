"""Cross-domain integration package.

This package provides comprehensive patterns and tools for secure, standardized
communication between domain packages while maintaining strict domain boundaries.
"""

from .cross_domain_patterns import (
    # Core message types
    CrossDomainMessage,
    IntegrationContext,
    IntegrationResult,
    IntegrationStatus,
    MessageType,
    MessagePriority,
    
    # Integration management
    CrossDomainIntegrationManager,
    DomainEventBus,
    DomainService,
    
    # Reliability patterns
    CircuitBreaker,
    RetryPolicy,
    
    # Distributed transactions
    SagaStep,
    SagaOrchestrator,
    
    # Anti-corruption layer
    AntiCorruptionLayer,
    
    # Utility functions
    get_integration_manager,
    send_domain_command,
    query_domain,
    publish_domain_event,
    
    # Serialization
    MessageSerializer,
    MessageValidator,
)

from .domain_adapters import (
    # Adapter types
    DomainAdapter,
    StandardDomainAdapter,
    EnrichmentAdapter,
    AggregationAdapter,
    CachingAdapter,
    
    # Data transformation
    DataTransformer,
    FieldMapping,
    ValidationRule,
    TransformationStrategy,
    
    # Registry
    DomainAdapterRegistry,
    get_adapter_registry,
    
    # Convenience functions
    register_standard_adapter,
    register_enrichment_adapter,
    
    # Protocols
    DomainEntity,
)

__all__ = [
    # Core message types
    "CrossDomainMessage",
    "IntegrationContext", 
    "IntegrationResult",
    "IntegrationStatus",
    "MessageType",
    "MessagePriority",
    
    # Integration management
    "CrossDomainIntegrationManager",
    "DomainEventBus",
    "DomainService",
    
    # Reliability patterns
    "CircuitBreaker",
    "RetryPolicy",
    
    # Distributed transactions
    "SagaStep",
    "SagaOrchestrator",
    
    # Anti-corruption layer
    "AntiCorruptionLayer",
    
    # Adapter types
    "DomainAdapter",
    "StandardDomainAdapter", 
    "EnrichmentAdapter",
    "AggregationAdapter",
    "CachingAdapter",
    
    # Data transformation
    "DataTransformer",
    "FieldMapping",
    "ValidationRule",
    "TransformationStrategy",
    
    # Registry
    "DomainAdapterRegistry",
    "get_adapter_registry",
    
    # Protocols
    "DomainEntity",
    "DomainService",
    
    # Utility functions
    "get_integration_manager",
    "send_domain_command",
    "query_domain", 
    "publish_domain_event",
    "register_standard_adapter",
    "register_enrichment_adapter",
    
    # Serialization
    "MessageSerializer",
    "MessageValidator",
]