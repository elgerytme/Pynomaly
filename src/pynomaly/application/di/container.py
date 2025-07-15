"""Application layer dependency injection container.

This module provides the dependency injection container specifically for the application layer,
implementing proper dependency inversion with protocol-based design.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Type

from dependency_injector import containers, providers

# Domain protocols - core business abstractions
from ...domain.protocols.detection_protocols import (
    AlgorithmFactoryProtocol,
    EnsembleAggregatorProtocol,
)
from ...domain.protocols.processing_protocols import (
    ConfigProtocol,
    ProcessorFactoryProtocol,
    TracingProtocol,
)

# Application protocols - extended interfaces for application-specific operations
from ..protocols.adapter_protocols import (
    ApplicationAlgorithmFactoryProtocol,
    ApplicationDataProcessorProtocol,
)
from ..protocols.repository_protocols import (
    ApplicationRepositoryProtocol,
    DatasetRepositoryProtocol,
    DetectorRepositoryProtocol,
    ModelRepositoryProtocol,
)
from ..protocols.service_protocols import (
    ApplicationCacheProtocol,
    ApplicationConfigProtocol,
    ApplicationMetricsProtocol,
    ApplicationSecurityProtocol,
)

# Application services that need dependency injection
from ..services.anomaly_detection_service import AnomalyDetectionService
from ..services.processing_orchestrator_service import ProcessingOrchestratorService

logger = logging.getLogger(__name__)


class InMemoryConfig:
    """Simple in-memory configuration implementation for ApplicationConfigProtocol."""
    
    def __init__(self, **settings: Any):
        self._settings = settings
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._settings[key] = value
    
    def update(self, **settings: Any) -> None:
        """Update multiple configuration values."""
        self._settings.update(settings)


class InMemoryCache:
    """Simple in-memory cache implementation for ApplicationCacheProtocol."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, Any] = {}
        self._max_size = max_size
    
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        return self._cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        if len(self._cache) >= self._max_size:
            # Simple LRU - remove first item
            self._cache.pop(next(iter(self._cache)), None)
        self._cache[key] = value
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return self._cache.pop(key, None) is not None
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()


class InMemoryMetrics:
    """Simple in-memory metrics implementation for ApplicationMetricsProtocol."""
    
    def __init__(self):
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
    
    def increment_counter(self, name: str, value: int = 1, **tags: Any) -> None:
        """Increment a counter metric."""
        self._counters[name] = self._counters.get(name, 0) + value
    
    def set_gauge(self, name: str, value: float, **tags: Any) -> None:
        """Set a gauge metric."""
        self._gauges[name] = value
    
    def record_histogram(self, name: str, value: float, **tags: Any) -> None:
        """Record a histogram metric."""
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)
    
    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics."""
        return {
            'counters': self._counters.copy(),
            'gauges': self._gauges.copy(),
            'histograms': {k: list(v) for k, v in self._histograms.items()},
        }


class InMemorySecurity:
    """Simple in-memory security implementation for ApplicationSecurityProtocol."""
    
    def __init__(self):
        self._permissions: dict[str, list[str]] = {}
    
    async def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has permission."""
        user_permissions = self._permissions.get(user_id, [])
        return permission in user_permissions
    
    async def grant_permission(self, user_id: str, permission: str) -> None:
        """Grant permission to user."""
        if user_id not in self._permissions:
            self._permissions[user_id] = []
        if permission not in self._permissions[user_id]:
            self._permissions[user_id].append(permission)
    
    async def revoke_permission(self, user_id: str, permission: str) -> None:
        """Revoke permission from user."""
        if user_id in self._permissions:
            try:
                self._permissions[user_id].remove(permission)
            except ValueError:
                pass
    
    def validate_input(self, data: str) -> str:
        """Validate and sanitize input data."""
        # Basic sanitization - remove potential XSS patterns
        dangerous_patterns = ['<script', '<iframe', 'javascript:', 'data:']
        clean_data = data
        for pattern in dangerous_patterns:
            clean_data = clean_data.replace(pattern, '')
        return clean_data


class MockRepository:
    """Mock repository implementation for testing and development."""
    
    def __init__(self):
        self._data: dict[str, Any] = {}
    
    async def save(self, entity: Any) -> None:
        """Save entity to repository."""
        entity_id = getattr(entity, 'id', str(id(entity)))
        self._data[str(entity_id)] = entity
    
    async def find_by_id(self, entity_id: Any) -> Any | None:
        """Find entity by ID."""
        return self._data.get(str(entity_id))
    
    async def find_all(self) -> list[Any]:
        """Find all entities."""
        return list(self._data.values())
    
    async def delete(self, entity_id: Any) -> bool:
        """Delete entity by ID."""
        return self._data.pop(str(entity_id), None) is not None


class MockAlgorithmFactory:
    """Mock algorithm factory for testing and development."""
    
    def create_algorithm(self, config: Any) -> Any:
        """Create algorithm adapter."""
        return MockAlgorithmAdapter()
    
    def get_available_algorithms(self) -> list[Any]:
        """Get available algorithms."""
        return []
    
    def validate_config(self, config: Any) -> bool:
        """Validate algorithm configuration."""
        return True


class MockAlgorithmAdapter:
    """Mock algorithm adapter for testing."""
    
    def fit(self, data: Any) -> None:
        """Fit algorithm to data."""
        pass
    
    def predict(self, data: Any) -> Any:
        """Predict using algorithm."""
        import numpy as np
        return np.array([0] * len(data)) if hasattr(data, '__len__') else np.array([0])
    
    def decision_function(self, data: Any) -> Any:
        """Get decision scores."""
        import numpy as np
        return np.array([0.5] * len(data)) if hasattr(data, '__len__') else np.array([0.5])


class MockProcessorFactory:
    """Mock processor factory for testing."""
    
    def create_stream_processor(self, config: Any) -> Any:
        """Create stream processor."""
        return MockStreamProcessor()
    
    def create_batch_processor(self, config: Any) -> Any:
        """Create batch processor."""
        return MockBatchProcessor()


class MockStreamProcessor:
    """Mock stream processor for testing."""
    
    async def start(self) -> None:
        """Start stream processing."""
        pass
    
    async def stop(self) -> None:
        """Stop stream processing."""
        pass
    
    def get_status(self) -> dict[str, Any]:
        """Get processor status."""
        return {'status': 'running', 'processed_records': 0, 'error_count': 0}
    
    async def add_test_record(self, data: dict[str, Any]) -> None:
        """Add test record for testing."""
        pass


class MockBatchProcessor:
    """Mock batch processor for testing."""
    
    async def submit_job(self, **kwargs: Any) -> str:
        """Submit batch job."""
        return "mock_job_id"
    
    async def cancel_job(self, job_id: str) -> None:
        """Cancel batch job."""
        pass
    
    async def shutdown(self) -> None:
        """Shutdown processor."""
        pass
    
    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get job status."""
        return {'job_id': job_id, 'status': 'completed', 'total_samples': 0, 'total_anomalies': 0}


class MockTracing:
    """Mock tracing for testing."""
    
    def trace(self, operation: str) -> Any:
        """Create trace span."""
        return MockSpan()


class MockSpan:
    """Mock trace span."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ApplicationContainer(containers.DeclarativeContainer):
    """Application layer dependency injection container.
    
    This container implements proper dependency inversion by injecting protocol
    abstractions into application services, enabling clean architecture compliance.
    """
    
    # Configuration
    config = providers.Singleton(
        InMemoryConfig,
        processing={'max_concurrent_sessions': 10, 'enable_auto_scaling': True}
    )
    
    # Infrastructure abstractions - these will be provided by infrastructure layer
    cache = providers.Singleton(InMemoryCache, max_size=1000)
    metrics = providers.Singleton(InMemoryMetrics)
    security = providers.Singleton(InMemorySecurity)
    
    # Repository abstractions - will be implemented by infrastructure layer
    dataset_repository = providers.Singleton(MockRepository)
    detector_repository = providers.Singleton(MockRepository)
    model_repository = providers.Singleton(MockRepository)
    
    # Algorithm factory abstraction - will be implemented by infrastructure layer
    algorithm_factory = providers.Singleton(MockAlgorithmFactory)
    
    # Processing abstractions - will be implemented by infrastructure layer
    processor_factory = providers.Singleton(MockProcessorFactory)
    tracing = providers.Singleton(MockTracing)
    
    # Application services with dependency injection
    anomaly_detection_service = providers.Singleton(
        AnomalyDetectionService,
        dataset_repository=dataset_repository,
        detector_repository=detector_repository,
        algorithm_factory=algorithm_factory,
    )
    
    processing_orchestrator_service = providers.Singleton(
        ProcessingOrchestratorService,
        processor_factory=processor_factory,
        tracing=tracing,
        config=config,
    )


class ProductionApplicationContainer(ApplicationContainer):
    """Production application container with real implementations.
    
    This extends the base container to provide actual infrastructure implementations
    instead of mocks, suitable for production use.
    """
    
    @classmethod
    def create_with_infrastructure(
        cls,
        dataset_repository: DatasetRepositoryProtocol,
        detector_repository: DetectorRepositoryProtocol,
        model_repository: ModelRepositoryProtocol,
        algorithm_factory: ApplicationAlgorithmFactoryProtocol,
        processor_factory: ProcessorFactoryProtocol,
        cache: ApplicationCacheProtocol | None = None,
        metrics: ApplicationMetricsProtocol | None = None,
        security: ApplicationSecurityProtocol | None = None,
        config: ApplicationConfigProtocol | None = None,
        tracing: TracingProtocol | None = None,
    ) -> 'ProductionApplicationContainer':
        """Create container with injected infrastructure implementations.
        
        Args:
            dataset_repository: Dataset repository implementation
            detector_repository: Detector repository implementation
            model_repository: Model repository implementation
            algorithm_factory: Algorithm factory implementation
            processor_factory: Processor factory implementation
            cache: Cache implementation (optional)
            metrics: Metrics implementation (optional)
            security: Security implementation (optional)
            config: Configuration implementation (optional)
            tracing: Tracing implementation (optional)
            
        Returns:
            Configured production container
        """
        container = cls()
        
        # Override providers with actual implementations
        container.dataset_repository.override(providers.Object(dataset_repository))
        container.detector_repository.override(providers.Object(detector_repository))
        container.model_repository.override(providers.Object(model_repository))
        container.algorithm_factory.override(providers.Object(algorithm_factory))
        container.processor_factory.override(providers.Object(processor_factory))
        
        # Override optional services if provided
        if cache:
            container.cache.override(providers.Object(cache))
        if metrics:
            container.metrics.override(providers.Object(metrics))
        if security:
            container.security.override(providers.Object(security))
        if config:
            container.config.override(providers.Object(config))
        if tracing:
            container.tracing.override(providers.Object(tracing))
        
        return container


def create_application_container(production: bool = False) -> ApplicationContainer:
    """Create application dependency injection container.
    
    Args:
        production: Whether to create production container (requires infrastructure setup)
        
    Returns:
        Configured application container
    """
    if production:
        # In production, this would be called by the infrastructure layer
        # after setting up all the concrete implementations
        logger.info("Creating production application container")
        return ProductionApplicationContainer()
    else:
        # Development/testing container with mocks
        logger.info("Creating development application container with mocks")
        return ApplicationContainer()


def wire_application_services(container: ApplicationContainer) -> None:
    """Wire application services for dependency injection.
    
    Args:
        container: Application container to wire
    """
    try:
        container.wire(
            modules=[
                'pynomaly.application.services',
                'pynomaly.presentation.api.endpoints',
                'pynomaly.presentation.cli.commands',
                'pynomaly.presentation.web.routes',
            ]
        )
        logger.info("Application services wired successfully")
    except Exception as e:
        logger.warning(f"Failed to wire application services: {e}")
        logger.warning("Some dependency injection features may not work")


# Global container instance for easy access
_container: ApplicationContainer | None = None


def get_application_container() -> ApplicationContainer:
    """Get global application container instance.
    
    Returns:
        Global application container
    """
    global _container
    if _container is None:
        _container = create_application_container()
        wire_application_services(_container)
    return _container


def reset_application_container() -> None:
    """Reset global application container (useful for testing)."""
    global _container
    if _container is not None:
        try:
            _container.unwire()
        except Exception:
            pass
        _container = None


def setup_production_container(
    dataset_repository: DatasetRepositoryProtocol,
    detector_repository: DetectorRepositoryProtocol,
    model_repository: ModelRepositoryProtocol,
    algorithm_factory: ApplicationAlgorithmFactoryProtocol,
    processor_factory: ProcessorFactoryProtocol,
    **optional_services: Any,
) -> ApplicationContainer:
    """Setup production application container with infrastructure implementations.
    
    This function should be called by the infrastructure layer during application startup
    to inject all the concrete implementations.
    
    Args:
        dataset_repository: Dataset repository implementation
        detector_repository: Detector repository implementation
        model_repository: Model repository implementation
        algorithm_factory: Algorithm factory implementation
        processor_factory: Processor factory implementation
        **optional_services: Optional service implementations
        
    Returns:
        Configured production container
    """
    global _container
    
    # Create production container with infrastructure implementations
    _container = ProductionApplicationContainer.create_with_infrastructure(
        dataset_repository=dataset_repository,
        detector_repository=detector_repository,
        model_repository=model_repository,
        algorithm_factory=algorithm_factory,
        processor_factory=processor_factory,
        **optional_services,
    )
    
    # Wire the container
    wire_application_services(_container)
    
    logger.info("Production application container setup complete")
    return _container