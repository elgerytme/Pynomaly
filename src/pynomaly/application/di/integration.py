"""Integration module for wiring application services with infrastructure.

This module provides utilities for integrating the application layer with infrastructure
implementations, demonstrating how dependency injection enables clean architecture.
"""

from __future__ import annotations

import logging
from typing import Any

from ...domain.protocols.detection_protocols import AlgorithmFactoryProtocol
from ...domain.protocols.processing_protocols import (
    ConfigProtocol,
    ProcessorFactoryProtocol,
    TracingProtocol,
)
from ..protocols.adapter_protocols import ApplicationAlgorithmFactoryProtocol
from ..protocols.repository_protocols import (
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
from .container import ApplicationContainer, setup_production_container

logger = logging.getLogger(__name__)


class InfrastructureIntegrator:
    """Integrates application services with infrastructure implementations."""
    
    def __init__(self):
        self._container: ApplicationContainer | None = None
        self._is_production = False
    
    def setup_with_infrastructure(
        self,
        # Required infrastructure implementations
        dataset_repository: DatasetRepositoryProtocol,
        detector_repository: DetectorRepositoryProtocol,
        algorithm_factory: ApplicationAlgorithmFactoryProtocol,
        processor_factory: ProcessorFactoryProtocol,
        
        # Optional infrastructure implementations
        model_repository: ModelRepositoryProtocol | None = None,
        cache: ApplicationCacheProtocol | None = None,
        metrics: ApplicationMetricsProtocol | None = None,
        security: ApplicationSecurityProtocol | None = None,
        config: ApplicationConfigProtocol | None = None,
        tracing: TracingProtocol | None = None,
    ) -> ApplicationContainer:
        """Setup application container with infrastructure implementations.
        
        Args:
            dataset_repository: Dataset repository implementation
            detector_repository: Detector repository implementation
            algorithm_factory: Algorithm factory implementation
            processor_factory: Processor factory implementation
            model_repository: Model repository implementation (optional)
            cache: Cache implementation (optional)
            metrics: Metrics implementation (optional)
            security: Security implementation (optional)
            config: Configuration implementation (optional)
            tracing: Tracing implementation (optional)
            
        Returns:
            Configured application container
        """
        # Create mock model repository if not provided
        if model_repository is None:
            from .container import MockRepository
            model_repository = MockRepository()
        
        # Setup production container with infrastructure
        self._container = setup_production_container(
            dataset_repository=dataset_repository,
            detector_repository=detector_repository,
            model_repository=model_repository,
            algorithm_factory=algorithm_factory,
            processor_factory=processor_factory,
            cache=cache,
            metrics=metrics,
            security=security,
            config=config,
            tracing=tracing,
        )
        
        self._is_production = True
        logger.info("Application layer integrated with infrastructure successfully")
        return self._container
    
    def get_container(self) -> ApplicationContainer:
        """Get the current application container.
        
        Returns:
            Application container
            
        Raises:
            RuntimeError: If container not setup
        """
        if self._container is None:
            raise RuntimeError("Container not setup. Call setup_with_infrastructure first.")
        return self._container
    
    def is_production_mode(self) -> bool:
        """Check if running in production mode with real infrastructure.
        
        Returns:
            True if production mode, False if development/mock mode
        """
        return self._is_production
    
    def reset(self) -> None:
        """Reset the integrator (useful for testing)."""
        if self._container:
            try:
                self._container.unwire()
            except Exception:
                pass
        self._container = None
        self._is_production = False


# Global integrator instance
_integrator = InfrastructureIntegrator()


def integrate_with_infrastructure(
    dataset_repository: DatasetRepositoryProtocol,
    detector_repository: DetectorRepositoryProtocol,
    algorithm_factory: ApplicationAlgorithmFactoryProtocol,
    processor_factory: ProcessorFactoryProtocol,
    **optional_services: Any,
) -> ApplicationContainer:
    """Global function to integrate application with infrastructure.
    
    This is the main entry point for setting up dependency injection
    between application and infrastructure layers.
    
    Args:
        dataset_repository: Dataset repository implementation
        detector_repository: Detector repository implementation
        algorithm_factory: Algorithm factory implementation
        processor_factory: Processor factory implementation
        **optional_services: Optional service implementations
        
    Returns:
        Configured application container
    """
    return _integrator.setup_with_infrastructure(
        dataset_repository=dataset_repository,
        detector_repository=detector_repository,
        algorithm_factory=algorithm_factory,
        processor_factory=processor_factory,
        **optional_services,
    )


def get_integrated_container() -> ApplicationContainer:
    """Get the integrated application container.
    
    Returns:
        Application container
        
    Raises:
        RuntimeError: If integration not setup
    """
    return _integrator.get_container()


def is_production_mode() -> bool:
    """Check if running in production mode.
    
    Returns:
        True if production mode, False if development mode
    """
    return _integrator.is_production_mode()


def reset_integration() -> None:
    """Reset integration (useful for testing)."""
    _integrator.reset()


class ApplicationServiceFactory:
    """Factory for creating application services with dependency injection."""
    
    def __init__(self, container: ApplicationContainer):
        self.container = container
    
    def create_anomaly_detection_service(self):
        """Create anomaly detection service with injected dependencies."""
        return self.container.anomaly_detection_service()
    
    def create_processing_orchestrator_service(self):
        """Create processing orchestrator service with injected dependencies."""
        return self.container.processing_orchestrator_service()
    
    def get_dataset_repository(self):
        """Get dataset repository."""
        return self.container.dataset_repository()
    
    def get_detector_repository(self):
        """Get detector repository."""
        return self.container.detector_repository()
    
    def get_algorithm_factory(self):
        """Get algorithm factory."""
        return self.container.algorithm_factory()


def create_service_factory(container: ApplicationContainer | None = None) -> ApplicationServiceFactory:
    """Create service factory with dependency injection.
    
    Args:
        container: Application container (uses global if not provided)
        
    Returns:
        Service factory
    """
    if container is None:
        container = get_integrated_container()
    return ApplicationServiceFactory(container)


# Example usage and demonstration
def demonstrate_dependency_injection():
    """Demonstrate dependency injection in action.
    
    This function shows how to set up and use dependency injection
    for application services.
    """
    # Step 1: Create mock infrastructure implementations
    from .container import MockRepository, MockAlgorithmFactory, MockProcessorFactory
    
    dataset_repo = MockRepository()
    detector_repo = MockRepository()
    algorithm_factory = MockAlgorithmFactory()
    processor_factory = MockProcessorFactory()
    
    # Step 2: Integrate application with infrastructure
    container = integrate_with_infrastructure(
        dataset_repository=dataset_repo,
        detector_repository=detector_repo,
        algorithm_factory=algorithm_factory,
        processor_factory=processor_factory,
    )
    
    # Step 3: Create services using dependency injection
    factory = create_service_factory(container)
    
    # Step 4: Use the services
    anomaly_service = factory.create_anomaly_detection_service()
    orchestrator_service = factory.create_processing_orchestrator_service()
    
    logger.info("Dependency injection demonstration complete")
    logger.info(f"Anomaly detection service: {type(anomaly_service).__name__}")
    logger.info(f"Processing orchestrator service: {type(orchestrator_service).__name__}")
    
    return container, factory