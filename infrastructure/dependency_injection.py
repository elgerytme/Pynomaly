"""Dependency injection infrastructure for cross-package service integration."""

from typing import Dict, Type, TypeVar, Any, Optional
from abc import ABC

T = TypeVar('T')


class ServiceRegistry:
    """Central service registry for dependency injection across packages."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
    
    def register(self, interface: Type[T], implementation: T, singleton: bool = True) -> None:
        """Register a service implementation for an interface."""
        self._services[interface] = implementation
        if singleton:
            self._singletons[interface] = implementation
    
    def get(self, interface: Type[T]) -> T:
        """Get a service implementation for an interface."""
        if interface in self._singletons:
            return self._singletons[interface]
        
        if interface in self._services:
            implementation = self._services[interface]
            if callable(implementation):
                instance = implementation()
                self._singletons[interface] = instance
                return instance
            return implementation
        
        raise ValueError(f"No implementation registered for {interface}")
    
    def has(self, interface: Type[T]) -> bool:
        """Check if a service is registered."""
        return interface in self._services
    
    def unregister(self, interface: Type[T]) -> None:
        """Unregister a service."""
        self._services.pop(interface, None)
        self._singletons.pop(interface, None)


# Global service registry instance
_service_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry instance."""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry


def register_service(interface: Type[T], implementation: T, singleton: bool = True) -> None:
    """Convenience function to register a service."""
    get_service_registry().register(interface, implementation, singleton)


def get_service(interface: Type[T]) -> T:
    """Convenience function to get a service."""
    return get_service_registry().get(interface)


def configure_data_services():
    """Configure data services in the registry."""
    from shared.interfaces.data_profiling import (
        DataProfilingInterface, 
        PatternDiscoveryInterface, 
        StatisticalProfilingInterface
    )
    from shared.interfaces.data_quality import (
        DataQualityInterface,
        ValidationEngineInterface, 
        DataCleansingInterface,
        QualityMonitoringInterface,
        RemediationInterface
    )
    
    registry = get_service_registry()
    
    # Register data profiling services (lazy loading)
    def create_profiling_service():
        try:
            from src.packages.data.data_profiling.application.services.profiling_engine import ProfilingEngine
            from src.packages.data.data_profiling.application.services.profiling_config import ProfilingConfig
            config = ProfilingConfig()
            return ProfilingEngine(config)
        except ImportError:
            from infrastructure.adapters.mock_profiling_adapter import MockProfilingAdapter
            return MockProfilingAdapter()
    
    def create_pattern_discovery_service():
        try:
            from src.packages.data.data_profiling.application.services.pattern_discovery_service import PatternDiscoveryService
            return PatternDiscoveryService()
        except ImportError:
            from infrastructure.adapters.mock_pattern_adapter import MockPatternAdapter
            return MockPatternAdapter()
    
    def create_quality_service():
        try:
            from src.packages.data.data_quality.application.services.quality_assessment_service import QualityAssessmentService
            return QualityAssessmentService()
        except ImportError:
            from infrastructure.adapters.mock_quality_adapter import MockQualityAdapter
            return MockQualityAdapter()
    
    # Register services
    registry.register(DataProfilingInterface, create_profiling_service, singleton=True)
    registry.register(PatternDiscoveryInterface, create_pattern_discovery_service, singleton=True)  
    registry.register(DataQualityInterface, create_quality_service, singleton=True)


def configure_mlops_services():
    """Configure MLOps services in the registry."""
    from shared.interfaces.mlops import ExperimentTrackingInterface, ModelRegistryInterface
    
    registry = get_service_registry()
    
    def create_experiment_tracker():
        try:
            from src.packages.ai.mlops.core.application.services.ml_lifecycle_service import MLLifecycleService
            return MLLifecycleService()
        except ImportError:
            from infrastructure.adapters.mock_mlops_adapter import MockExperimentTracker
            return MockExperimentTracker()
    
    def create_model_registry():
        try:
            from src.packages.ai.mlops.core.domain.services.model_management_service import ModelManagementService
            return ModelManagementService()
        except ImportError:
            from infrastructure.adapters.mock_mlops_adapter import MockModelRegistry
            return MockModelRegistry()
    
    registry.register(ExperimentTrackingInterface, create_experiment_tracker, singleton=True)
    registry.register(ModelRegistryInterface, create_model_registry, singleton=True)


def initialize_service_registry():
    """Initialize the service registry with all services."""
    configure_data_services()
    configure_mlops_services()