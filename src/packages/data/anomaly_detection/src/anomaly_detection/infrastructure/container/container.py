"""Dependency Injection Container.

This module implements a dependency injection container for the anomaly detection
package, following hexagonal architecture principles. It wires domain interfaces
with their concrete infrastructure implementations.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union
from abc import ABC, abstractmethod

# Domain interfaces
from anomaly_detection.domain.interfaces.ml_operations import (
    MLModelTrainingPort,
    MLModelRegistryPort,
    MLFeatureEngineeringPort,
    MLModelExplainabilityPort,
)
from anomaly_detection.domain.interfaces.mlops_operations import (
    MLOpsExperimentTrackingPort,
    MLOpsModelRegistryPort,
    MLOpsModelDeploymentPort,
    MLOpsModelMonitoringPort,
)
from anomaly_detection.domain.interfaces.analytics_operations import (
    AnalyticsABTestingPort,
    AnalyticsPerformancePort,
    AnalyticsReportingPort,
    AnalyticsAlertingPort,
)

# Infrastructure adapters
from anomaly_detection.infrastructure.adapters.ml.training_adapter import MachinelearningTrainingAdapter
from anomaly_detection.infrastructure.adapters.mlops.experiment_tracking_adapter import MLOpsExperimentTrackingAdapter
from anomaly_detection.infrastructure.adapters.mlops.model_registry_adapter import MLOpsModelRegistryAdapter

# Domain services
from anomaly_detection.domain.services.detection_service import DetectionService

T = TypeVar('T')


class ContainerError(Exception):
    """Exception raised by container operations."""
    pass


class DependencyNotFoundError(ContainerError):
    """Exception raised when a dependency is not found."""
    pass


class DependencyResolver(ABC):
    """Abstract base class for dependency resolvers."""
    
    @abstractmethod
    def resolve(self, container: 'Container') -> Any:
        """Resolve the dependency using the container."""
        pass


class SingletonResolver(DependencyResolver):
    """Resolver for singleton dependencies."""
    
    def __init__(self, factory_func, *args, **kwargs):
        self.factory_func = factory_func
        self.args = args
        self.kwargs = kwargs
        self._instance = None
    
    def resolve(self, container: 'Container') -> Any:
        if self._instance is None:
            # Resolve dependencies in args and kwargs
            resolved_args = [
                container.get(arg) if isinstance(arg, str) else arg
                for arg in self.args
            ]
            resolved_kwargs = {
                key: container.get(value) if isinstance(value, str) else value
                for key, value in self.kwargs.items()
            }
            self._instance = self.factory_func(*resolved_args, **resolved_kwargs)
        return self._instance


class TransientResolver(DependencyResolver):
    """Resolver for transient dependencies."""
    
    def __init__(self, factory_func, *args, **kwargs):
        self.factory_func = factory_func
        self.args = args
        self.kwargs = kwargs
    
    def resolve(self, container: 'Container') -> Any:
        # Resolve dependencies in args and kwargs
        resolved_args = [
            container.get(arg) if isinstance(arg, str) else arg
            for arg in self.args
        ]
        resolved_kwargs = {
            key: container.get(value) if isinstance(value, str) else value
            for key, value in self.kwargs.items()
        }
        return self.factory_func(*resolved_args, **resolved_kwargs)


class Container:
    """Dependency injection container for the anomaly detection package.
    
    This container follows hexagonal architecture principles by wiring
    domain interfaces with their concrete infrastructure implementations.
    It supports both singleton and transient lifetimes for dependencies.
    """
    
    def __init__(self):
        self._services: Dict[str, DependencyResolver] = {}
        self._logger = logging.getLogger(__name__)
        
        # Initialize container with default services
        self._register_default_services()
        
        self._logger.info("Container initialized successfully")
    
    def register_singleton(
        self, 
        interface: Union[str, Type], 
        implementation: Union[Type, Any], 
        *args, 
        **kwargs
    ) -> 'Container':
        """Register a singleton service.
        
        Args:
            interface: Interface type or string identifier
            implementation: Implementation class or factory function
            *args: Constructor arguments
            **kwargs: Constructor keyword arguments
            
        Returns:
            Self for method chaining
        """
        key = self._get_service_key(interface)
        resolver = SingletonResolver(implementation, *args, **kwargs)
        self._services[key] = resolver
        
        self._logger.debug(f"Registered singleton service: {key}")
        return self
    
    def register_transient(
        self, 
        interface: Union[str, Type], 
        implementation: Union[Type, Any], 
        *args, 
        **kwargs
    ) -> 'Container':
        """Register a transient service.
        
        Args:
            interface: Interface type or string identifier
            implementation: Implementation class or factory function
            *args: Constructor arguments
            **kwargs: Constructor keyword arguments
            
        Returns:
            Self for method chaining
        """
        key = self._get_service_key(interface)
        resolver = TransientResolver(implementation, *args, **kwargs)
        self._services[key] = resolver
        
        self._logger.debug(f"Registered transient service: {key}")
        return self
    
    def register_instance(self, interface: Union[str, Type], instance: Any) -> 'Container':
        """Register a specific instance.
        
        Args:
            interface: Interface type or string identifier
            instance: Instance to register
            
        Returns:
            Self for method chaining
        """
        key = self._get_service_key(interface)
        resolver = SingletonResolver(lambda: instance)
        self._services[key] = resolver
        
        self._logger.debug(f"Registered instance: {key}")
        return self
    
    def get(self, interface: Union[str, Type]) -> Any:
        """Get a service from the container.
        
        Args:
            interface: Interface type or string identifier
            
        Returns:
            Service instance
            
        Raises:
            DependencyNotFoundError: If service is not registered
        """
        key = self._get_service_key(interface)
        
        if key not in self._services:
            raise DependencyNotFoundError(f"Service not registered: {key}")
        
        try:
            resolver = self._services[key]
            return resolver.resolve(self)
        except Exception as e:
            self._logger.error(f"Failed to resolve service {key}: {str(e)}")
            raise ContainerError(f"Failed to resolve service {key}: {str(e)}") from e
    
    def has(self, interface: Union[str, Type]) -> bool:
        """Check if a service is registered.
        
        Args:
            interface: Interface type or string identifier
            
        Returns:
            True if service is registered, False otherwise
        """
        key = self._get_service_key(interface)
        return key in self._services
    
    def configure_ml_integration(
        self, 
        enable_machine_learning: bool = True,
        ml_config: Optional[Dict[str, Any]] = None
    ) -> 'Container':
        """Configure machine learning package integration.
        
        Args:
            enable_machine_learning: Whether to enable ML integration
            ml_config: Optional ML configuration
            
        Returns:
            Self for method chaining
        """
        if enable_machine_learning:
            try:
                # Import ML services dynamically
                from machine_learning.domain.services.automl_service import AutoMLService
                from machine_learning.application.use_cases.train_model import TrainModelUseCase
                from machine_learning.application.use_cases.evaluate_model import EvaluateModelUseCase
                from machine_learning.application.use_cases.automl_optimization import AutoMLOptimizationUseCase
                
                # Register ML services
                self.register_singleton("automl_service", AutoMLService, ml_config or {})
                self.register_singleton("train_model_use_case", TrainModelUseCase, "automl_service")
                self.register_singleton("evaluate_model_use_case", EvaluateModelUseCase, "automl_service")
                self.register_singleton("automl_optimization_use_case", AutoMLOptimizationUseCase, "automl_service")
                
                # Register ML adapter
                self.register_singleton(
                    MLModelTrainingPort,
                    MachinelearningTrainingAdapter,
                    "automl_service",
                    "train_model_use_case",
                    "evaluate_model_use_case",
                    "automl_optimization_use_case"
                )
                
                self._logger.info("Machine learning integration configured successfully")
                
            except ImportError as e:
                self._logger.warning(f"Machine learning package not available: {str(e)}")
                # Register stub implementation
                self._register_ml_stubs()
        else:
            self._register_ml_stubs()
        
        return self
    
    def configure_mlops_integration(
        self, 
        enable_mlops: bool = True,
        mlops_config: Optional[Dict[str, Any]] = None
    ) -> 'Container':
        """Configure MLOps package integration.
        
        Args:
            enable_mlops: Whether to enable MLOps integration
            mlops_config: Optional MLOps configuration
            
        Returns:
            Self for method chaining
        """
        if enable_mlops:
            try:
                # Import MLOps services dynamically
                from mlops.domain.services.experiment_tracking_service import ExperimentTrackingService
                from mlops.domain.services.model_management_service import ModelManagementService
                from mlops.application.use_cases.create_experiment_use_case import CreateExperimentUseCase
                from mlops.application.use_cases.run_experiment_use_case import RunExperimentUseCase
                from mlops.application.use_cases.create_model_use_case import CreateModelUseCase
                
                # Register MLOps services
                self.register_singleton("experiment_tracking_service", ExperimentTrackingService, mlops_config or {})
                self.register_singleton("model_management_service", ModelManagementService, mlops_config or {})
                self.register_singleton("create_experiment_use_case", CreateExperimentUseCase, "experiment_tracking_service")
                self.register_singleton("run_experiment_use_case", RunExperimentUseCase, "experiment_tracking_service")
                self.register_singleton("create_model_use_case", CreateModelUseCase, "model_management_service")
                
                # Register MLOps adapters
                self.register_singleton(
                    MLOpsExperimentTrackingPort,
                    MLOpsExperimentTrackingAdapter,
                    "experiment_tracking_service",
                    "create_experiment_use_case",
                    "run_experiment_use_case"
                )
                
                self.register_singleton(
                    MLOpsModelRegistryPort,
                    MLOpsModelRegistryAdapter,
                    "model_management_service",
                    "create_model_use_case"
                )
                
                self._logger.info("MLOps integration configured successfully")
                
            except ImportError as e:
                self._logger.warning(f"MLOps package not available: {str(e)}")
                # Register stub implementations
                self._register_mlops_stubs()
        else:
            self._register_mlops_stubs()
        
        return self
    
    def configure_domain_services(self) -> 'Container':
        """Configure domain services with proper dependencies.
        
        Returns:
            Self for method chaining
        """
        # Register domain services with their dependencies
        self.register_singleton(
            DetectionService,
            DetectionService,
            MLModelTrainingPort,  # Will be resolved from container
            MLOpsExperimentTrackingPort,  # Will be resolved from container
        )
        
        self._logger.info("Domain services configured successfully")
        return self
    
    def _register_default_services(self) -> None:
        """Register default services and configurations."""
        # This method can be extended to register common services
        pass
    
    def _register_ml_stubs(self) -> None:
        """Register stub implementations for ML services when package is not available."""
        from anomaly_detection.infrastructure.adapters.stubs.ml_stubs import MLTrainingStub
        
        self.register_singleton(MLModelTrainingPort, MLTrainingStub)
        self._logger.info("ML stub implementations registered")
    
    def _register_mlops_stubs(self) -> None:
        """Register stub implementations for MLOps services when package is not available."""
        from anomaly_detection.infrastructure.adapters.stubs.mlops_stubs import (
            MLOpsExperimentTrackingStub,
            MLOpsModelRegistryStub
        )
        
        self.register_singleton(MLOpsExperimentTrackingPort, MLOpsExperimentTrackingStub)
        self.register_singleton(MLOpsModelRegistryPort, MLOpsModelRegistryStub)
        self._logger.info("MLOps stub implementations registered")
    
    def _get_service_key(self, interface: Union[str, Type]) -> str:
        """Get string key for service registration.
        
        Args:
            interface: Interface type or string identifier
            
        Returns:
            String key for the service
        """
        if isinstance(interface, str):
            return interface
        elif hasattr(interface, '__name__'):
            return interface.__name__
        else:
            return str(interface)


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance.
    
    Returns:
        Global container instance
    """
    global _container
    if _container is None:
        _container = Container()
    return _container


def configure_container(
    enable_ml: bool = True,
    enable_mlops: bool = True,
    ml_config: Optional[Dict[str, Any]] = None,
    mlops_config: Optional[Dict[str, Any]] = None
) -> Container:
    """Configure the global container with integrations.
    
    Args:
        enable_ml: Whether to enable ML integration
        enable_mlops: Whether to enable MLOps integration
        ml_config: Optional ML configuration
        mlops_config: Optional MLOps configuration
        
    Returns:
        Configured container instance
    """
    container = get_container()
    
    container.configure_ml_integration(enable_ml, ml_config)
    container.configure_mlops_integration(enable_mlops, mlops_config)
    container.configure_domain_services()
    
    return container


def reset_container() -> None:
    """Reset the global container instance."""
    global _container
    _container = None