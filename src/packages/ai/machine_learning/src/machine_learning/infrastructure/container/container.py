"""Dependency injection container for machine learning package.

This container manages the instantiation and configuration of all
dependencies following the hexagonal architecture pattern.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union
from dataclasses import dataclass

# Domain interfaces
from machine_learning.domain.interfaces.automl_operations import (
    AutoMLOptimizationPort,
    ModelSelectionPort,
    HyperparameterOptimizationPort,
)
from machine_learning.domain.interfaces.explainability_operations import (
    ExplainabilityPort,
    ModelInterpretabilityPort,
)
from machine_learning.domain.interfaces.monitoring_operations import (
    MonitoringPort,
    DistributedTracingPort,
    AlertingPort,
    HealthCheckPort,
)

# Domain services
from machine_learning.domain.services.refactored_automl_service import AutoMLService

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ContainerConfig:
    """Configuration for the dependency injection container."""
    
    # AutoML configuration
    enable_sklearn_automl: bool = True
    enable_optuna_optimization: bool = True
    
    # Explainability configuration
    enable_shap_explainer: bool = True
    enable_lime_explainer: bool = True
    
    # Monitoring configuration
    enable_distributed_tracing: bool = True
    tracing_backend: str = "local"  # "local", "jaeger", "zipkin"
    enable_prometheus_monitoring: bool = True
    
    # Alerting configuration
    enable_alerting: bool = True
    alert_backend: str = "local"  # "local", "pagerduty", "slack"
    
    # Health check configuration
    enable_health_checks: bool = True
    health_check_interval: int = 60
    
    # General configuration
    log_level: str = "INFO"
    environment: str = "development"  # "development", "staging", "production"


class Container:
    """Dependency injection container for machine learning package.
    
    This container implements the Service Locator pattern and manages
    all dependencies according to hexagonal architecture principles.
    """
    
    def __init__(self, config: Optional[ContainerConfig] = None):
        """Initialize the container with configuration.
        
        Args:
            config: Container configuration, uses defaults if None
        """
        self._config = config or ContainerConfig()
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, self._config.log_level))
        
        self._logger.info(f"Initializing ML container (env: {self._config.environment})")
        
        # Initialize core services
        self._configure_adapters()
        self._configure_domain_services()
    
    def _configure_adapters(self):
        """Configure infrastructure adapters based on configuration."""
        
        # Configure AutoML adapters
        if self._config.enable_sklearn_automl:
            self._configure_sklearn_automl_adapter()
        
        # Configure explainability adapters
        if self._config.enable_shap_explainer or self._config.enable_lime_explainer:
            self._configure_explainability_adapters()
        
        # Configure monitoring adapters
        if self._config.enable_distributed_tracing:
            self._configure_tracing_adapter()
        
        if self._config.enable_prometheus_monitoring:
            self._configure_monitoring_adapter()
        
        if self._config.enable_alerting:
            self._configure_alerting_adapter()
        
        if self._config.enable_health_checks:
            self._configure_health_check_adapter()
    
    def _configure_sklearn_automl_adapter(self):
        """Configure scikit-learn AutoML adapter."""
        try:
            from machine_learning.infrastructure.adapters.automl.sklearn_automl_adapter import (
                SklearnAutoMLAdapter
            )
            
            adapter = SklearnAutoMLAdapter()
            
            # Register for multiple interfaces
            self.register_singleton(AutoMLOptimizationPort, adapter)
            self.register_singleton(ModelSelectionPort, adapter)
            self.register_singleton(HyperparameterOptimizationPort, adapter)
            
            self._logger.info("Configured scikit-learn AutoML adapter")
            
        except ImportError as e:
            self._logger.warning(f"Could not configure scikit-learn adapter: {e}")
            self._configure_automl_stubs()
    
    def _configure_explainability_adapters(self):
        """Configure explainability adapters."""
        try:
            # This would import actual explainability adapters
            # For now, we'll use stubs
            self._configure_explainability_stubs()
            
        except ImportError as e:
            self._logger.warning(f"Could not configure explainability adapters: {e}")
            self._configure_explainability_stubs()
    
    def _configure_tracing_adapter(self):
        """Configure distributed tracing adapter."""
        try:
            from machine_learning.infrastructure.adapters.monitoring.distributed_tracing_adapter import (
                DistributedTracingAdapter
            )
            
            adapter = DistributedTracingAdapter(tracing_backend=self._config.tracing_backend)
            self.register_singleton(DistributedTracingPort, adapter)
            
            self._logger.info(f"Configured distributed tracing adapter (backend: {self._config.tracing_backend})")
            
        except ImportError as e:
            self._logger.warning(f"Could not configure tracing adapter: {e}")
            self._configure_monitoring_stubs()
    
    def _configure_monitoring_adapter(self):
        """Configure monitoring adapter."""
        try:
            # This would import actual monitoring adapters
            # For now, we'll use stubs
            self._configure_monitoring_stubs()
            
        except ImportError as e:
            self._logger.warning(f"Could not configure monitoring adapter: {e}")
            self._configure_monitoring_stubs()
    
    def _configure_alerting_adapter(self):
        """Configure alerting adapter."""
        try:
            # This would import actual alerting adapters
            # For now, we'll use stubs
            self._configure_alerting_stubs()
            
        except ImportError as e:
            self._logger.warning(f"Could not configure alerting adapter: {e}")
            self._configure_alerting_stubs()
    
    def _configure_health_check_adapter(self):
        """Configure health check adapter."""
        try:
            # This would import actual health check adapters
            # For now, we'll use stubs
            self._configure_health_check_stubs()
            
        except ImportError as e:
            self._logger.warning(f"Could not configure health check adapter: {e}")
            self._configure_health_check_stubs()
    
    def _configure_automl_stubs(self):
        """Configure AutoML stub implementations."""
        from machine_learning.infrastructure.adapters.stubs.automl_stubs import (
            AutoMLOptimizationStub,
            ModelSelectionStub,
            HyperparameterOptimizationStub,
        )
        
        automl_stub = AutoMLOptimizationStub()
        model_selection_stub = ModelSelectionStub()
        hyperopt_stub = HyperparameterOptimizationStub()
        
        self.register_singleton(AutoMLOptimizationPort, automl_stub)
        self.register_singleton(ModelSelectionPort, model_selection_stub)
        self.register_singleton(HyperparameterOptimizationPort, hyperopt_stub)
        
        self._logger.info("Configured AutoML stub implementations")
    
    def _configure_explainability_stubs(self):
        """Configure explainability stub implementations."""
        from machine_learning.infrastructure.adapters.stubs.explainability_stubs import (
            ExplainabilityStub,
            ModelInterpretabilityStub,
        )
        
        explainability_stub = ExplainabilityStub()
        interpretability_stub = ModelInterpretabilityStub()
        
        self.register_singleton(ExplainabilityPort, explainability_stub)
        self.register_singleton(ModelInterpretabilityPort, interpretability_stub)
        
        self._logger.info("Configured explainability stub implementations")
    
    def _configure_monitoring_stubs(self):
        """Configure monitoring stub implementations."""
        from machine_learning.infrastructure.adapters.stubs.monitoring_stubs import (
            MonitoringStub,
            DistributedTracingStub,
        )
        
        monitoring_stub = MonitoringStub()
        tracing_stub = DistributedTracingStub()
        
        # Only register if not already registered
        if not self.is_registered(MonitoringPort):
            self.register_singleton(MonitoringPort, monitoring_stub)
        if not self.is_registered(DistributedTracingPort):
            self.register_singleton(DistributedTracingPort, tracing_stub)
        
        self._logger.info("Configured monitoring stub implementations")
    
    def _configure_alerting_stubs(self):
        """Configure alerting stub implementations."""
        from machine_learning.infrastructure.adapters.stubs.monitoring_stubs import AlertingStub
        
        alerting_stub = AlertingStub()
        self.register_singleton(AlertingPort, alerting_stub)
        
        self._logger.info("Configured alerting stub implementations")
    
    def _configure_health_check_stubs(self):
        """Configure health check stub implementations."""
        from machine_learning.infrastructure.adapters.stubs.monitoring_stubs import HealthCheckStub
        
        health_check_stub = HealthCheckStub()
        self.register_singleton(HealthCheckPort, health_check_stub)
        
        self._logger.info("Configured health check stub implementations")
    
    def _configure_domain_services(self):
        """Configure domain services with their dependencies."""
        
        # Configure AutoML service
        automl_service = self._create_automl_service()
        self.register_singleton(AutoMLService, automl_service)
        
        self._logger.info("Configured domain services")
    
    def _create_automl_service(self) -> AutoMLService:
        """Create AutoML service with its dependencies."""
        from machine_learning.domain.services.refactored_automl_service import AutoMLService
        
        return AutoMLService(
            automl_port=self.get(AutoMLOptimizationPort),
            model_selection_port=self.get(ModelSelectionPort),
            monitoring_port=self.get(MonitoringPort) if self.is_registered(MonitoringPort) else None,
            tracing_port=self.get(DistributedTracingPort) if self.is_registered(DistributedTracingPort) else None,
        )
    
    def register_singleton(self, interface: Type[T], implementation: T) -> None:
        """Register a singleton service implementation.
        
        Args:
            interface: Interface type to register
            implementation: Implementation instance
        """
        self._singletons[interface] = implementation
        self._logger.debug(f"Registered singleton: {interface.__name__} -> {type(implementation).__name__}")
    
    def register_transient(self, name: str, factory: Any) -> None:
        """Register a transient service factory.
        
        Args:
            name: Service name
            factory: Factory function or class
        """
        self._services[name] = factory
        self._logger.debug(f"Registered transient service: {name}")
    
    def get(self, interface: Type[T]) -> T:
        """Get service instance by interface.
        
        Args:
            interface: Interface type to resolve
            
        Returns:
            Service implementation instance
            
        Raises:
            ValueError: If service is not registered
        """
        if interface in self._singletons:
            return self._singletons[interface]
        
        raise ValueError(f"Service not registered: {interface.__name__}")
    
    def get_by_name(self, name: str) -> Any:
        """Get service instance by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service is not registered
        """
        if name in self._services:
            factory = self._services[name]
            if callable(factory):
                return factory()
            return factory
        
        raise ValueError(f"Service not registered: {name}")
    
    def is_registered(self, interface: Type) -> bool:
        """Check if interface is registered.
        
        Args:
            interface: Interface type to check
            
        Returns:
            True if interface is registered
        """
        return interface in self._singletons
    
    def configure_ml_integration(
        self, 
        enable_sklearn: bool = True,
        enable_optuna: bool = True,
        sklearn_config: Optional[Dict[str, Any]] = None,
        optuna_config: Optional[Dict[str, Any]] = None
    ):
        """Configure machine learning library integrations.
        
        Args:
            enable_sklearn: Enable scikit-learn integration
            enable_optuna: Enable Optuna integration
            sklearn_config: Configuration for scikit-learn
            optuna_config: Configuration for Optuna
        """
        self._config.enable_sklearn_automl = enable_sklearn
        self._config.enable_optuna_optimization = enable_optuna
        
        # Reconfigure adapters with new settings
        self._configure_adapters()
        
        # Reconfigure domain services
        self._configure_domain_services()
        
        self._logger.info(
            f"Reconfigured ML integration (sklearn: {enable_sklearn}, optuna: {enable_optuna})"
        )
    
    def configure_monitoring_integration(
        self,
        enable_tracing: bool = True,
        tracing_backend: str = "local",
        enable_monitoring: bool = True,
        monitoring_config: Optional[Dict[str, Any]] = None
    ):
        """Configure monitoring and observability integrations.
        
        Args:
            enable_tracing: Enable distributed tracing
            tracing_backend: Tracing backend to use
            enable_monitoring: Enable metrics monitoring
            monitoring_config: Configuration for monitoring
        """
        self._config.enable_distributed_tracing = enable_tracing
        self._config.tracing_backend = tracing_backend
        self._config.enable_prometheus_monitoring = enable_monitoring
        
        # Reconfigure monitoring adapters
        if enable_tracing:
            self._configure_tracing_adapter()
        
        if enable_monitoring:
            self._configure_monitoring_adapter()
        
        # Reconfigure domain services
        self._configure_domain_services()
        
        self._logger.info(
            f"Reconfigured monitoring integration (tracing: {enable_tracing}, monitoring: {enable_monitoring})"
        )
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current container configuration.
        
        Returns:
            Configuration summary
        """
        return {
            "environment": self._config.environment,
            "automl": {
                "sklearn_enabled": self._config.enable_sklearn_automl,
                "optuna_enabled": self._config.enable_optuna_optimization,
            },
            "explainability": {
                "shap_enabled": self._config.enable_shap_explainer,
                "lime_enabled": self._config.enable_lime_explainer,
            },
            "monitoring": {
                "tracing_enabled": self._config.enable_distributed_tracing,
                "tracing_backend": self._config.tracing_backend,
                "prometheus_enabled": self._config.enable_prometheus_monitoring,
                "alerting_enabled": self._config.enable_alerting,
                "health_checks_enabled": self._config.enable_health_checks,
            },
            "registered_services": {
                "singletons": [interface.__name__ for interface in self._singletons.keys()],
                "transients": list(self._services.keys()),
            }
        }


# Global container instance
_container: Optional[Container] = None


def get_container(config: Optional[ContainerConfig] = None) -> Container:
    """Get the global container instance.
    
    Args:
        config: Optional configuration for container initialization
        
    Returns:
        Container instance
    """
    global _container
    if _container is None:
        _container = Container(config)
    return _container


def reset_container():
    """Reset the global container instance."""
    global _container
    _container = None