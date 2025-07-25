"""
Modernized Dependency Injection Container using shared framework.

This module demonstrates how to migrate from the legacy DI container
to the new shared dependency injection framework while maintaining
all existing functionality.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar

# Import from shared framework
from shared import (
    DIContainer, LifecycleScope, get_container, configure_container,
    register_service, register_repository
)

# Import from interfaces for stable contracts
from interfaces.patterns import Repository, Service

# Domain interfaces (internal to anomaly_detection)
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

# Application services
from anomaly_detection.application.services.anomaly_detection_service import AnomalyDetectionApplicationService


logger = logging.getLogger(__name__)
T = TypeVar('T')


class AnomalyDetectionContainer:
    """
    Modernized container for anomaly detection using shared DI framework.
    
    This container replaces the legacy implementation with the new shared
    dependency injection framework while maintaining all existing functionality.
    """
    
    def __init__(self, container: Optional[DIContainer] = None):
        """Initialize with optional external container."""
        self.container = container or DIContainer()
        self._configured = False
    
    def configure(self) -> None:
        """Configure all dependencies using the new DI framework."""
        if self._configured:
            return
        
        # Configure infrastructure adapters
        self._configure_ml_adapters()
        self._configure_mlops_adapters()
        self._configure_analytics_adapters()
        
        # Configure domain services
        self._configure_domain_services()
        
        # Configure application services
        self._configure_application_services()
        
        self._configured = True
        logger.info("Anomaly detection container configured with shared DI framework")
    
    def _configure_ml_adapters(self) -> None:
        """Configure ML operation adapters."""
        # ML Training
        self.container.register_singleton(
            MLModelTrainingPort,
            MachinelearningTrainingAdapter
        )
        
        # ML Registry (if adapter exists)
        try:
            from anomaly_detection.infrastructure.adapters.ml.registry_adapter import MLModelRegistryAdapter
            self.container.register_singleton(
                MLModelRegistryPort,
                MLModelRegistryAdapter
            )
        except ImportError:
            logger.debug("ML Registry adapter not available")
        
        # Feature Engineering (if adapter exists)
        try:
            from anomaly_detection.infrastructure.adapters.ml.feature_adapter import FeatureEngineeringAdapter
            self.container.register_singleton(
                MLFeatureEngineeringPort,
                FeatureEngineeringAdapter
            )
        except ImportError:
            logger.debug("Feature Engineering adapter not available")
        
        # Model Explainability (if adapter exists)
        try:
            from anomaly_detection.infrastructure.adapters.ml.explainability_adapter import ExplainabilityAdapter
            self.container.register_singleton(
                MLModelExplainabilityPort,
                ExplainabilityAdapter
            )
        except ImportError:
            logger.debug("Explainability adapter not available")
    
    def _configure_mlops_adapters(self) -> None:
        """Configure MLOps operation adapters."""
        # Experiment Tracking
        self.container.register_singleton(
            MLOpsExperimentTrackingPort,
            MLOpsExperimentTrackingAdapter
        )
        
        # Model Registry
        self.container.register_singleton(
            MLOpsModelRegistryPort,
            MLOpsModelRegistryAdapter
        )
        
        # Model Deployment (if adapter exists)
        try:
            from anomaly_detection.infrastructure.adapters.mlops.deployment_adapter import MLOpsDeploymentAdapter
            self.container.register_singleton(
                MLOpsModelDeploymentPort,
                MLOpsDeploymentAdapter
            )
        except ImportError:
            logger.debug("MLOps Deployment adapter not available")
        
        # Model Monitoring (if adapter exists)
        try:
            from anomaly_detection.infrastructure.adapters.mlops.monitoring_adapter import MLOpsMonitoringAdapter
            self.container.register_singleton(
                MLOpsModelMonitoringPort,
                MLOpsMonitoringAdapter
            )
        except ImportError:
            logger.debug("MLOps Monitoring adapter not available")
    
    def _configure_analytics_adapters(self) -> None:
        """Configure analytics operation adapters."""
        # A/B Testing (if adapter exists)
        try:
            from anomaly_detection.infrastructure.adapters.analytics.ab_testing_adapter import ABTestingAdapter
            self.container.register_singleton(
                AnalyticsABTestingPort,
                ABTestingAdapter
            )
        except ImportError:
            logger.debug("A/B Testing adapter not available")
        
        # Performance Analytics (if adapter exists)
        try:
            from anomaly_detection.infrastructure.adapters.analytics.performance_adapter import PerformanceAdapter
            self.container.register_singleton(
                AnalyticsPerformancePort,
                PerformanceAdapter
            )
        except ImportError:
            logger.debug("Performance Analytics adapter not available")
        
        # Reporting (if adapter exists)
        try:
            from anomaly_detection.infrastructure.adapters.analytics.reporting_adapter import ReportingAdapter
            self.container.register_singleton(
                AnalyticsReportingPort,
                ReportingAdapter
            )
        except ImportError:
            logger.debug("Reporting adapter not available")
        
        # Alerting (if adapter exists)
        try:
            from anomaly_detection.infrastructure.adapters.analytics.alerting_adapter import AlertingAdapter
            self.container.register_singleton(
                AnalyticsAlertingPort,
                AlertingAdapter
            )
        except ImportError:
            logger.debug("Alerting adapter not available")
    
    def _configure_domain_services(self) -> None:
        """Configure domain services."""
        # Detection Service
        register_service(self.container, DetectionService, DetectionService)
        
        # Register additional domain services if they exist
        try:
            from anomaly_detection.domain.services.data_preprocessing_service import DataPreprocessingService
            register_service(self.container, DataPreprocessingService, DataPreprocessingService)
        except ImportError:
            logger.debug("Data Preprocessing service not available")
        
        try:
            from anomaly_detection.domain.services.model_training_service import ModelTrainingService
            register_service(self.container, ModelTrainingService, ModelTrainingService)
        except ImportError:
            logger.debug("Model Training service not available")
    
    def _configure_application_services(self) -> None:
        """Configure application services."""
        # Main application service
        register_service(self.container, AnomalyDetectionApplicationService, AnomalyDetectionApplicationService)
        
        # Register additional application services if they exist
        try:
            from anomaly_detection.application.services.streaming_service import StreamingDetectionService
            register_service(self.container, StreamingDetectionService, StreamingDetectionService)
        except ImportError:
            logger.debug("Streaming Detection service not available")
        
        try:
            from anomaly_detection.application.services.batch_service import BatchDetectionService
            register_service(self.container, BatchDetectionService, BatchDetectionService)
        except ImportError:
            logger.debug("Batch Detection service not available")
    
    def get(self, service_type: Type[T]) -> T:
        """Get a service from the container."""
        if not self._configured:
            self.configure()
        
        return self.container.resolve(service_type)
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return self.container.is_registered(service_type)
    
    def create_scope(self, scope_name: str):
        """Create a new dependency scope."""
        return self.container.create_scope(scope_name)


def create_anomaly_detection_container() -> AnomalyDetectionContainer:
    """Factory function to create a configured anomaly detection container."""
    container = AnomalyDetectionContainer()
    container.configure()
    return container


def configure_global_container() -> None:
    """Configure the global shared container with anomaly detection services."""
    def setup_anomaly_detection(container: DIContainer):
        # Create anomaly detection container and merge configurations
        ad_container = AnomalyDetectionContainer(container)
        ad_container.configure()
    
    configure_container(setup_anomaly_detection)


# Backward compatibility with legacy container
class LegacyContainerAdapter:
    """
    Adapter to provide backward compatibility with the legacy container interface.
    
    This allows existing code to continue working while gradually migrating
    to the new DI framework.
    """
    
    def __init__(self):
        self.modern_container = create_anomaly_detection_container()
    
    def get(self, service_name: str) -> Any:
        """Legacy get method that maps string names to types."""
        # Map legacy string names to new types
        service_mapping = {
            'detection_service': DetectionService,
            'ml_training': MLModelTrainingPort,
            'mlops_tracking': MLOpsExperimentTrackingPort,
            'mlops_registry': MLOpsModelRegistryPort,
            'application_service': AnomalyDetectionApplicationService,
        }
        
        service_type = service_mapping.get(service_name)
        if service_type:
            return self.modern_container.get(service_type)
        else:
            raise ValueError(f"Unknown service: {service_name}")
    
    def register_singleton(self, name: str, factory_func, *args, **kwargs):
        """Legacy registration method."""
        logger.warning(f"Legacy registration for {name}. Consider using modern DI framework.")
        # For backward compatibility, we could implement this, but it's better to migrate


# Global instance for backward compatibility
_legacy_container = None


def get_legacy_container() -> LegacyContainerAdapter:
    """Get the legacy container adapter for backward compatibility."""
    global _legacy_container
    if _legacy_container is None:
        _legacy_container = LegacyContainerAdapter()
    return _legacy_container