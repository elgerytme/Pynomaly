"""Dependency injection container for MLOps services."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar, get_type_hints
from pathlib import Path

# Domain interfaces
from mlops.domain.interfaces.experiment_tracking_operations import (
    ExperimentTrackingPort,
    ExperimentRunPort,
    ArtifactManagementPort,
    ExperimentAnalysisPort,
    MetricsTrackingPort,
    ExperimentSearchPort
)
from mlops.domain.interfaces.model_registry_operations import (
    ModelRegistryPort,
    ModelLifecyclePort,
    ModelDeploymentPort,
    ModelStoragePort,
    ModelVersioningPort,
    ModelSearchPort
)
from mlops.domain.interfaces.mlops_monitoring_operations import (
    ModelPerformanceMonitoringPort,
    InfrastructureMonitoringPort,
    DataQualityMonitoringPort,
    DataDriftMonitoringPort,
    AlertingPort,
    HealthCheckPort
)
from mlops.domain.interfaces.service_discovery_operations import (
    ServiceRegistryPort,
    ServiceDiscoveryPort,
    HealthCheckPort as ServiceHealthCheckPort,
    LoadBalancerPort
)
from mlops.domain.interfaces.configuration_management_operations import (
    ConfigurationProviderPort,
    ConfigurationWatcherPort,
    ConfigurationValidatorPort,
    SecretManagementPort,
    EnvironmentConfigurationPort
)

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Configuration for the dependency injection container."""
    
    # Experiment tracking configuration
    enable_file_experiment_tracking: bool = True
    enable_mlflow_tracking: bool = False
    enable_wandb_tracking: bool = False
    experiment_storage_path: str = "experiments"
    
    # Model registry configuration  
    enable_local_model_registry: bool = True
    enable_mlflow_registry: bool = False
    enable_s3_model_storage: bool = False
    model_storage_path: str = "models"
    
    # Monitoring configuration
    enable_prometheus_monitoring: bool = False
    enable_datadog_monitoring: bool = False
    enable_local_monitoring: bool = True
    monitoring_config: Dict[str, Any] = None
    
    # Service discovery configuration
    enable_file_service_discovery: bool = True
    enable_consul_service_discovery: bool = False
    enable_etcd_service_discovery: bool = False
    service_discovery_path: str = "service_discovery"
    
    # Configuration management
    enable_file_configuration: bool = True
    enable_consul_configuration: bool = False
    enable_vault_secrets: bool = False
    configuration_path: str = "configuration"
    secrets_path: str = "secrets"
    
    # Database configuration
    enable_postgresql: bool = False
    enable_sqlite: bool = True
    database_url: str = "sqlite:///mlops.db"
    
    # External services configuration
    enable_kubernetes_deployment: bool = False
    enable_docker_deployment: bool = True
    
    # Environment settings
    environment: str = "development"
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.monitoring_config is None:
            self.monitoring_config = {
                "metrics_retention_days": 30,
                "alert_retention_days": 90,
                "health_check_interval": 60
            }


class Container:
    """Dependency injection container for MLOps services."""
    
    _instance: Optional['Container'] = None
    
    def __init__(self, config: Optional[ContainerConfig] = None):
        """Initialize the container with configuration.
        
        Args:
            config: Container configuration
        """
        self._config = config or ContainerConfig()
        self._singletons: Dict[Type, Any] = {}
        self._configure_adapters()
        self._configure_domain_services()
        
        logger.info(f"MLOps container initialized for {self._config.environment} environment")
    
    def _configure_adapters(self):
        """Configure infrastructure adapters based on configuration."""
        self._configure_experiment_tracking_adapters()
        self._configure_model_registry_adapters()
        self._configure_monitoring_adapters()
        self._configure_storage_adapters()
        self._configure_service_discovery_adapters()
        self._configure_configuration_management_adapters()
    
    def _configure_experiment_tracking_adapters(self):
        """Configure experiment tracking adapters."""
        # File-based experiment tracking (default)
        if self._config.enable_file_experiment_tracking:
            try:
                from mlops.infrastructure.adapters.experiment_tracking.file_experiment_adapter import (
                    FileExperimentTrackingAdapter,
                    FileExperimentRunAdapter,
                    FileArtifactManagementAdapter,
                    FileExperimentAnalysisAdapter,
                    FileMetricsTrackingAdapter,
                    FileExperimentSearchAdapter
                )
                
                storage_path = Path(self._config.experiment_storage_path)
                
                self._singletons[ExperimentTrackingPort] = FileExperimentTrackingAdapter(storage_path)
                self._singletons[ExperimentRunPort] = FileExperimentRunAdapter(storage_path)
                self._singletons[ArtifactManagementPort] = FileArtifactManagementAdapter(storage_path)
                self._singletons[ExperimentAnalysisPort] = FileExperimentAnalysisAdapter(storage_path)
                self._singletons[MetricsTrackingPort] = FileMetricsTrackingAdapter(storage_path)
                self._singletons[ExperimentSearchPort] = FileExperimentSearchAdapter(storage_path)
                
                logger.info("File-based experiment tracking adapters configured")
                return
                
            except ImportError:
                logger.warning("File experiment tracking adapter not available, falling back to stubs")
        
        # MLflow integration
        if self._config.enable_mlflow_tracking:
            try:
                from mlops.infrastructure.adapters.experiment_tracking.mlflow_adapter import (
                    MLflowExperimentTrackingAdapter,
                    MLflowExperimentRunAdapter,
                    MLflowArtifactManagementAdapter
                )
                
                self._singletons[ExperimentTrackingPort] = MLflowExperimentTrackingAdapter()
                self._singletons[ExperimentRunPort] = MLflowExperimentRunAdapter()
                self._singletons[ArtifactManagementPort] = MLflowArtifactManagementAdapter()
                
                logger.info("MLflow experiment tracking adapters configured")
                return
                
            except ImportError:
                logger.warning("MLflow not available, falling back to stubs")
        
        # Fallback to stub implementations
        self._configure_experiment_tracking_stubs()
    
    def _configure_model_registry_adapters(self):
        """Configure model registry adapters."""
        # Local model registry (default)
        if self._config.enable_local_model_registry:
            try:
                from mlops.infrastructure.adapters.model_registry.local_registry_adapter import (
                    LocalModelRegistryAdapter,
                    LocalModelLifecycleAdapter,
                    LocalModelDeploymentAdapter,
                    LocalModelStorageAdapter,
                    LocalModelVersioningAdapter,
                    LocalModelSearchAdapter
                )
                
                storage_path = Path(self._config.model_storage_path)
                database_url = self._config.database_url
                
                self._singletons[ModelRegistryPort] = LocalModelRegistryAdapter(database_url, storage_path)
                self._singletons[ModelLifecyclePort] = LocalModelLifecycleAdapter(database_url)
                self._singletons[ModelDeploymentPort] = LocalModelDeploymentAdapter()
                self._singletons[ModelStoragePort] = LocalModelStorageAdapter(storage_path)
                self._singletons[ModelVersioningPort] = LocalModelVersioningAdapter(database_url)
                self._singletons[ModelSearchPort] = LocalModelSearchAdapter(database_url)
                
                logger.info("Local model registry adapters configured")
                return
                
            except ImportError:
                logger.warning("Local model registry adapter not available, falling back to stubs")
        
        # MLflow model registry
        if self._config.enable_mlflow_registry:
            try:
                from mlops.infrastructure.adapters.model_registry.mlflow_registry_adapter import (
                    MLflowModelRegistryAdapter,
                    MLflowModelLifecycleAdapter
                )
                
                self._singletons[ModelRegistryPort] = MLflowModelRegistryAdapter()
                self._singletons[ModelLifecyclePort] = MLflowModelLifecycleAdapter()
                
                logger.info("MLflow model registry adapters configured")
                return
                
            except ImportError:
                logger.warning("MLflow not available, falling back to stubs")
        
        # Fallback to stub implementations
        self._configure_model_registry_stubs()
    
    def _configure_monitoring_adapters(self):
        """Configure monitoring adapters."""
        # Prometheus monitoring
        if self._config.enable_prometheus_monitoring:
            try:
                from mlops.infrastructure.adapters.monitoring.prometheus_adapter import (
                    PrometheusModelPerformanceAdapter,
                    PrometheusInfrastructureAdapter,
                    PrometheusAlertingAdapter
                )
                
                self._singletons[ModelPerformanceMonitoringPort] = PrometheusModelPerformanceAdapter()
                self._singletons[InfrastructureMonitoringPort] = PrometheusInfrastructureAdapter()
                self._singletons[AlertingPort] = PrometheusAlertingAdapter()
                
                logger.info("Prometheus monitoring adapters configured")
                
            except ImportError:
                logger.warning("Prometheus adapter not available, falling back to local monitoring")
        
        # Local monitoring (default fallback)
        if self._config.enable_local_monitoring:
            try:
                from mlops.infrastructure.adapters.monitoring.local_monitoring_adapter import (
                    LocalModelPerformanceAdapter,
                    LocalInfrastructureAdapter,
                    LocalDataQualityAdapter,
                    LocalDataDriftAdapter,
                    LocalAlertingAdapter,
                    LocalHealthCheckAdapter
                )
                
                self._singletons[ModelPerformanceMonitoringPort] = LocalModelPerformanceAdapter(
                    self._config.monitoring_config
                )
                self._singletons[InfrastructureMonitoringPort] = LocalInfrastructureAdapter(
                    self._config.monitoring_config
                )
                self._singletons[DataQualityMonitoringPort] = LocalDataQualityAdapter(
                    self._config.monitoring_config
                )
                self._singletons[DataDriftMonitoringPort] = LocalDataDriftAdapter(
                    self._config.monitoring_config
                )
                self._singletons[AlertingPort] = LocalAlertingAdapter(
                    self._config.monitoring_config
                )
                self._singletons[HealthCheckPort] = LocalHealthCheckAdapter(
                    self._config.monitoring_config
                )
                
                logger.info("Local monitoring adapters configured")
                return
                
            except ImportError:
                logger.warning("Local monitoring adapter not available, falling back to stubs")
        
        # Fallback to stub implementations
        self._configure_monitoring_stubs()
    
    def _configure_service_discovery_adapters(self):
        """Configure service discovery adapters."""
        # File-based service discovery (default)
        if self._config.enable_file_service_discovery:
            try:
                from mlops.infrastructure.adapters.file_based.service_discovery_adapters import (
                    FileBasedServiceRegistry,
                    FileBasedServiceDiscovery,
                    FileBasedHealthCheck,
                    FileBasedLoadBalancer
                )
                
                discovery_path = Path(self._config.service_discovery_path)
                
                # Create service registry
                service_registry = FileBasedServiceRegistry(str(discovery_path / "registry"))
                self._singletons[ServiceRegistryPort] = service_registry
                
                # Create service discovery
                service_discovery = FileBasedServiceDiscovery(service_registry)
                self._singletons[ServiceDiscoveryPort] = service_discovery
                
                # Create health check
                health_check = FileBasedHealthCheck(str(discovery_path / "health"))
                self._singletons[ServiceHealthCheckPort] = health_check
                
                # Create load balancer
                load_balancer = FileBasedLoadBalancer(service_discovery, str(discovery_path / "load_balancer"))
                self._singletons[LoadBalancerPort] = load_balancer
                
                logger.info("File-based service discovery adapters configured")
                return
                
            except ImportError:
                logger.warning("File service discovery adapter not available, falling back to stubs")
        
        # Fallback to stub implementations
        self._configure_service_discovery_stubs()
    
    def _configure_configuration_management_adapters(self):
        """Configure configuration management adapters."""
        # File-based configuration management (default)
        if self._config.enable_file_configuration:
            try:
                from mlops.infrastructure.adapters.file_based.configuration_management_adapters import (
                    FileBasedConfigurationProvider,
                    FileBasedConfigurationWatcher,
                    FileBasedConfigurationValidator,
                    FileBasedSecretManagement,
                    FileBasedEnvironmentConfiguration
                )
                
                config_path = Path(self._config.configuration_path)
                secrets_path = Path(self._config.secrets_path)
                
                # Create configuration provider
                config_provider = FileBasedConfigurationProvider(str(config_path))
                self._singletons[ConfigurationProviderPort] = config_provider
                
                # Create configuration watcher
                config_watcher = FileBasedConfigurationWatcher(config_provider)
                self._singletons[ConfigurationWatcherPort] = config_watcher
                
                # Create configuration validator
                config_validator = FileBasedConfigurationValidator(str(config_path / "validation"))
                self._singletons[ConfigurationValidatorPort] = config_validator
                
                # Create secret management
                secret_management = FileBasedSecretManagement(str(secrets_path))
                self._singletons[SecretManagementPort] = secret_management
                
                # Create environment configuration
                env_config = FileBasedEnvironmentConfiguration(str(config_path / "environments"))
                self._singletons[EnvironmentConfigurationPort] = env_config
                
                logger.info("File-based configuration management adapters configured")
                return
                
            except ImportError:
                logger.warning("File configuration management adapter not available, falling back to stubs")
        
        # Fallback to stub implementations
        self._configure_configuration_management_stubs()
    
    def _configure_storage_adapters(self):
        """Configure storage adapters."""
        # S3 storage
        if self._config.enable_s3_model_storage:
            try:
                from mlops.infrastructure.adapters.storage.s3_storage_adapter import S3ModelStorageAdapter
                
                self._singletons[ModelStoragePort] = S3ModelStorageAdapter()
                logger.info("S3 storage adapter configured")
                return
                
            except ImportError:
                logger.warning("S3 storage adapter not available, using local storage")
    
    def _configure_experiment_tracking_stubs(self):
        """Configure experiment tracking stub implementations."""
        from mlops.infrastructure.adapters.stubs.experiment_tracking_stubs import (
            ExperimentTrackingStub,
            ExperimentRunStub,
            ArtifactManagementStub,
            ExperimentAnalysisStub,
            MetricsTrackingStub,
            ExperimentSearchStub
        )
        
        self._singletons[ExperimentTrackingPort] = ExperimentTrackingStub()
        self._singletons[ExperimentRunPort] = ExperimentRunStub()
        self._singletons[ArtifactManagementPort] = ArtifactManagementStub()
        self._singletons[ExperimentAnalysisPort] = ExperimentAnalysisStub()
        self._singletons[MetricsTrackingPort] = MetricsTrackingStub()
        self._singletons[ExperimentSearchPort] = ExperimentSearchStub()
        
        logger.info("Experiment tracking stubs configured")
    
    def _configure_model_registry_stubs(self):
        """Configure model registry stub implementations."""
        from mlops.infrastructure.adapters.stubs.model_registry_stubs import (
            ModelRegistryStub,
            ModelLifecycleStub,
            ModelDeploymentStub,
            ModelStorageStub,
            ModelVersioningStub,
            ModelSearchStub
        )
        
        self._singletons[ModelRegistryPort] = ModelRegistryStub()
        self._singletons[ModelLifecyclePort] = ModelLifecycleStub()
        self._singletons[ModelDeploymentPort] = ModelDeploymentStub()
        self._singletons[ModelStoragePort] = ModelStorageStub()
        self._singletons[ModelVersioningPort] = ModelVersioningStub()
        self._singletons[ModelSearchPort] = ModelSearchStub()
        
        logger.info("Model registry stubs configured")
    
    def _configure_monitoring_stubs(self):
        """Configure monitoring stub implementations."""
        from mlops.infrastructure.adapters.stubs.monitoring_stubs import (
            ModelPerformanceMonitoringStub,
            InfrastructureMonitoringStub,
            DataQualityMonitoringStub,
            DataDriftMonitoringStub,
            AlertingStub,
            HealthCheckStub
        )
        
        self._singletons[ModelPerformanceMonitoringPort] = ModelPerformanceMonitoringStub()
        self._singletons[InfrastructureMonitoringPort] = InfrastructureMonitoringStub()
        self._singletons[DataQualityMonitoringPort] = DataQualityMonitoringStub()
        self._singletons[DataDriftMonitoringPort] = DataDriftMonitoringStub()
        self._singletons[AlertingPort] = AlertingStub()
        self._singletons[HealthCheckPort] = HealthCheckStub()
        
        logger.info("Monitoring stubs configured")
    
    def _configure_service_discovery_stubs(self):
        """Configure service discovery stub implementations."""
        from mlops.infrastructure.adapters.stubs.service_discovery_stubs import (
            ServiceRegistryStub,
            ServiceDiscoveryStub,
            ServiceHealthCheckStub,
            LoadBalancerStub
        )
        
        self._singletons[ServiceRegistryPort] = ServiceRegistryStub()
        self._singletons[ServiceDiscoveryPort] = ServiceDiscoveryStub()
        self._singletons[ServiceHealthCheckPort] = ServiceHealthCheckStub()
        self._singletons[LoadBalancerPort] = LoadBalancerStub()
        
        logger.info("Service discovery stubs configured")
    
    def _configure_configuration_management_stubs(self):
        """Configure configuration management stub implementations."""
        from mlops.infrastructure.adapters.stubs.configuration_management_stubs import (
            ConfigurationProviderStub,
            ConfigurationWatcherStub,
            ConfigurationValidatorStub,
            SecretManagementStub,
            EnvironmentConfigurationStub
        )
        
        self._singletons[ConfigurationProviderPort] = ConfigurationProviderStub()
        self._singletons[ConfigurationWatcherPort] = ConfigurationWatcherStub()
        self._singletons[ConfigurationValidatorPort] = ConfigurationValidatorStub()
        self._singletons[SecretManagementPort] = SecretManagementStub()
        self._singletons[EnvironmentConfigurationPort] = EnvironmentConfigurationStub()
        
        logger.info("Configuration management stubs configured")
    
    def _configure_domain_services(self):
        """Configure domain services with dependency injection."""
        try:
            from mlops.domain.services.refactored_experiment_service import ExperimentService
            from mlops.domain.services.refactored_model_registry_service import ModelRegistryService
            from mlops.domain.services.refactored_monitoring_service import MonitoringService
            from mlops.domain.services.service_discovery_service import ServiceDiscoveryService
            from mlops.domain.services.configuration_management_service import ConfigurationManagementService
            
            # Experiment service
            experiment_service = ExperimentService(
                experiment_tracking_port=self.get(ExperimentTrackingPort),
                experiment_run_port=self.get(ExperimentRunPort),
                artifact_management_port=self.get(ArtifactManagementPort),
                experiment_analysis_port=self.get(ExperimentAnalysisPort),
                metrics_tracking_port=self.get(MetricsTrackingPort),
                experiment_search_port=self.get(ExperimentSearchPort)
            )
            self._singletons[ExperimentService] = experiment_service
            
            # Model registry service
            model_registry_service = ModelRegistryService(
                model_registry_port=self.get(ModelRegistryPort),
                model_lifecycle_port=self.get(ModelLifecyclePort),
                model_deployment_port=self.get(ModelDeploymentPort),
                model_storage_port=self.get(ModelStoragePort),
                model_versioning_port=self.get(ModelVersioningPort),
                model_search_port=self.get(ModelSearchPort)
            )
            self._singletons[ModelRegistryService] = model_registry_service
            
            # Monitoring service
            monitoring_service = MonitoringService(
                performance_monitoring_port=self.get(ModelPerformanceMonitoringPort),
                infrastructure_monitoring_port=self.get(InfrastructureMonitoringPort),
                data_quality_monitoring_port=self.get(DataQualityMonitoringPort),
                data_drift_monitoring_port=self.get(DataDriftMonitoringPort),
                alerting_port=self.get(AlertingPort),
                health_check_port=self.get(HealthCheckPort)
            )
            self._singletons[MonitoringService] = monitoring_service
            
            # Service discovery service
            service_discovery_service = ServiceDiscoveryService(
                service_registry_port=self.get(ServiceRegistryPort),
                service_discovery_port=self.get(ServiceDiscoveryPort),
                health_check_port=self.get(ServiceHealthCheckPort),
                load_balancer_port=self.get(LoadBalancerPort)
            )
            self._singletons[ServiceDiscoveryService] = service_discovery_service
            
            # Configuration management service
            configuration_management_service = ConfigurationManagementService(
                configuration_provider_port=self.get(ConfigurationProviderPort),
                configuration_watcher_port=self.get(ConfigurationWatcherPort),
                configuration_validator_port=self.get(ConfigurationValidatorPort),
                secret_management_port=self.get(SecretManagementPort),
                environment_configuration_port=self.get(EnvironmentConfigurationPort)
            )
            self._singletons[ConfigurationManagementService] = configuration_management_service
            
            logger.info("Domain services configured")
            
        except ImportError as e:
            logger.warning(f"Could not configure domain services: {e}")
    
    def get(self, interface: Type[T]) -> T:
        """Get a service instance by interface type.
        
        Args:
            interface: The interface type to retrieve
            
        Returns:
            Service instance implementing the interface
            
        Raises:
            ValueError: If service is not registered
        """
        if interface in self._singletons:
            return self._singletons[interface]
        
        raise ValueError(f"Service not registered: {interface.__name__}")
    
    def is_registered(self, interface: Type) -> bool:
        """Check if a service is registered.
        
        Args:
            interface: The interface type to check
            
        Returns:
            True if service is registered
        """
        return interface in self._singletons
    
    def register_singleton(self, interface: Type[T], implementation: T):
        """Register a singleton service.
        
        Args:
            interface: The interface type
            implementation: The implementation instance
        """
        self._singletons[interface] = implementation
        logger.debug(f"Registered singleton: {interface.__name__}")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the container configuration.
        
        Returns:
            Configuration summary
        """
        return {
            "environment": self._config.environment,
            "experiment_tracking": {
                "file_enabled": self._config.enable_file_experiment_tracking,
                "mlflow_enabled": self._config.enable_mlflow_tracking,
                "wandb_enabled": self._config.enable_wandb_tracking,
                "storage_path": self._config.experiment_storage_path
            },
            "model_registry": {
                "local_enabled": self._config.enable_local_model_registry,
                "mlflow_enabled": self._config.enable_mlflow_registry,
                "s3_enabled": self._config.enable_s3_model_storage,
                "storage_path": self._config.model_storage_path
            },
            "monitoring": {
                "prometheus_enabled": self._config.enable_prometheus_monitoring,
                "datadog_enabled": self._config.enable_datadog_monitoring,
                "local_enabled": self._config.enable_local_monitoring,
                "config": self._config.monitoring_config
            },
            "database": {
                "postgresql_enabled": self._config.enable_postgresql,
                "sqlite_enabled": self._config.enable_sqlite,
                "url": self._config.database_url
            },
            "deployment": {
                "kubernetes_enabled": self._config.enable_kubernetes_deployment,
                "docker_enabled": self._config.enable_docker_deployment
            },
            "registered_services": {
                "singletons": [service.__name__ for service in self._singletons.keys()],
                "count": len(self._singletons)
            }
        }
    
    def configure_experiment_tracking(
        self,
        enable_file: Optional[bool] = None,
        enable_mlflow: Optional[bool] = None,
        enable_wandb: Optional[bool] = None,
        storage_path: Optional[str] = None
    ):
        """Reconfigure experiment tracking at runtime.
        
        Args:
            enable_file: Enable file-based tracking
            enable_mlflow: Enable MLflow tracking
            enable_wandb: Enable Weights & Biases tracking
            storage_path: Storage path for experiments
        """
        if enable_file is not None:
            self._config.enable_file_experiment_tracking = enable_file
        if enable_mlflow is not None:
            self._config.enable_mlflow_tracking = enable_mlflow
        if enable_wandb is not None:
            self._config.enable_wandb_tracking = enable_wandb
        if storage_path is not None:
            self._config.experiment_storage_path = storage_path
        
        # Reconfigure adapters
        self._configure_experiment_tracking_adapters()
        logger.info("Experiment tracking configuration updated")
    
    def configure_model_registry(
        self,
        enable_local: Optional[bool] = None,
        enable_mlflow: Optional[bool] = None,
        enable_s3: Optional[bool] = None,
        storage_path: Optional[str] = None,
        database_url: Optional[str] = None
    ):
        """Reconfigure model registry at runtime.
        
        Args:
            enable_local: Enable local model registry
            enable_mlflow: Enable MLflow model registry
            enable_s3: Enable S3 model storage  
            storage_path: Storage path for models
            database_url: Database URL
        """
        if enable_local is not None:
            self._config.enable_local_model_registry = enable_local
        if enable_mlflow is not None:
            self._config.enable_mlflow_registry = enable_mlflow
        if enable_s3 is not None:
            self._config.enable_s3_model_storage = enable_s3
        if storage_path is not None:
            self._config.model_storage_path = storage_path
        if database_url is not None:
            self._config.database_url = database_url
        
        # Reconfigure adapters
        self._configure_model_registry_adapters()
        logger.info("Model registry configuration updated")
    
    def configure_monitoring(
        self,
        enable_prometheus: Optional[bool] = None,
        enable_datadog: Optional[bool] = None,
        enable_local: Optional[bool] = None,
        monitoring_config: Optional[Dict[str, Any]] = None
    ):
        """Reconfigure monitoring at runtime.
        
        Args:
            enable_prometheus: Enable Prometheus monitoring
            enable_datadog: Enable Datadog monitoring
            enable_local: Enable local monitoring
            monitoring_config: Monitoring configuration
        """
        if enable_prometheus is not None:
            self._config.enable_prometheus_monitoring = enable_prometheus
        if enable_datadog is not None:
            self._config.enable_datadog_monitoring = enable_datadog
        if enable_local is not None:
            self._config.enable_local_monitoring = enable_local
        if monitoring_config is not None:
            self._config.monitoring_config.update(monitoring_config)
        
        # Reconfigure adapters
        self._configure_monitoring_adapters()
        logger.info("Monitoring configuration updated")


# Global container instance
_global_container: Optional[Container] = None


def get_container(config: Optional[ContainerConfig] = None) -> Container:
    """Get the global container instance.
    
    Args:
        config: Optional configuration for new container
        
    Returns:
        Global container instance
    """
    global _global_container
    
    if _global_container is None:
        _global_container = Container(config)
    
    return _global_container


def reset_container():
    """Reset the global container instance."""
    global _global_container
    _global_container = None