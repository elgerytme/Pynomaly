"""Enterprise MLOps configuration with full enterprise features."""

from pathlib import Path
from typing import Optional, Dict, Any, Type
from uuid import UUID

# Use shared interfaces instead of direct imports
from shared.interfaces.mlops import (
    ExperimentTrackingInterface,
    ModelRegistryInterface,
)
from shared.interfaces.enterprise import (
    AuthServiceInterface,
    MultiTenantInterface,
    OperationsInterface,
)
from infrastructure.dependency_injection import ServiceRegistry
from infrastructure.configuration import get_config

# TODO: Refactor to use proper dependency injection
# These imports violate domain boundaries and should be removed:
# from ....core.ai.machine_learning.mlops.services.experiment_tracking_service import ExperimentTrackingService
# from ....core.ai.machine_learning.mlops.services.model_registry_service import ModelRegistryService
# from ....enterprise.auth import EnterpriseAuthService, SAMLAuthProvider, BasicAuthProvider
# from ....enterprise.multi_tenancy import MultiTenantService, InMemoryTenantStorage
# from ....enterprise.operations import EnterpriseOperationsService, SystemMetricsCollector

# Optional MLOps monorepo integrations - use infrastructure adapters
# try:
#     from ....integrations.mlops import MLflowIntegration
# except ImportError:
#     MLflowIntegration = None

# TODO: Replace with infrastructure integrations
# try:
#     from ....integrations.mlops import KubeflowIntegration  
# except ImportError:
#     KubeflowIntegration = None

# try:
#     from ....integrations.monitoring import DatadogIntegration

# Use infrastructure integrations instead
from infrastructure.integrations import get_integration

try:
    KubeflowIntegration = get_integration("kubeflow")
except ImportError:
    KubeflowIntegration = None

try:
    DatadogIntegration = get_integration("datadog")
except ImportError:
    DatadogIntegration = None


class EnterpriseMLOpsConfiguration:
    """
    Enterprise MLOps configuration with full enterprise features.
    
    Provides enterprise-grade MLOps functionality:
    - Multi-tenant architecture
    - Enterprise authentication (SAML/SSO)
    - Distributed scalability
    - Production monitoring & alerting
    - Audit logging & compliance
    - Platform integrations (MLflow, Kubeflow, etc.)
    """
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        default_tenant_id: Optional[UUID] = None,
        auth_config: Optional[Dict[str, Any]] = None,
        mlflow_config: Optional[Dict[str, Any]] = None,
        kubeflow_config: Optional[Dict[str, Any]] = None,
        monitoring_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize enterprise MLOps configuration."""
        self.data_path = data_path or Path("./enterprise_mlops_data")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.default_tenant_id = default_tenant_id
        self.auth_config = auth_config or {}
        self.mlflow_config = mlflow_config or {}
        self.kubeflow_config = kubeflow_config or {}
        self.monitoring_config = monitoring_config or {}
    
    def create_auth_service(self) -> EnterpriseAuthService:
        """Create enterprise authentication service."""
        auth_type = self.auth_config.get("type", "basic")
        
        if auth_type == "saml":
            auth_provider = SAMLAuthProvider(self.auth_config.get("saml", {}))
        else:
            auth_provider = BasicAuthProvider()
        
        return EnterpriseAuthService(
            auth_provider=auth_provider,
            enable_rbac=self.auth_config.get("enable_rbac", True),
            enable_audit=self.auth_config.get("enable_audit", True)
        )
    
    def create_multi_tenant_service(self) -> MultiTenantService:
        """Create multi-tenancy service."""
        # In production, use proper database storage
        storage = InMemoryTenantStorage()
        
        return MultiTenantService(
            storage=storage,
            default_tenant_id=self.default_tenant_id,
            enable_tenant_isolation=True
        )
    
    def create_operations_service(self) -> EnterpriseOperationsService:
        """Create enterprise operations service."""
        health_providers = []
        metrics_collector = SystemMetricsCollector()
        alert_manager = None
        
        # Add monitoring integrations if available
        if DatadogIntegration and self.monitoring_config.get("datadog"):
            # alert_manager = DatadogAlertManager(self.monitoring_config["datadog"])
            pass
        
        return EnterpriseOperationsService(
            health_providers=health_providers,
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            health_check_interval=self.monitoring_config.get("health_check_interval", 30),
            metrics_collection_interval=self.monitoring_config.get("metrics_interval", 60)
        )
    
    def create_mlflow_integration(self) -> Optional[Any]:
        """Create MLflow integration if configured."""
        if not MLflowIntegration or not self.mlflow_config.get("enabled", False):
            return None
        
        return MLflowIntegration(
            tracking_uri=self.mlflow_config.get("tracking_uri", "http://localhost:5000"),
            artifact_uri=self.mlflow_config.get("artifact_uri"),
            experiment_name=self.mlflow_config.get("experiment_name", "default")
        )
    
    def create_kubeflow_integration(self) -> Optional[Any]:
        """Create Kubeflow integration if configured.""" 
        if not KubeflowIntegration or not self.kubeflow_config.get("enabled", False):
            return None
        
        return KubeflowIntegration(
            host=self.kubeflow_config.get("host", "http://localhost:8080"),
            namespace=self.kubeflow_config.get("namespace", "kubeflow-user")
        )
    
    def create_experiment_tracker(self) -> ExperimentTrackingService:
        """Create enterprise experiment tracking service."""
        experiments_path = self.data_path / "experiments"
        return ExperimentTrackingService(experiments_path)
    
    def create_model_registry(self) -> ModelRegistryService:
        """Create enterprise model registry service."""
        registry_path = self.data_path / "models"
        return ModelRegistryService(registry_path)
    
    def create_mlops_services(self) -> dict:
        """Create all enterprise MLOps services."""
        auth_service = self.create_auth_service()
        tenant_service = self.create_multi_tenant_service()
        operations_service = self.create_operations_service()
        
        # Core MLOps services
        experiment_tracker = self.create_experiment_tracker()
        model_registry = self.create_model_registry()
        
        # Platform integrations
        mlflow = self.create_mlflow_integration()
        kubeflow = self.create_kubeflow_integration()
        
        return {
            # Core MLOps
            "experiment_tracker": experiment_tracker,
            "model_registry": model_registry,
            
            # Enterprise cross-cutting
            "auth": auth_service,
            "multi_tenant": tenant_service,
            "operations": operations_service,
            
            # Platform integrations
            "mlflow": mlflow,
            "kubeflow": kubeflow,
            
            # Configuration
            "config": {
                "enterprise_mode": True,
                "multi_tenant": True,
                "auth_enabled": True,
                "monitoring_enabled": True,
            }
        }
    
    async def start_services(self) -> Dict[str, Any]:
        """Start all enterprise services."""
        services = self.create_mlops_services()
        
        # Start operations monitoring
        if services["operations"]:
            await services["operations"].start()
        
        return services
    
    async def stop_services(self, services: Dict[str, Any]) -> None:
        """Stop all enterprise services."""
        if services.get("operations"):
            await services["operations"].stop()


def create_enterprise_mlops_config(
    data_path: Optional[Path] = None,
    **config_kwargs
) -> EnterpriseMLOpsConfiguration:
    """Factory function for enterprise MLOps configuration."""
    return EnterpriseMLOpsConfiguration(data_path=data_path, **config_kwargs)


# Configuration presets
def create_production_config(data_path: Path) -> EnterpriseMLOpsConfiguration:
    """Create production-ready configuration."""
    return EnterpriseMLOpsConfiguration(
        data_path=data_path,
        auth_config={
            "type": "saml",
            "enable_rbac": True,
            "enable_audit": True,
            "saml": {
                "entity_id": "anomaly_detection-mlops",
                "sso_url": "https://sso.company.com/saml",
            }
        },
        mlflow_config={
            "enabled": True,
            "tracking_uri": "https://mlflow.company.com",
            "artifact_uri": "s3://mlflow-artifacts",
        },
        monitoring_config={
            "health_check_interval": 15,
            "metrics_interval": 30,
            "datadog": {
                "api_key": "your-api-key",
                "app_key": "your-app-key",
            }
        }
    )


def create_development_config(data_path: Path) -> EnterpriseMLOpsConfiguration:
    """Create development configuration with enterprise features."""
    return EnterpriseMLOpsConfiguration(
        data_path=data_path,
        auth_config={
            "type": "basic",
            "enable_rbac": False,
            "enable_audit": False,
        },
        mlflow_config={
            "enabled": True,
            "tracking_uri": "http://localhost:5000",
        },
        monitoring_config={
            "health_check_interval": 60,
            "metrics_interval": 120,
        }
    )


__all__ = [
    "EnterpriseMLOpsConfiguration",
    "create_enterprise_mlops_config",
    "create_production_config",
    "create_development_config"
]