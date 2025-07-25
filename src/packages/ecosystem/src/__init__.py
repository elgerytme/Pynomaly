"""
Ecosystem Integration Framework

A comprehensive framework for managing partnerships and integrations
across the MLOps platform ecosystem.
"""

from .core.interfaces import (
    IntegrationInterface,
    IntegrationConfig,
    IntegrationStatus,
    ConnectionHealth,
    PartnerInterface,
    PartnerTier,
    PartnerCapability,
    PartnerContract,
    PartnerMetrics,
    DataConnectorInterface,
    DataSchema,
    DataFormat,
    EventInterface,
    Event,
    EventType,
    EventPriority
)

from .management.registry import PartnerRegistry

# Version info
__version__ = "1.0.0"
__author__ = "MLOps Platform Team"
__email__ = "platform@company.com"

# Available connectors (imported dynamically to avoid dependency issues)
AVAILABLE_CONNECTORS = {
    "databricks": "ecosystem.connectors.databricks.DatabricksIntegration",
    "snowflake": "ecosystem.connectors.snowflake.SnowflakeIntegration",
    "mlflow": "ecosystem.connectors.mlflow.MLflowIntegration",
    "kubeflow": "ecosystem.connectors.kubeflow.KubeflowIntegration",
    "datadog": "ecosystem.connectors.datadog.DatadogIntegration"
}

# Available templates
AVAILABLE_TEMPLATES = {
    "mlops_platform": "ecosystem.templates.mlops_platform.MLOpsPlatformTemplate",
    "data_platform": "ecosystem.templates.data_platform.DataPlatformTemplate",
    "monitoring_platform": "ecosystem.templates.monitoring_platform.MonitoringPlatformTemplate",
    "cloud_provider": "ecosystem.templates.cloud_provider.CloudProviderTemplate"
}

__all__ = [
    # Core interfaces
    "IntegrationInterface",
    "IntegrationConfig", 
    "IntegrationStatus",
    "ConnectionHealth",
    "PartnerInterface",
    "PartnerTier",
    "PartnerCapability",
    "PartnerContract",
    "PartnerMetrics",
    "DataConnectorInterface",
    "DataSchema",
    "DataFormat",
    "EventInterface",
    "Event",
    "EventType",
    "EventPriority",
    
    # Management
    "PartnerRegistry",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "AVAILABLE_CONNECTORS",
    "AVAILABLE_TEMPLATES"
]


def get_connector(connector_name: str):
    """
    Dynamically import and return a connector class.
    
    Args:
        connector_name: Name of the connector to import
        
    Returns:
        Connector class
        
    Raises:
        ImportError: If connector is not available
    """
    if connector_name not in AVAILABLE_CONNECTORS:
        raise ImportError(f"Connector '{connector_name}' not available")
    
    module_path = AVAILABLE_CONNECTORS[connector_name]
    module_name, class_name = module_path.rsplit(".", 1)
    
    try:
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Failed to import connector '{connector_name}': {e}")


def get_template(template_name: str):
    """
    Dynamically import and return a template class.
    
    Args:
        template_name: Name of the template to import
        
    Returns:
        Template class
        
    Raises:
        ImportError: If template is not available
    """
    if template_name not in AVAILABLE_TEMPLATES:
        raise ImportError(f"Template '{template_name}' not available")
    
    module_path = AVAILABLE_TEMPLATES[template_name]
    module_name, class_name = module_path.rsplit(".", 1)
    
    try:
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Failed to import template '{template_name}': {e}")


def list_available_connectors():
    """List all available connectors."""
    return list(AVAILABLE_CONNECTORS.keys())


def list_available_templates():
    """List all available templates."""
    return list(AVAILABLE_TEMPLATES.keys())