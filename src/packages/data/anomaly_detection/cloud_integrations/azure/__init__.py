"""
Microsoft Azure Integration for Pynomaly Detection
===================================================

This module provides comprehensive Azure cloud integration including:
- AKS cluster management
- Azure Storage for data and model persistence
- Azure Monitor and Log Analytics
- Azure Functions for serverless processing
- Azure Machine Learning integration
"""

from .aks_manager import AKSManager
from .storage_manager import AzureStorageManager
from .monitor_manager import AzureMonitorManager
from .deployment_manager import AzureDeploymentManager

__all__ = [
    'AKSManager',
    'AzureStorageManager',
    'AzureMonitorManager',
    'AzureDeploymentManager'
]

# Azure Integration Version
__version__ = "1.0.0"

def get_azure_integration_info():
    """Get Azure integration information."""
    return {
        "provider": "Microsoft Azure",
        "version": __version__,
        "services": {
            "aks": "Azure Kubernetes Service",
            "storage": "Azure Storage",
            "monitor": "Azure Monitor",
            "log_analytics": "Log Analytics",
            "functions": "Azure Functions",
            "ml": "Azure Machine Learning"
        },
        "regions": [
            "eastus", "westus2", "centralus",
            "westeurope", "northeurope", "eastasia"
        ]
    }