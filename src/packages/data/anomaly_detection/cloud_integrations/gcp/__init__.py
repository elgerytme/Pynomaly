"""
Google Cloud Platform Integration for Pynomaly Detection
========================================================

This module provides comprehensive GCP cloud integration including:
- GKE cluster management
- Cloud Storage for data and model persistence
- Cloud Monitoring and Logging
- Cloud Functions for serverless processing
- Cloud AI Platform integration
"""

from .gke_manager import GKEManager
from .storage_manager import GCPStorageManager
from .monitoring_manager import GCPMonitoringManager
from .deployment_manager import GCPDeploymentManager

__all__ = [
    'GKEManager',
    'GCPStorageManager',
    'GCPMonitoringManager', 
    'GCPDeploymentManager'
]

# GCP Integration Version
__version__ = "1.0.0"

def get_gcp_integration_info():
    """Get GCP integration information."""
    return {
        "provider": "Google Cloud Platform",
        "version": __version__,
        "services": {
            "gke": "Google Kubernetes Engine",
            "storage": "Cloud Storage",
            "monitoring": "Cloud Monitoring",
            "logging": "Cloud Logging",
            "functions": "Cloud Functions",
            "ai_platform": "AI Platform"
        },
        "regions": [
            "us-central1", "us-east1", "us-west1",
            "europe-west1", "asia-northeast1"
        ]
    }