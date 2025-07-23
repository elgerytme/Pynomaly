"""Third-party service integrations.

This module provides integrations with external services like cloud providers,
monitoring services, and business applications. It abstracts vendor-specific
APIs behind common interfaces.

Example usage:
    from infrastructure.integrations import CloudStorageIntegration
    
    storage = CloudStorageIntegration(provider="aws")
    await storage.upload_file("bucket", "key", file_data)
"""

from .cloud_storage import CloudStorageIntegration
from .monitoring_services import MonitoringIntegration  
from .notification_services import NotificationIntegration

__all__ = [
    "CloudStorageIntegration",
    "MonitoringIntegration",
    "NotificationIntegration"
]