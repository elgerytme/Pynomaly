"""Monitoring service initialization module."""

from __future__ import annotations

import logging
from typing import Optional

from pynomaly.infrastructure.config.settings import get_settings, Settings
from pynomaly.infrastructure.monitoring.external_monitoring_service import (
    ExternalMonitoringService,
    MonitoringConfiguration,
    MonitoringProvider,
)

logger = logging.getLogger(__name__)


def create_monitoring_service(settings: Optional[Settings] = None) -> ExternalMonitoringService:
    """Create and configure the external monitoring service."""
    if settings is None:
        settings = get_settings()
    
    # Create monitoring service
    service = ExternalMonitoringService()
    
    # Get monitoring configuration
    monitoring_config = settings.get_monitoring_config()
    
    # Set buffer size and flush interval
    service.buffer_size = monitoring_config["buffer_size"]
    service.flush_interval = monitoring_config["flush_interval"]
    
    # Add configured providers
    providers = monitoring_config["providers"]
    for provider_config in providers:
        try:
            # Create monitoring configuration
            config = MonitoringConfiguration(
                provider=MonitoringProvider(provider_config["provider"]),
                endpoint_url=provider_config["endpoint"],
                api_key=provider_config["api_key"],
                enabled=provider_config["enabled"]
            )
            
            # Add provider to service
            provider_name = f"{provider_config['provider']}_provider"
            service.add_provider(provider_name, config)
            
            logger.info(f"Added monitoring provider: {provider_name}")
            
        except Exception as e:
            logger.error(f"Failed to add monitoring provider {provider_config['provider']}: {e}")
            continue
    
    return service


async def initialize_monitoring_service(settings: Optional[Settings] = None) -> ExternalMonitoringService:
    """Initialize the external monitoring service."""
    service = create_monitoring_service(settings)
    await service.initialize()
    return service


async def shutdown_monitoring_service(service: ExternalMonitoringService) -> None:
    """Shutdown the external monitoring service."""
    await service.shutdown()
