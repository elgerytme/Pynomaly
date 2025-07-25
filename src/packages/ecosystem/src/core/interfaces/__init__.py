"""
Core integration interfaces for the ecosystem framework.

This module defines the fundamental contracts and interfaces that all
ecosystem integrations must implement, ensuring consistency and
interoperability across the platform.
"""

from .integration_interface import (
    IntegrationInterface,
    IntegrationConfig,
    IntegrationStatus,
    ConnectionHealth,
    DataFlowCapability,
    AuthenticationMethod
)

from .partner_interface import (
    PartnerInterface,
    PartnerTier,
    PartnerCapability,
    PartnerMetrics,
    PartnerContract
)

from .data_interface import (
    DataConnectorInterface,
    DataSchema,
    DataFormat,
    DataTransferMode,
    DataValidationResult
)

from .event_interface import (
    EventInterface,
    EventType,
    EventPriority,
    EventHandler,
    EventSubscription
)

__all__ = [
    # Integration interfaces
    "IntegrationInterface",
    "IntegrationConfig", 
    "IntegrationStatus",
    "ConnectionHealth",
    "DataFlowCapability",
    "AuthenticationMethod",
    
    # Partner interfaces
    "PartnerInterface",
    "PartnerTier",
    "PartnerCapability", 
    "PartnerMetrics",
    "PartnerContract",
    
    # Data interfaces
    "DataConnectorInterface",
    "DataSchema",
    "DataFormat",
    "DataTransferMode",
    "DataValidationResult",
    
    # Event interfaces
    "EventInterface",
    "EventType",
    "EventPriority",
    "EventHandler",
    "EventSubscription"
]