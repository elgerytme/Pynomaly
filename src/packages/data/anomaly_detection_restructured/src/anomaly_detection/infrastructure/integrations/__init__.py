"""
Ecosystem Development for Pynomaly Detection
============================================

This module provides comprehensive ecosystem features including:
- Plugin architecture and marketplace
- Third-party integrations framework
- Community features and collaboration tools
- Developer SDK and API documentation
- Extension registry and management
- Webhook and event system
"""

from .plugin_system import PluginManager, PluginRegistry, PluginMarketplace
from .integrations import IntegrationFramework, ThirdPartyConnector
from .community import CommunityHub, CollaborationTools
from .developer_sdk import DeveloperSDK, APIDocumentation
from .extensions import ExtensionRegistry, ExtensionManager
from .webhooks import WebhookManager, EventSystem

__all__ = [
    # Plugin System
    'PluginManager',
    'PluginRegistry',
    'PluginMarketplace',
    
    # Integrations
    'IntegrationFramework',
    'ThirdPartyConnector',
    
    # Community
    'CommunityHub',
    'CollaborationTools',
    
    # Developer Tools
    'DeveloperSDK',
    'APIDocumentation',
    
    # Extensions
    'ExtensionRegistry',
    'ExtensionManager',
    
    # Webhooks and Events
    'WebhookManager',
    'EventSystem'
]

# Ecosystem Development Version
__version__ = "1.0.0"

def get_ecosystem_info():
    """Get ecosystem development information."""
    return {
        "version": __version__,
        "capabilities": {
            "plugin_system": "Extensible plugin architecture with marketplace",
            "integrations": "Third-party system integration framework",
            "community": "Collaboration tools and community features",
            "developer_sdk": "Comprehensive SDK and API documentation",
            "extensions": "Registry and management for platform extensions",
            "webhooks": "Event-driven webhook and notification system"
        },
        "supported_platforms": [
            "Python", "REST API", "GraphQL", "gRPC", "WebSocket"
        ],
        "integration_types": [
            "Data Sources", "ML Frameworks", "Monitoring Tools", 
            "Cloud Services", "Databases", "Message Queues"
        ]
    }