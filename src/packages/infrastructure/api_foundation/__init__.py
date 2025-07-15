"""
Infrastructure and API Foundation for Data Science Packages

This package provides the foundational infrastructure, API layer, and
development environment for the data science packages ecosystem.
"""

__version__ = "1.0.0"

from .api_app import create_data_science_api
from .auth_system import AuthenticationSystem, AuthorizationSystem
from .database_infrastructure import DatabaseInfrastructure
from .integration_framework import IntegrationFramework
from .security_framework import SecurityFramework

__all__ = [
    "create_data_science_api",
    "AuthenticationSystem",
    "AuthorizationSystem", 
    "DatabaseInfrastructure",
    "IntegrationFramework",
    "SecurityFramework",
]
