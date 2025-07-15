"""
Domain Library Package

A comprehensive domain catalog and logic management system for the Pynomaly ecosystem.
This package provides tools for managing domain entities, business logic templates,
and cross-domain relationships.

Author: Pynomaly Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "team@pynomaly.io"

from .domain.entities import (
    DomainEntity,
    BusinessLogicTemplate,
    BusinessLogicInstance,
    TemplateParameter,
    EntityRelationship,
    RelationshipType,
    CascadeAction,
    RelationshipConstraint,
    DomainCatalog,
    CatalogIndex,
    CatalogStatistics
)

from .application.services import (
    CatalogService,
    EntityService,
    SearchService
)

__all__ = [
    "DomainEntity",
    "BusinessLogicTemplate",
    "BusinessLogicInstance",
    "TemplateParameter",
    "EntityRelationship",
    "RelationshipType", 
    "CascadeAction",
    "RelationshipConstraint",
    "DomainCatalog",
    "CatalogIndex",
    "CatalogStatistics",
    "CatalogService",
    "EntityService",
    "SearchService"
]