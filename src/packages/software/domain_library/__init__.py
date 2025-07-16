"""
Domain Library Package

A comprehensive domain catalog and logic management system for the Monorepo ecosystem.
This package provides tools for managing domain entities, business logic templates,
and cross-domain relationships.

Author: Monorepo Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Monorepo Team"
__email__ = "team@monorepo.io"

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