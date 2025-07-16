"""
Data Observability Dependency Injection Container

Configures and provides dependency injection for data observability services.
"""

from dependency_injector import containers, providers

from ...application.services.data_lineage_service import DataLineageService
from ...application.services.pipeline_health_service import PipelineHealthService
from ...application.services.data_catalog_service import DataCatalogService
from ...application.services.predictive_quality_service import PredictiveQualityService


class DataObservabilityContainer(containers.DeclarativeContainer):
    """Dependency injection container for data observability package."""
    
    # Configuration
    config = providers.Configuration()
    
    # Core services
    data_lineage_service = providers.Singleton(DataLineageService)
    
    pipeline_health_service = providers.Singleton(PipelineHealthService)
    
    data_catalog_service = providers.Singleton(DataCatalogService)
    
    predictive_quality_service = providers.Singleton(PredictiveQualityService)
    
    # Composed services that depend on core services
    observability_facade = providers.Factory(
        "monorepo.packages.data_observability.application.facades.observability_facade.DataObservabilityFacade",
        lineage_service=data_lineage_service,
        health_service=pipeline_health_service,
        catalog_service=data_catalog_service,
        quality_service=predictive_quality_service
    )