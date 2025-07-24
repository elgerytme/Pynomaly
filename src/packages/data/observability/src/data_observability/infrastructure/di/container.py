"""
Data Observability Dependency Injection Container

Configures and provides dependency injection for data observability services.
"""

from dependency_injector import containers, providers

from ...application.services.data_lineage_service import DataLineageService
from ...application.services.pipeline_health_service import PipelineHealthService
from ...application.services.data_catalog_service import DataCatalogService
from ...application.services.predictive_quality_service import PredictiveQualityService
from ...application.facades.observability_facade import DataObservabilityFacade

from ...infrastructure.persistence.database import get_db_session
from ...infrastructure.repositories.postgres_data_catalog_repository import PostgresDataCatalogRepository
from ...infrastructure.repositories.postgres_data_lineage_repository import PostgresDataLineageRepository
from ...infrastructure.repositories.postgres_pipeline_health_repository import PostgresPipelineHealthRepository
from ...infrastructure.repositories.postgres_quality_prediction_repository import PostgresQualityPredictionRepository


class DataObservabilityContainer(containers.DeclarativeContainer):
    """Dependency injection container for data observability package."""
    
    # Configuration
    config = providers.Configuration()
    
    # Database session provider
    db_session = providers.Resource(get_db_session)

    # Repositories
    data_catalog_repository = providers.Singleton(
        PostgresDataCatalogRepository,
        session=db_session
    )
    data_lineage_repository = providers.Singleton(
        PostgresDataLineageRepository,
        session=db_session
    )
    pipeline_health_repository = providers.Singleton(
        PostgresPipelineHealthRepository,
        session=db_session
    )
    quality_prediction_repository = providers.Singleton(
        PostgresQualityPredictionRepository,
        session=db_session
    )
    
    # Core services
    data_lineage_service = providers.Singleton(
        DataLineageService,
        repository=data_lineage_repository
    )
    
    pipeline_health_service = providers.Singleton(
        PipelineHealthService,
        repository=pipeline_health_repository
    )
    
    data_catalog_service = providers.Singleton(
        DataCatalogService,
        repository=data_catalog_repository
    )
    
    predictive_quality_service = providers.Singleton(
        PredictiveQualityService,
        repository=quality_prediction_repository
    )
    
    # Composed services that depend on core services
    observability_facade = providers.Factory(
        DataObservabilityFacade,
        lineage_service=data_lineage_service,
        health_service=pipeline_health_service,
        catalog_service=data_catalog_service,
        quality_service=predictive_quality_service
    )