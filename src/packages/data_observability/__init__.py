"""
Data Observability Package

This package provides comprehensive data observability capabilities for the Pynomaly platform,
including data lineage tracking, pipeline health monitoring, data catalog management,
and predictive data quality services.
"""

from .domain.entities.data_lineage import DataLineage, LineageNode, LineageEdge
from .domain.entities.pipeline_health import PipelineHealth, PipelineMetric, PipelineAlert
from .domain.entities.data_catalog import DataCatalogEntry, DataSchema, DataUsage
from .domain.entities.quality_prediction import QualityPrediction, QualityTrend, QualityForecast

from .application.services.data_lineage_service import DataLineageService
from .application.services.pipeline_health_service import PipelineHealthService
from .application.services.data_catalog_service import DataCatalogService
from .application.services.predictive_quality_service import PredictiveQualityService

from .application.facades.observability_facade import DataObservabilityFacade

from .infrastructure.di.container import DataObservabilityContainer

__version__ = "1.0.0"
__all__ = [
    # Facades
    "DataObservabilityFacade",
    
    # Domain entities
    "DataLineage",
    "LineageNode", 
    "LineageEdge",
    "PipelineHealth",
    "PipelineMetric",
    "PipelineAlert",
    "DataCatalogEntry",
    "DataSchema",
    "DataUsage",
    "QualityPrediction",
    "QualityTrend",
    "QualityForecast",
    
    # Application services
    "DataLineageService",
    "PipelineHealthService", 
    "DataCatalogService",
    "PredictiveQualityService",
    
    # Infrastructure
    "DataObservabilityContainer"
]