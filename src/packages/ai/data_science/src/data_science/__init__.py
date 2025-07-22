"""Data Science Package.

A comprehensive data science package providing experiment management,
feature validation, metrics calculation, and workflow orchestration.
"""

__version__ = "0.1.0"
__author__ = "Data Science Team"
__email__ = "data-science@domain.com"

# Application layer exports
from .application.services.integrated_data_science_service import IntegratedDataScienceService
from .application.services.performance_degradation_monitoring_service import PerformanceDegradationMonitoringService
from .application.services.workflow_orchestration_engine import WorkflowOrchestrationEngine

# Domain layer exports
from .domain.entities.dataset import Dataset
from .domain.entities.lineage_record import LineageRecord
from .domain.entities.pipeline import Pipeline

from .domain.services.feature_validator import FeatureValidator
from .domain.services.metrics_calculator import MetricsCalculator
from .domain.services.processing_orchestrator import ProcessingOrchestrator

from .domain.value_objects.confidence_interval import ConfidenceInterval

# Infrastructure layer exports
from .infrastructure.config.data_science_settings import DataScienceSettings
from .infrastructure.di.container import Container

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    # Application services
    "IntegratedDataScienceService",
    "PerformanceDegradationMonitoringService", 
    "WorkflowOrchestrationEngine",
    # Domain entities
    "Dataset",
    "LineageRecord",
    "Pipeline",
    # Domain services
    "FeatureValidator",
    "MetricsCalculator",
    "ProcessingOrchestrator",
    # Domain value objects
    "ConfidenceInterval",
    # Infrastructure
    "DataScienceSettings",
    "Container",
]