"""Data Transformation Package.

A comprehensive data transformation package providing pipeline orchestration,
data cleaning, feature processing, and ETL capabilities.
"""

__version__ = "0.1.0"
__author__ = "Data Transformation Team"
__email__ = "data-transformation@domain.com"

# Application layer exports
from .application.use_cases.data_pipeline import DataPipeline

# Domain layer exports
from .domain.entities.transformation_pipeline import TransformationPipeline
from .domain.services.data_cleaning_service import DataCleaningService
from .domain.value_objects.pipeline_config import PipelineConfig
from .domain.value_objects.transformation_step import TransformationStep

# Infrastructure layer exports
from .infrastructure.adapters.data_source_adapter import DataSourceAdapter
from .infrastructure.processors.feature_processor import FeatureProcessor

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    # Application use cases
    "DataPipeline",
    # Domain entities
    "TransformationPipeline",
    # Domain services
    "DataCleaningService",
    # Domain value objects
    "PipelineConfig",
    "TransformationStep",
    # Infrastructure
    "DataSourceAdapter",
    "FeatureProcessor",
]